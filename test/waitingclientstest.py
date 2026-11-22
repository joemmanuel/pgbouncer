"""Integration contract for the proposed PgBouncer max_waiting_clients setting.

Copy into a configured PgBouncer checkout's test/ directory. Requires that
checkout's conftest.py, bouncer/pg fixtures, psycopg 3, and pytest-asyncio setup.
See max_waiting_clients.md for the contract and execution instructions.
"""

import asyncio
import configparser
import io
import re
from contextlib import asynccontextmanager

import psycopg
import pytest
from psycopg.rows import dict_row


DATABASE_NAME = "waiting_limit"
OTHER_DATABASE_NAME = "waiting_other"
USER_NAME = "muser1"
WAIT_SECONDS = 10
POLL_SECONDS = 0.02
UNSET = None


def build_configuration(
    original_configuration: str,
    *,
    global_limit=UNSET,
    database_limit=UNSET,
    user_limit=UNSET,
    pool_mode="transaction",
) -> str:
    configuration = configparser.ConfigParser(interpolation=None, strict=False)
    configuration.read_string(original_configuration)
    backend_configuration = configuration["databases"]["p0"]
    backend_configuration = re.sub(
        r"\b(?:user|pool_size|reserve_pool_size|reserve_pool|pool_mode)\s*=\s*\S+",
        "",
        backend_configuration,
    )
    backend_configuration += f" user={USER_NAME} pool_size=1 reserve_pool_size=0"
    configuration["databases"][OTHER_DATABASE_NAME] = backend_configuration
    if database_limit is not UNSET:
        backend_configuration += f" max_waiting_clients={database_limit}"
    configuration["databases"][DATABASE_NAME] = backend_configuration

    if not configuration.has_section("users"):
        configuration.add_section("users")
    user_configuration = "pool_size=1 reserve_pool_size=0"
    if user_limit is not UNSET:
        user_configuration += f" max_waiting_clients={user_limit}"
    configuration["users"][USER_NAME] = user_configuration

    settings = configuration["pgbouncer"]
    settings.update(
        {
            "pool_mode": pool_mode,
            "default_pool_size": "1",
            "reserve_pool_size": "0",
            "max_client_conn": "100",
            "query_wait_timeout": "0",
            "query_timeout": "0",
            "idle_transaction_timeout": "0",
            "transaction_timeout": "0",
            "client_idle_timeout": "0",
        }
    )
    settings.pop("max_waiting_clients", None)
    if global_limit is not UNSET:
        settings["max_waiting_clients"] = str(global_limit)
    output = io.StringIO()
    configuration.write(output)
    return output.getvalue()


class WaitingPool:
    def __init__(self, bouncer, database_name: str) -> None:
        self.bouncer = bouncer
        self.database_name = database_name
        self.connections = []
        self.tasks = []
        self.holder = None

    async def connect(self):
        connection = await self.bouncer.aconn(
            dbname=self.database_name,
            user=USER_NAME,
            connect_timeout=WAIT_SECONDS,
        )
        self.connections.append(connection)
        return connection

    async def occupy_backend(self) -> None:
        self.holder = await self.connect()
        await self.holder.execute("BEGIN")
        await self.holder.execute("SELECT 1")

    async def release_backend(self) -> None:
        if self.holder is not None:
            await self.holder.close()
            self.holder = None

    def waiting_count(self) -> int:
        pools = self.bouncer.admin("SHOW POOLS", row_factory=dict_row)
        return sum(
            int(pool["cl_waiting"])
            for pool in pools
            if pool["database"] == self.database_name
            and pool["user"] == USER_NAME
        )

    async def wait_for_count(self, expected_count: int) -> None:
        async def poll() -> None:
            while self.waiting_count() != expected_count:
                await asyncio.sleep(POLL_SECONDS)

        try:
            await asyncio.wait_for(poll(), timeout=WAIT_SECONDS)
        except TimeoutError:
            pytest.fail(
                f"Expected {expected_count} waiters in {self.database_name}; "
                f"observed {self.waiting_count()}"
            )

    async def start_request(self, query="SELECT 42", parameters=None):
        connection = await self.connect()
        task = asyncio.create_task(connection.execute(query, parameters))
        self.tasks.append(task)
        return connection, task

    async def enqueue(self, count: int) -> list:
        initial_count = self.waiting_count()
        requests = []
        for offset in range(count):
            request = await self.start_request()
            requests.append(request)
            await self.wait_for_count(initial_count + offset + 1)
            assert not request[1].done()
        return requests

    async def assert_rejected(self, query="SELECT 42", parameters=None) -> None:
        connection, task = await self.start_request(query, parameters)
        with pytest.raises(psycopg.OperationalError, match="max_waiting_clients"):
            await asyncio.wait_for(asyncio.shield(task), timeout=WAIT_SECONDS)
        assert connection.closed

    async def assert_completed(self, requests: list) -> None:
        for connection, task in requests:
            cursor = await asyncio.wait_for(asyncio.shield(task), timeout=WAIT_SECONDS)
            assert await cursor.fetchall() == [(42,)]
            await connection.close()
        await self.wait_for_count(0)

    async def close(self) -> None:
        for connection in reversed(self.connections):
            await connection.close()
        for task in self.tasks:
            if not task.done():
                task.cancel()
        if self.tasks:
            await asyncio.wait_for(
                asyncio.gather(*self.tasks, return_exceptions=True),
                timeout=WAIT_SECONDS,
            )


@pytest.fixture
def waiting_configuration(bouncer):
    original_configuration = bouncer.ini_path.read_text()
    settings = bouncer.admin("SHOW CONFIG", row_factory=dict_row)
    setting = next(
        (entry for entry in settings if entry["key"] == "max_waiting_clients"),
        None,
    )
    assert setting is not None, "Implement max_waiting_clients before running this suite"
    assert str(setting["default"]) == "0"

    @asynccontextmanager
    async def configure(**options):
        configuration = build_configuration(original_configuration, **options)
        with bouncer.run_with_config(configuration):
            pools = [
                WaitingPool(bouncer, DATABASE_NAME),
                WaitingPool(bouncer, OTHER_DATABASE_NAME),
            ]
            try:
                yield pools
            finally:
                for pool in pools:
                    await pool.close()

    return configure


@pytest.mark.parametrize("limit", [1, 2, 5])
@pytest.mark.parametrize("pool_mode", ["transaction", "session"])
async def test_exact_capacity_and_overflow(waiting_configuration, limit, pool_mode):
    async with waiting_configuration(global_limit=limit, pool_mode=pool_mode) as pools:
        pool = pools[0]
        await pool.occupy_backend()
        admitted_requests = await pool.enqueue(limit)
        for attempt in range(3):
            await pool.assert_rejected()
            assert pool.waiting_count() == limit
            assert all(not task.done() for connection, task in admitted_requests)
        await pool.release_backend()
        await pool.assert_completed(admitted_requests)


@pytest.mark.parametrize("global_limit", [UNSET, 0])
async def test_zero_and_omitted_limit_are_unlimited(waiting_configuration, global_limit):
    async with waiting_configuration(global_limit=global_limit) as pools:
        pool = pools[0]
        await pool.occupy_backend()
        admitted_requests = await pool.enqueue(8)
        assert pool.waiting_count() == 8
        await pool.release_backend()
        await pool.assert_completed(admitted_requests)


@pytest.mark.parametrize(
    ("global_limit", "database_limit", "user_limit", "expected_limit"),
    [
        (2, UNSET, UNSET, 2),
        (1, 3, UNSET, 3),
        (3, 1, UNSET, 1),
        (1, UNSET, 3, 3),
        (3, UNSET, 1, 1),
        (5, 1, 3, 3),
        (5, 3, 1, 1),
        (0, 2, UNSET, 2),
        (0, 0, 2, 2),
        (1, 0, UNSET, 0),
        (1, 2, 0, 0),
    ],
)
async def test_override_precedence(
    waiting_configuration, global_limit, database_limit, user_limit, expected_limit
):
    async with waiting_configuration(
        global_limit=global_limit,
        database_limit=database_limit,
        user_limit=user_limit,
    ) as pools:
        pool = pools[0]
        await pool.occupy_backend()
        admitted_requests = await pool.enqueue(expected_limit or 8)
        if expected_limit:
            await pool.assert_rejected()
            assert pool.waiting_count() == expected_limit
        await pool.release_backend()
        await pool.assert_completed(admitted_requests)


async def test_limit_is_per_database_pool(waiting_configuration):
    async with waiting_configuration(global_limit=2) as pools:
        requests_by_pool = []
        for pool in pools:
            await pool.occupy_backend()
            requests_by_pool.append(await pool.enqueue(2))
        assert sum(pool.waiting_count() for pool in pools) == 4
        for pool, requests in zip(pools, requests_by_pool):
            await pool.assert_rejected()
            await pool.release_backend()
            await pool.assert_completed(requests)


async def test_disconnect_reclaims_capacity(waiting_configuration):
    async with waiting_configuration(global_limit=2) as pools:
        pool = pools[0]
        await pool.occupy_backend()
        first_request, second_request = await pool.enqueue(2)
        await first_request[0].close()
        await pool.wait_for_count(1)
        replacement_requests = await pool.enqueue(1)
        await pool.assert_rejected()
        await pool.release_backend()
        await pool.assert_completed([second_request, *replacement_requests])


async def test_timeout_reclaims_capacity(waiting_configuration, bouncer):
    async with waiting_configuration(global_limit=1) as pools:
        pool = pools[0]
        await pool.occupy_backend()
        bouncer.admin("SET query_wait_timeout = 1")
        requests = await pool.enqueue(1)
        with pytest.raises(psycopg.OperationalError, match="query_wait_timeout"):
            await asyncio.wait_for(asyncio.shield(requests[0][1]), timeout=WAIT_SECONDS)
        await pool.wait_for_count(0)
        bouncer.admin("SET query_wait_timeout = 0")
        replacement_requests = await pool.enqueue(1)
        await pool.assert_rejected()
        await pool.release_backend()
        await pool.assert_completed(replacement_requests)


@pytest.mark.parametrize(("initial_limit", "new_limit", "extra_count"), [(1, 3, 2), (1, 0, 6), (0, 0, 3)])
async def test_runtime_increase_or_disable(
    waiting_configuration, bouncer, initial_limit, new_limit, extra_count
):
    async with waiting_configuration(global_limit=initial_limit) as pools:
        pool = pools[0]
        await pool.occupy_backend()
        admitted_requests = await pool.enqueue(1)
        bouncer.admin(f"SET max_waiting_clients = {new_limit}")
        admitted_requests.extend(await pool.enqueue(extra_count))
        if new_limit:
            await pool.assert_rejected()
        await pool.release_backend()
        await pool.assert_completed(admitted_requests)


@pytest.mark.parametrize("initial_limit", [0, 4])
async def test_reload_decrease_preserves_existing_waiters(
    waiting_configuration, bouncer, initial_limit
):
    async with waiting_configuration(global_limit=initial_limit) as pools:
        pool = pools[0]
        await pool.occupy_backend()
        admitted_requests = await pool.enqueue(4)
        configuration = configparser.ConfigParser(interpolation=None, strict=False)
        configuration.read(bouncer.ini_path)
        configuration["pgbouncer"]["max_waiting_clients"] = "2"
        with bouncer.ini_path.open("w") as configuration_file:
            configuration.write(configuration_file)
        bouncer.admin("RELOAD")
        assert pool.waiting_count() == 4
        await pool.assert_rejected()
        assert all(not task.done() for connection, task in admitted_requests)
        for connection, task in admitted_requests[:2]:
            await connection.close()
        await pool.wait_for_count(2)
        await pool.assert_rejected()
        await admitted_requests[2][0].close()
        await pool.wait_for_count(1)
        replacement_requests = await pool.enqueue(1)
        await pool.assert_rejected()
        await pool.release_backend()
        await pool.assert_completed([admitted_requests[3], *replacement_requests])


@pytest.mark.parametrize("invalid_value", ["-1", "-2", "2147483648", "-2147483649", "1.5", "abc", "2junk", ""])
def test_invalid_runtime_value_preserves_previous_setting(bouncer, invalid_value):
    bouncer.admin("SET max_waiting_clients = 2")
    with pytest.raises(psycopg.Error):
        bouncer.admin(f"SET max_waiting_clients = '{invalid_value}'")
    settings = bouncer.admin("SHOW CONFIG", row_factory=dict_row)
    setting = next(entry for entry in settings if entry["key"] == "max_waiting_clients")
    assert setting["value"] == "2"


def test_largest_supported_integer(bouncer):
    bouncer.admin("SET max_waiting_clients = 2147483647")
    settings = bouncer.admin("SHOW CONFIG", row_factory=dict_row)
    setting = next(entry for entry in settings if entry["key"] == "max_waiting_clients")
    assert setting["value"] == "2147483647"


async def test_rejected_request_does_not_execute(waiting_configuration, pg):
    async with waiting_configuration(global_limit=1) as pools:
        pool = pools[0]
        sequence_name = "waiting_limit_rejection_sequence"
        pg.sql(f"CREATE SEQUENCE {sequence_name}", dbname="p0")
        pg.sql(f"GRANT USAGE ON SEQUENCE {sequence_name} TO {USER_NAME}", dbname="p0")
        try:
            await pool.occupy_backend()
            admitted_requests = await pool.enqueue(1)
            await pool.assert_rejected(f"SELECT nextval('{sequence_name}')")
            await pool.release_backend()
            await pool.assert_completed(admitted_requests)
            assert pg.sql_value(f"SELECT is_called FROM {sequence_name}", dbname="p0") is False
        finally:
            pg.sql(f"DROP SEQUENCE {sequence_name}", dbname="p0")


async def test_concurrent_burst_is_bounded(waiting_configuration):
    async with waiting_configuration(global_limit=3) as pools:
        pool = pools[0]
        connections = [await pool.connect() for position in range(12)]
        await pool.occupy_backend()
        requests = [
            (connection, asyncio.create_task(connection.execute("SELECT 42")))
            for connection in connections
        ]
        pool.tasks.extend(task for connection, task in requests)

        async def wait_for_rejections() -> None:
            while sum(task.done() for connection, task in requests) < 9:
                assert pool.waiting_count() <= 3
                await asyncio.sleep(POLL_SECONDS)

        await asyncio.wait_for(wait_for_rejections(), timeout=WAIT_SECONDS)
        assert pool.waiting_count() == 3
        admitted_requests = []
        for connection, task in requests:
            if task.done():
                with pytest.raises(psycopg.OperationalError, match="max_waiting_clients"):
                    await task
            else:
                admitted_requests.append((connection, task))
        assert len(admitted_requests) == 3
        await pool.release_backend()
        await pool.assert_completed(admitted_requests)
