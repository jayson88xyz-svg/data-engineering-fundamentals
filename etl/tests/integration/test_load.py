"""Run against the FIXTURE database only:
    uv run --locked --env-file ../.env.fixture python -m pytest -q tests/integration
Every test rolls back, so the fixture database is left as it was found.
"""
from datetime import datetime
from decimal import Decimal

import polars as pl
import psycopg
import pytest

from extract import Zone
from load import load_trips, load_zones
from transform import clean_trips


def tiny_trips() -> pl.DataFrame:
    start = datetime(2024, 1, 2, 12)
    end = datetime(2024, 1, 2, 12, 15)
    source = pl.DataFrame(
        {
            "VendorID": [1, 2, 1],
            "tpep_pickup_datetime": [start, start, start],
            "tpep_dropoff_datetime": [end, end, end],
            "passenger_count": [None, 1.0, 2.0],
            "trip_distance": [0.0, None, 3.25],
            "PULocationID": [999, 1, 2],
            "DOLocationID": [1, 1, 1],
            "payment_type": [1, 2, 1],
            "fare_amount": [-1.23, 2.37, 14.1],
            "tip_amount": [None, 0.2, 3.0],
            "total_amount": [-1.23, 2.57, 20.5],
        }
    )
    return clean_trips(source)


@pytest.fixture
def conn():
    with psycopg.connect(connect_timeout=5) as connection:
        yield connection
        connection.rollback()


def count(conn, table: str) -> int:
    return conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]


def test_copy_round_trips_values_and_nulls(conn):
    before = count(conn, "raw.yellow_taxi_trips")
    inserted = load_trips(conn, tiny_trips())
    assert inserted == 3
    assert count(conn, "raw.yellow_taxi_trips") == before + 3

    rows = conn.execute(
        """
        SELECT vendor_id, pickup_datetime, passenger_count, trip_distance,
               pickup_location_id, fare_amount, tip_amount
        FROM raw.yellow_taxi_trips ORDER BY trip_id DESC LIMIT 3
        """
    ).fetchall()
    rows.reverse()
    assert rows[0] == (1, datetime(2024, 1, 2, 12), None, 0.0, 999, Decimal("-1.23"), None)
    assert rows[1][3] is None and rows[1][6] == Decimal("0.20")


def test_running_the_same_trips_twice_appends_twice(conn):
    before = count(conn, "raw.yellow_taxi_trips")
    load_trips(conn, tiny_trips())
    load_trips(conn, tiny_trips())
    assert count(conn, "raw.yellow_taxi_trips") == before + 6


def test_failed_copy_rolls_back_everything_in_the_transaction(conn):
    before = count(conn, "raw.yellow_taxi_trips")
    broken = tiny_trips().with_columns(
        pl.when(pl.int_range(pl.len()) == 2)
        .then(None)
        .otherwise(pl.col("pickup_datetime"))
        .alias("pickup_datetime")  # NOT NULL column, so COPY must fail
    )
    with pytest.raises(psycopg.errors.NotNullViolation):
        with conn.transaction():
            load_trips(conn, tiny_trips())  # succeeds inside the transaction...
            load_trips(conn, broken)  # ...then this one blows up
    assert count(conn, "raw.yellow_taxi_trips") == before


def test_zone_load_can_see_the_same_keys_again(conn):
    zones = [Zone(900001, "Test", "Zone A", "Yellow Zone"), Zone(900002, "Test", "Zone B", None)]
    first = load_zones(conn, zones)
    second = load_zones(conn, zones)
    assert (first, second) == (2, 0)
    assert conn.execute(
        "SELECT count(*) FROM raw.taxi_zones WHERE location_id IN (900001, 900002)"
    ).fetchone()[0] == 2
