import polars as pl
from psycopg import Connection

from extract import Zone
from transform import TRIP_COLUMNS

CHUNK_ROWS = 200_000
_TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S%.f"


def load_trips(conn: Connection, trips: pl.DataFrame) -> int:
    """Append accepted trips with COPY and return the number written.

    The caller owns the transaction: nothing here commits. If COPY fails
    part-way, the caller's rollback removes every row from this call.
    """
    # Explicit column order: the DataFrame is re-selected to match the COPY list.
    ordered = trips.select(TRIP_COLUMNS)
    column_list = ", ".join(TRIP_COLUMNS)
    statement = (
        f"COPY raw.yellow_taxi_trips ({column_list}) FROM STDIN WITH (FORMAT csv)"
    )
    with conn.cursor() as cur, cur.copy(statement) as copy:
        for chunk in ordered.iter_slices(n_rows=CHUNK_ROWS):
            # Nulls become empty, unquoted fields, which CSV COPY reads as NULL.
            copy.write(
                chunk.write_csv(
                    include_header=False, datetime_format=_TIMESTAMP_FORMAT
                ).encode()
            )
    return ordered.height


def load_zones(conn: Connection, zones: list[Zone]) -> int:
    """Insert zones keyed by location_id; keys already present are skipped.

    Safe to run repeatedly. Returns the number of NEW rows inserted.
    """
    rows = [(z.location_id, z.borough, z.zone, z.service_zone) for z in zones]
    with conn.cursor() as cur:
        cur.executemany(
            """
            INSERT INTO raw.taxi_zones (location_id, borough, zone, service_zone)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (location_id) DO NOTHING
            """,
            rows,
        )
        return cur.rowcount
