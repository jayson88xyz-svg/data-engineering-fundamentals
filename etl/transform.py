import polars as pl

# Output columns, in the same order as raw.yellow_taxi_trips (minus trip_id,
# which PostgreSQL generates).
TRIP_COLUMNS = [
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "pickup_location_id",
    "dropoff_location_id",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
]

MONEY = pl.Decimal(14, 2)  # matches numeric(14, 2)


def _local_timestamp(column: str, dtype: pl.DataType) -> pl.Expr:
    """Parse to a naive timestamp. Unusable values become null (and get filtered).

    The TLC times are local wall-clock times, so we never convert time zones:
    a timezone-aware value just has its zone label dropped, keeping the clock time.
    """
    col = pl.col(column)
    if dtype == pl.String:
        return col.str.to_datetime(strict=False)
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is not None:
        return col.dt.replace_time_zone(None)
    return col.cast(pl.Datetime("us"), strict=False)


def clean_trips(source: pl.DataFrame) -> pl.DataFrame:
    """Apply the Lecture 02 cleaning policy and return SQL-aligned columns.

    Rows are REJECTED when:
      * the pickup or drop-off timestamp is missing or unparseable
      * the drop-off is earlier than the pickup (reversed)
      * trip_distance is negative

    Rows are KEPT even when they have:
      * zero distance, or a null passenger_count / distance / tip
      * a pickup or drop-off zone ID that is not in the zone lookup (e.g. 999)
      * a negative fare (a refund or adjustment, not a bad record)
    """
    schema = source.schema

    renamed = source.select(
        pl.col("VendorID").cast(pl.Int32, strict=False).alias("vendor_id"),
        _local_timestamp("tpep_pickup_datetime", schema["tpep_pickup_datetime"]).alias(
            "pickup_datetime"
        ),
        _local_timestamp("tpep_dropoff_datetime", schema["tpep_dropoff_datetime"]).alias(
            "dropoff_datetime"
        ),
        # TLC stores passenger_count as a float; the table wants an integer.
        pl.col("passenger_count").cast(pl.Int32, strict=False),
        pl.col("trip_distance").cast(pl.Float64).fill_nan(None),
        pl.col("PULocationID").cast(pl.Int32, strict=False).alias("pickup_location_id"),
        pl.col("DOLocationID").cast(pl.Int32, strict=False).alias("dropoff_location_id"),
        pl.col("payment_type").cast(pl.Int32, strict=False),
        # Round to cents first, then store as an exact decimal.
        *[
            pl.col(name).cast(pl.Float64).fill_nan(None).round(2).cast(MONEY, strict=False)
            for name in ("fare_amount", "tip_amount", "total_amount")
        ],
    )

    keep = (
        pl.col("pickup_datetime").is_not_null()
        & pl.col("dropoff_datetime").is_not_null()
        & (pl.col("dropoff_datetime") >= pl.col("pickup_datetime"))
        # null distance passes: `is_null() | >= 0` keeps unknown, drops negatives
        & (pl.col("trip_distance").is_null() | (pl.col("trip_distance") >= 0))
    )
    return renamed.filter(keep).select(TRIP_COLUMNS)
