from datetime import datetime, timezone
from decimal import Decimal

import polars as pl

from transform import TRIP_COLUMNS, clean_trips

START = datetime(2024, 1, 2, 12)
END = datetime(2024, 1, 2, 12, 15)


def frame(**overrides) -> pl.DataFrame:
    base = {
        "VendorID": [1],
        "tpep_pickup_datetime": [START],
        "tpep_dropoff_datetime": [END],
        "passenger_count": [1.0],
        "trip_distance": [1.5],
        "PULocationID": [10],
        "DOLocationID": [20],
        "payment_type": [1],
        "fare_amount": [10.0],
        "tip_amount": [1.0],
        "total_amount": [11.0],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_output_columns_match_the_table_order():
    assert clean_trips(frame()).columns == TRIP_COLUMNS


def test_zero_length_trip_is_kept_but_reversed_trip_is_not():
    same = clean_trips(frame(tpep_dropoff_datetime=[START]))
    reversed_ = clean_trips(frame(tpep_dropoff_datetime=[datetime(2024, 1, 2, 11)]))
    assert same.height == 1
    assert reversed_.height == 0


def test_negative_fare_is_a_legitimate_adjustment():
    result = clean_trips(frame(fare_amount=[-52.5], total_amount=[-55.0]))
    assert result["fare_amount"].to_list() == [Decimal("-52.50")]


def test_money_is_rounded_to_cents():
    result = clean_trips(frame(fare_amount=[2.3700000001], tip_amount=[0.199999999]))
    assert result["fare_amount"].to_list() == [Decimal("2.37")]
    assert result["tip_amount"].to_list() == [Decimal("0.20")]


def test_unparseable_text_timestamp_is_rejected():
    result = clean_trips(
        frame(
            VendorID=[1, 1],
            tpep_pickup_datetime=["2024-01-02 12:00:00", "not a date"],
            tpep_dropoff_datetime=["2024-01-02 12:10:00", "2024-01-02 12:10:00"],
            passenger_count=[1.0, 1.0],
            trip_distance=[1.0, 1.0],
            PULocationID=[1, 1],
            DOLocationID=[1, 1],
            payment_type=[1, 1],
            fare_amount=[5.0, 5.0],
            tip_amount=[0.0, 0.0],
            total_amount=[5.0, 5.0],
        )
    )
    assert result.height == 1
    assert result["pickup_datetime"].to_list() == [datetime(2024, 1, 2, 12, 0)]


def test_timezone_label_is_dropped_without_shifting_the_clock():
    aware = pl.Series([datetime(2024, 1, 2, 12, tzinfo=timezone.utc)]).dt.convert_time_zone(
        "America/New_York"
    )
    # 12:00 UTC is 07:00 in New York. The clock time we KEEP is whatever the
    # column shows, because we only strip the zone label.
    result = clean_trips(frame(tpep_pickup_datetime=aware, tpep_dropoff_datetime=aware))
    assert result["pickup_datetime"].to_list() == [datetime(2024, 1, 2, 7)]


def test_integer_columns_are_integers_and_distance_stays_float():
    result = clean_trips(frame())
    assert result.schema["passenger_count"] == pl.Int32
    assert result.schema["pickup_location_id"] == pl.Int32
    assert result.schema["trip_distance"] == pl.Float64
    assert result.schema["fare_amount"] == pl.Decimal(14, 2)
