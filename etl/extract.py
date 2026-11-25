import csv
import os
import tempfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import polars as pl

TRIP_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{month}.parquet"
ZONES_URL = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zone_lookup.csv"
DOWNLOAD_TIMEOUT = 60  # seconds, applies to each network operation

# Only the TLC fields the raw table needs. Reading fewer columns saves memory.
TRIP_SOURCE_COLUMNS = [
    "VendorID",
    "tpep_pickup_datetime",
    "tpep_dropoff_datetime",
    "passenger_count",
    "trip_distance",
    "PULocationID",
    "DOLocationID",
    "payment_type",
    "fare_amount",
    "tip_amount",
    "total_amount",
]


@dataclass(frozen=True)
class Zone:
    """One row of the taxi zone lookup, already typed."""

    location_id: int
    borough: str
    zone: str
    service_zone: str | None


def _download(url: str, destination: Path) -> None:
    """Download to a temporary file, then move it into place when complete.

    Another process (or a later run) never sees a half-written file at
    `destination`, because os.replace only happens after the last byte arrives.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=destination.parent, suffix=".part")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(
            url, timeout=DOWNLOAD_TIMEOUT
        ) as response:
            expected = response.headers.get("Content-Length")
            written = 0
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
                written += len(chunk)
        if expected is not None and written != int(expected):
            raise OSError(f"Incomplete download: got {written} of {expected} bytes")
        os.replace(tmp, destination)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise


def _ensure_local(source: Path, url: str, *, download: bool) -> None:
    if source.exists():
        return
    if not download:
        # The person named this file explicitly, so a missing file is an error.
        raise FileNotFoundError(f"Source file not found: {source}")
    _download(url, source)


def extract_trips(source: Path, *, month: str, download: bool) -> pl.DataFrame:
    """Read a local Parquet file, downloading the requested month if allowed."""
    _ensure_local(source, TRIP_URL.format(month=month), download=download)
    return pl.read_parquet(source, columns=TRIP_SOURCE_COLUMNS)


def extract_zones(source: Path, *, download: bool) -> list[Zone]:
    """Read the taxi zone lookup CSV into typed rows."""
    _ensure_local(source, ZONES_URL, download=download)
    zones: list[Zone] = []
    with source.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            zones.append(
                Zone(
                    location_id=int(row["LocationID"]),
                    borough=row["Borough"],
                    zone=row["Zone"],
                    service_zone=row["service_zone"] or None,
                )
            )
    return zones
