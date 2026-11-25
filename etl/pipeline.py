import argparse
import os
import sys
import time
from datetime import date
from pathlib import Path

import psycopg

from extract import extract_trips, extract_zones
from load import load_trips, load_zones
from transform import clean_trips

DEFAULT_MONTH = "2024-01"
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def parse_month(value: str) -> str:
    try:
        date.fromisoformat(f"{value}-01")
    except ValueError:
        raise argparse.ArgumentTypeError("month must be in YYYY-MM format") from None
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append a month of Yellow Taxi trips.")
    parser.add_argument("--month", type=parse_month, default=DEFAULT_MONTH)
    parser.add_argument(
        "--data-dir", type=Path, default=os.environ.get("TAXI_DATA_DIR") or DEFAULT_DATA_DIR,
        help="absolute data directory (overrides TAXI_DATA_DIR)",
    )
    parser.add_argument("--source", type=Path, help="use a local Parquet file without downloading")
    parser.add_argument(
        "--zones-only", action="store_true", help="load the taxi zone lookup and skip trips"
    )
    parser.add_argument(
        "--zones-source", type=Path, help="use a local zone CSV without downloading"
    )

    args = parser.parse_args(argv)
    if not args.data_dir.is_absolute():
        parser.error("the data directory must be an absolute path")
    args.zones_download = args.zones_source is None
    if args.zones_download:
        args.zones_source = args.data_dir / "raw" / "reference" / "taxi_zone_lookup.csv"
    else:
        args.zones_source = args.zones_source.resolve()
    args.download = args.source is None
    if args.download:
        args.source = args.data_dir / "raw" / "yellow_taxi" / f"yellow_tripdata_{args.month}.parquet"
    else:
        args.source = args.source.resolve()
    return args


def run_zones(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    zones = extract_zones(args.zones_source, download=args.zones_download)
    with psycopg.connect(connect_timeout=10) as conn:
        inserted = load_zones(conn, zones)
    elapsed = time.perf_counter() - started
    print(f"source: {args.zones_source}")
    print(f"zones read: {len(zones):,}  inserted: {inserted:,}  elapsed: {elapsed:.1f}s")


def run_trips(args: argparse.Namespace) -> None:
    started = time.perf_counter()
    source = extract_trips(args.source, month=args.month, download=args.download)
    trips = clean_trips(source)
    # `with connect()` commits on success and rolls back if anything raises,
    # so the whole trip load is one transaction.
    with psycopg.connect(connect_timeout=10) as conn:
        inserted = load_trips(conn, trips)
    elapsed = time.perf_counter() - started
    print(f"source: {args.source}")
    print(f"source rows: {source.height:,}")
    print(f"accepted rows: {trips.height:,}  (rejected: {source.height - trips.height:,})")
    print(f"inserted rows: {inserted:,}  elapsed: {elapsed:.1f}s")


def run(args: argparse.Namespace) -> None:
    if args.zones_only:
        run_zones(args)
    else:
        run_trips(args)


def main() -> int:
    try:
        run(parse_args())
    except (FileNotFoundError, psycopg.Error, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
