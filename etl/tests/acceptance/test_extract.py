from pathlib import Path

import polars as pl
import pytest

import extract
from extract import Zone, extract_trips, extract_zones


class FakeResponse:
    def __init__(self, payload: bytes, content_length: int | None = None):
        self._payload = payload
        self._pos = 0
        length = len(payload) if content_length is None else content_length
        self.headers = {"Content-Length": str(length)}

    def read(self, size: int = -1) -> bytes:
        chunk = self._payload[self._pos : self._pos + size]
        self._pos += len(chunk)
        return chunk

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def write_trip_parquet(path: Path) -> None:
    pl.DataFrame(
        {name: [1] for name in extract.TRIP_SOURCE_COLUMNS}
    ).write_parquet(path)


def test_missing_explicit_source_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_trips(tmp_path / "nope.parquet", month="2024-01", download=False)


def test_existing_file_is_read_without_downloading(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise AssertionError("must not download when the file exists")

    monkeypatch.setattr(extract.urllib.request, "urlopen", boom)
    source = tmp_path / "yellow_tripdata_2024-01.parquet"
    write_trip_parquet(source)

    frame = extract_trips(source, month="2024-01", download=True)

    assert isinstance(frame, pl.DataFrame)
    assert frame.columns == extract.TRIP_SOURCE_COLUMNS


def test_missing_default_file_is_downloaded_then_published(tmp_path, monkeypatch):
    seed = tmp_path / "seed.parquet"
    write_trip_parquet(seed)
    payload = seed.read_bytes()
    seen = {}

    def fake_urlopen(url, timeout):
        seen["url"], seen["timeout"] = url, timeout
        return FakeResponse(payload)

    monkeypatch.setattr(extract.urllib.request, "urlopen", fake_urlopen)
    target = tmp_path / "cache" / "yellow_tripdata_2024-01.parquet"

    frame = extract_trips(target, month="2024-01", download=True)

    assert seen["url"].endswith("/yellow_tripdata_2024-01.parquet")
    assert seen["timeout"] is not None
    assert frame.height == 1
    assert target.read_bytes() == payload
    assert list(target.parent.glob("*.part")) == []


def test_incomplete_download_never_appears_in_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(
        extract.urllib.request,
        "urlopen",
        lambda url, timeout: FakeResponse(b"abc", content_length=100),
    )
    target = tmp_path / "cache" / "yellow_tripdata_2024-01.parquet"

    with pytest.raises(OSError):
        extract_trips(target, month="2024-01", download=True)

    assert not target.exists()
    assert list(target.parent.glob("*.part")) == []


def test_zones_are_typed_and_keep_placeholder_values(tmp_path):
    csv_file = tmp_path / "taxi_zone_lookup.csv"
    csv_file.write_text(
        '\ufeff"LocationID","Borough","Zone","service_zone"\n'
        '1,"EWR","Newark Airport","EWR"\n'
        '4,"Manhattan","Alphabet City","Yellow Zone"\n'
        '264,"Unknown","N/A","N/A"\n'
        '300,"Queens","Odd, Name",\n',
        encoding="utf-8",
    )

    zones = extract_zones(csv_file, download=False)

    assert zones[0] == Zone(1, "EWR", "Newark Airport", "EWR")
    assert zones[2] == Zone(264, "Unknown", "N/A", "N/A")
    assert zones[3] == Zone(300, "Queens", "Odd, Name", None)
    assert all(isinstance(z.location_id, int) for z in zones)


def test_missing_explicit_zone_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_zones(tmp_path / "nope.csv", download=False)
