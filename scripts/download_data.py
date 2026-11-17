#!/usr/bin/env python3
"""Download and verify the public UCI Diabetes 130-US Hospitals dataset."""

from __future__ import annotations

import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path


URL = (
    "https://archive.ics.uci.edu/static/public/296/"
    "diabetes+130-us+hospitals+for+years+1999-2008.zip"
)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
ARCHIVE = RAW_DIR / "diabetes_130_us_hospitals.zip"
DATA_FILE = RAW_DIR / "diabetic_data.csv"
EXPECTED_ROWS = 101_766
EXPECTED_ARCHIVE_SHA256 = (
    "f82ac129da2ddd2299391ff6fbae3a6a58b3edcf59ac9d7bd480c00fe453112a"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle) - 1


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if not ARCHIVE.exists():
        print(f"Downloading {URL}")
        with urllib.request.urlopen(URL) as response, ARCHIVE.open("wb") as output:
            shutil.copyfileobj(response, output)

    archive_hash = sha256(ARCHIVE)
    if archive_hash != EXPECTED_ARCHIVE_SHA256:
        raise RuntimeError(
            "Unexpected UCI archive hash. "
            f"Expected {EXPECTED_ARCHIVE_SHA256}, got {archive_hash}."
        )

    if not DATA_FILE.exists():
        with zipfile.ZipFile(ARCHIVE) as zipped:
            for member in ("diabetic_data.csv", "IDS_mapping.csv"):
                zipped.extract(member, RAW_DIR)

    rows = count_csv_rows(DATA_FILE)
    if rows != EXPECTED_ROWS:
        raise RuntimeError(f"Expected {EXPECTED_ROWS:,} rows, found {rows:,}.")
    print(f"Ready: {DATA_FILE} ({rows:,} rows, SHA-256 verified)")


if __name__ == "__main__":
    main()

