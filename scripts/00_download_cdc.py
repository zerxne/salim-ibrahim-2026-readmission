"""Download the CDC Diabetes Health Indicators (BRFSS 2015, 253,680 rows) from UCI and verify it."""
import hashlib
import sys
import urllib.request
from pathlib import Path

URL = "https://archive.ics.uci.edu/static/public/891/data.csv"  # UCI dataset id 891 (CC BY 4.0)
SHA256 = "9f71fda9d4ae5f4878c99b9233b6a16accfa9a17c194116a6b78100540934964"
TARGET = Path(__file__).resolve().parents[1] / "data" / "external" / "cdc" / "cdc_diabetes_health_indicators.csv"


def main():
    TARGET.parent.mkdir(parents=True, exist_ok=True)
    if not TARGET.exists():
        with urllib.request.urlopen(URL, timeout=120) as response:
            TARGET.write_bytes(response.read())
    digest = hashlib.sha256(TARGET.read_bytes()).hexdigest()
    if digest != SHA256:
        sys.exit(f"SHA-256 mismatch for {TARGET}: {digest}")
    print(f"OK {TARGET}")


if __name__ == "__main__":
    main()
