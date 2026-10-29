"""Download the Home Credit Default Risk data from Kaggle into data/raw/.

Prerequisites (once):
  1. pip install kaggle
  2. Kaggle > Settings > API > "Create New Token": save kaggle.json to ~/.kaggle/kaggle.json
     (Windows: C:\\Users\\<you>\\.kaggle\\kaggle.json), or set KAGGLE_USERNAME and KAGGLE_KEY.
  3. Accept the competition rules once in the browser:
     https://www.kaggle.com/competitions/home-credit-default-risk/rules

Usage:
  python scripts/download_data.py            # download and unzip
  python scripts/download_data.py --force    # re-download even if files exist
"""

from __future__ import annotations

import argparse
import logging
import sys
import zipfile
from pathlib import Path

COMPETITION = "home-credit-default-risk"
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

EXPECTED_FILES = [
    "application_train.csv",
    "application_test.csv",
    "bureau.csv",
    "bureau_balance.csv",
    "previous_application.csv",
    "POS_CASH_balance.csv",
    "credit_card_balance.csv",
    "installments_payments.csv",
    "HomeCredit_columns_description.csv",
]

log = logging.getLogger("download_data")


def already_downloaded(raw_dir: Path) -> bool:
    return all((raw_dir / name).exists() for name in EXPECTED_FILES)


def download(raw_dir: Path) -> Path:
    # Imported here so that --help works without the kaggle package or credentials.
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()  # raises if kaggle.json / env variables are missing
    log.info("Downloading %s to %s (about 690 MB)...", COMPETITION, raw_dir)
    api.competition_download_files(COMPETITION, path=str(raw_dir), quiet=False)
    return raw_dir / f"{COMPETITION}.zip"


def unzip(archive: Path, raw_dir: Path) -> None:
    log.info("Unzipping %s...", archive.name)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(raw_dir)
    archive.unlink()


def check(raw_dir: Path) -> None:
    missing = [name for name in EXPECTED_FILES if not (raw_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing after download: {missing}")
    # Quick sanity check on the training table: 307,511 rows expected.
    with open(raw_dir / "application_train.csv", encoding="utf-8") as f:
        n_rows = sum(1 for _ in f) - 1
    log.info("application_train.csv: %s rows", f"{n_rows:,}")
    if n_rows != 307_511:
        log.warning("Unexpected row count: the source data may have changed.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--force", action="store_true", help="re-download even if the files exist")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if already_downloaded(RAW_DIR) and not args.force:
        log.info("Data already in %s: nothing to do (use --force to re-download).", RAW_DIR)
        return 0

    try:
        archive = download(RAW_DIR)
    except Exception as exc:  # auth error, rules not accepted, network
        log.error("Download failed: %s", exc)
        log.error(
            "Check ~/.kaggle/kaggle.json and accept the rules at "
            "https://www.kaggle.com/competitions/%s/rules",
            COMPETITION,
        )
        return 1

    unzip(archive, RAW_DIR)
    check(RAW_DIR)
    log.info("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
