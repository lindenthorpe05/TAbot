"""
odds_capture.py - Scheduled TAB odds/form snapshot for the training pipeline.

Runs the existing interceptor (headless, already proven reliable) to capture
upcoming races, parses them, and accumulates rows into harvest_matrix.csv -
unlike current_race_matrix.csv (used by the live agent), this file is never
overwritten wholesale. Each runner's row is kept up to date (latest capture
wins) until betfair_harvest.py matches it to a settled result and removes it.

Meant to run every 15-20 minutes via a scheduled job (see
.github/workflows/odds_capture.yml). No TAB credentials or browser session
state is needed - this only reads public race data, same as interceptor.py.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd

from interceptor import run_multi
from parser import build_matrix

BASE_DIR = Path(__file__).parent
ACCUM_CSV = BASE_DIR / "harvest_matrix.csv"
LOG_FILE = BASE_DIR / "odds_capture.log"

# How far ahead to look for races each run. Wider than the run interval so a
# missed or delayed run doesn't lose coverage of a race.
LOOKAHEAD_MINUTES = 90

# Drop accumulated rows this old if they were never matched to a result
# (e.g. an abandoned race) so the file doesn't grow forever.
MAX_ROW_AGE_DAYS = 3

DEDUP_KEYS = ["meeting_date", "track_name", "race_number", "runner_number"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("odds_capture")


def _prune_old_rows(df: pd.DataFrame) -> pd.DataFrame:
    if "meeting_date" not in df.columns or df.empty:
        return df
    cutoff = (datetime.now(timezone.utc) - timedelta(days=MAX_ROW_AGE_DAYS)).date().isoformat()
    before = len(df)
    df = df[df["meeting_date"].astype(str) >= cutoff]
    dropped = before - len(df)
    if dropped:
        log.info(f"Pruned {dropped} stale row(s) older than {MAX_ROW_AGE_DAYS} days")
    return df


def accumulate(new_rows: pd.DataFrame) -> int:
    """Merge new_rows into ACCUM_CSV, keeping the latest snapshot per runner.
    Returns the number of rows in the file after accumulation."""
    if ACCUM_CSV.exists():
        existing = pd.read_csv(ACCUM_CSV)
        combined = pd.concat([existing, new_rows], ignore_index=True)
    else:
        combined = new_rows

    combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")
    combined = _prune_old_rows(combined)
    combined.to_csv(ACCUM_CSV, index=False)
    return len(combined)


async def main() -> None:
    log.info("=== ODDS CAPTURE START ===")
    await run_multi(minutes=LOOKAHEAD_MINUTES)

    df = build_matrix()
    if df.empty:
        log.info("No races captured this run.")
        return

    total = accumulate(df)
    log.info(f"Captured {len(df)} runner rows this run. harvest_matrix.csv now has {total} rows.")
    log.info("=== ODDS CAPTURE DONE ===")


if __name__ == "__main__":
    asyncio.run(main())
