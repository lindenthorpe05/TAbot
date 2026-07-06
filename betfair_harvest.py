"""
betfair_harvest.py - Label harvest_matrix.csv rows with results from Betfair
and append them to historical_results.csv for model training.

This deliberately does NOT scrape TAB for results (that was the fragile,
Cloudflare-fighting part of the old harvester.py). It only needs to know
who won each race, which the free Betfair Delayed API provides directly -
no browser, no stealth, just authenticated HTTPS calls. The actual runner
features (odds, form ratings, jockey/trainer) still come from TAB, captured
ahead of time into harvest_matrix.csv by odds_capture.py.

Meant to run every 15-20 minutes via a scheduled job (see
.github/workflows/betfair_results.yml). Requires Betfair credentials - see
betfair_client.py for the required environment variables.
"""

import logging
import re
from datetime import datetime, timezone, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from betfair_client import BetfairClient, parse_runner_number

BASE_DIR = Path(__file__).parent
ACCUM_CSV = BASE_DIR / "harvest_matrix.csv"
HISTORICAL_CSV = BASE_DIR / "historical_results.csv"
LOG_FILE = BASE_DIR / "betfair_harvest.log"

AU_TZ = ZoneInfo("Australia/Sydney")

# How far back to look for markets that may have settled by now. Races
# that started less than MIN_SETTLE_BUFFER ago are skipped - Betfair markets
# often take a few minutes to move to CLOSED after the race finishes.
LOOKBACK_HOURS = 6
MIN_SETTLE_BUFFER_MINUTES = 10

HIST_COLUMNS = [
    "meeting_date", "track_name", "race_number", "race_distance",
    "track_condition", "race_class", "runner_name", "runner_number",
    "barrier_draw", "weight_kg", "jockey", "trainer", "jt_pair",
    "fixed_odds_win", "fixed_odds_win_open", "odds_drift",
    "win_rate_last5", "place_rate_last5", "jt_pair_win_rate",
    "form_rating_dfs", "early_speed_rating", "early_speed_band",
    "win_flag",
]

DEDUP_KEYS = ["meeting_date", "track_name", "race_number", "runner_number"]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, encoding="utf-8"), logging.StreamHandler()],
)
log = logging.getLogger("betfair_harvest")


def _normalize_track(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9 ]", " ", str(name)).upper()
    return re.sub(r"\s+", " ", name).strip()


def _parse_race_number(market_name: str) -> int | None:
    m = re.search(r"R(\d+)", market_name)
    if m:
        return int(m.group(1))
    m = re.search(r"^(\d+)", market_name.strip())
    return int(m.group(1)) if m else None


def _local_date(market_start_time: str) -> str:
    dt = datetime.fromisoformat(market_start_time.replace("Z", "+00:00"))
    return dt.astimezone(AU_TZ).date().isoformat()


def match_and_label(harvest_df: pd.DataFrame, markets: list[dict], results: dict[str, dict]) -> pd.DataFrame:
    """Return labeled rows (HIST_COLUMNS schema) for markets that closed and
    matched a row in harvest_df, plus the set of (track, race_number, date)
    keys that were resolved (whether matched or not) so callers can decide
    what to drop from the pending accumulator."""
    harvest_df = harvest_df.copy()
    harvest_df["_track_norm"] = harvest_df["track_name"].map(_normalize_track)

    labeled_rows = []
    resolved_keys = set()

    for market in markets:
        market_id = market["market_id"]
        settle = results.get(market_id)
        if not settle:
            continue
        if settle["status"] != "CLOSED":
            continue

        race_number = _parse_race_number(market["market_name"])
        track_norm = _normalize_track(market["venue"])
        race_date = _local_date(market["market_start_time"])

        if race_number is None:
            log.warning(f"Could not parse race number from market_name={market['market_name']!r}, skipping")
            continue

        candidates = harvest_df[
            (harvest_df["race_number"] == race_number)
            & (harvest_df["meeting_date"].astype(str) == race_date)
            & (
                harvest_df["_track_norm"].str.contains(track_norm, na=False)
                | harvest_df["_track_norm"].apply(lambda t: track_norm in t or t in track_norm)
            )
        ]

        key = (track_norm, race_number, race_date)
        if candidates.empty:
            log.info(f"No harvest_matrix match for {market['venue']} R{race_number} ({race_date}) - skipping")
            continue

        winner_selection_ids = {
            sid for sid, status in settle["runners"].items() if status == "WINNER"
        }

        matched_any = False
        for runner in market["runners"]:
            runner_num, _horse_name = parse_runner_number(runner["runner_name"])
            if runner_num is None:
                continue
            row_match = candidates[candidates["runner_number"] == runner_num]
            if row_match.empty:
                continue
            row = row_match.iloc[0].to_dict()
            row["win_flag"] = 1 if runner["selection_id"] in winner_selection_ids else 0
            labeled_rows.append(row)
            matched_any = True

        if matched_any:
            resolved_keys.add(key)
            log.info(f"Labeled {market['venue']} R{race_number} ({race_date})")

    if not labeled_rows:
        return pd.DataFrame(columns=HIST_COLUMNS), resolved_keys

    out = pd.DataFrame(labeled_rows)
    for col in HIST_COLUMNS:
        if col not in out.columns:
            out[col] = None
    return out[HIST_COLUMNS], resolved_keys


def append_to_historical(new_rows: pd.DataFrame) -> int:
    if new_rows.empty:
        return 0
    if HISTORICAL_CSV.exists():
        existing = pd.read_csv(HISTORICAL_CSV)
        combined = pd.concat([existing, new_rows], ignore_index=True)
        combined = combined.drop_duplicates(subset=DEDUP_KEYS, keep="last")
        added = len(combined) - len(existing)
    else:
        combined = new_rows.drop_duplicates(subset=DEDUP_KEYS, keep="last")
        added = len(combined)
    combined.to_csv(HISTORICAL_CSV, index=False)
    return added


def drop_resolved(harvest_df: pd.DataFrame, resolved_keys: set) -> pd.DataFrame:
    if not resolved_keys or harvest_df.empty:
        return harvest_df
    track_norm = harvest_df["track_name"].map(_normalize_track)
    keep_mask = pd.Series(True, index=harvest_df.index)
    for tn, rn, rd in resolved_keys:
        match = (
            (track_norm == tn)
            & (harvest_df["race_number"] == rn)
            & (harvest_df["meeting_date"].astype(str) == rd)
        )
        keep_mask &= ~match
    return harvest_df[keep_mask]


def main() -> None:
    log.info("=== BETFAIR HARVEST START ===")

    if not ACCUM_CSV.exists():
        log.info(f"{ACCUM_CSV.name} not found - nothing pending. Run odds_capture.py first.")
        return

    harvest_df = pd.read_csv(ACCUM_CSV)
    if harvest_df.empty:
        log.info("harvest_matrix.csv is empty - nothing pending.")
        return

    now = datetime.now(timezone.utc)
    started_after = now - timedelta(hours=LOOKBACK_HOURS)
    started_before = now - timedelta(minutes=MIN_SETTLE_BUFFER_MINUTES)

    client = BetfairClient()
    log.info(f"Querying Betfair for AU thoroughbred WIN markets started {started_after} - {started_before}")
    markets = client.list_recently_started_au_thoroughbred_markets(started_after, started_before)
    log.info(f"Found {len(markets)} candidate market(s)")

    if not markets:
        log.info("No candidate markets in window. Nothing to do.")
        return

    results = client.get_settled_results([m["market_id"] for m in markets])
    closed_count = sum(1 for r in results.values() if r["status"] == "CLOSED")
    log.info(f"{closed_count}/{len(results)} market(s) are CLOSED (settled)")

    labeled, resolved_keys = match_and_label(harvest_df, markets, results)
    added = append_to_historical(labeled)
    log.info(f"Appended {added} new labeled row(s) to {HISTORICAL_CSV.name}")

    if resolved_keys:
        remaining = drop_resolved(harvest_df, resolved_keys)
        remaining.to_csv(ACCUM_CSV, index=False)
        log.info(f"Removed {len(resolved_keys)} resolved race(s) from {ACCUM_CSV.name}; {len(remaining)} row(s) remain pending")

    log.info("=== BETFAIR HARVEST DONE ===")


if __name__ == "__main__":
    main()
