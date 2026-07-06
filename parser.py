"""
parser.py - Parse raw_race_data.json into a flat CSV race matrix.

Extracts per-runner variables from TAB API race detail payloads and
computes derived metrics using vectorized pandas operations.

Output: current_race_matrix.csv
"""

import json
import re
from pathlib import Path

import pandas as pd

RAW_FILE = Path(__file__).parent / "raw_race_data.json"
OUTPUT_CSV = Path(__file__).parent / "current_race_matrix.csv"


def load_race_entries(path: Path) -> list[dict]:
    """Load raw JSON and return only race-detail entries that contain runners."""
    data = json.loads(path.read_text(encoding="utf-8"))

    # Handle both dict-wrapped and bare-list formats
    entries = data.get("racing_data", data) if isinstance(data, dict) else data

    return [
        e for e in entries
        if isinstance(e.get("data"), dict) and "runners" in e["data"]
    ]


def _extract_result(race: dict) -> tuple[dict, float, float]:
    """Return (runner_number -> finishing_position, win_dividend, place_dividend)."""
    results = race.get("results") or []
    finishing = {}
    for pos, group in enumerate(results, start=1):
        for runner_num in (group if isinstance(group, list) else [group]):
            finishing[int(runner_num)] = pos

    win_div = place_div = 0.0
    for div in race.get("dividends") or []:
        product = div.get("wageringProduct", "")
        pools = div.get("poolDividends") or []
        if product == "Win" and pools:
            win_div = float(pools[0].get("amount", 0))
        elif product == "Place" and pools:
            place_div = float(pools[0].get("amount", 0))

    return finishing, win_div or None, place_div or None


def flatten_runners(race_entries: list[dict]) -> pd.DataFrame:
    """Flatten all race entries into a single DataFrame of runners."""
    rows = []
    for entry in race_entries:
        race = entry["data"]
        meeting = race.get("meeting", {})

        finishing, win_div, place_div = _extract_result(race)

        race_meta = {
            "race_name": race.get("raceName"),
            "race_number": race.get("raceNumber"),
            "race_distance": race.get("raceDistance"),
            "race_start_time": race.get("raceStartTime"),
            "race_status": race.get("raceStatus"),
            "race_class": race.get("raceClassConditions"),
            "prize_money": race.get("prizeMoney"),
            "track_name": meeting.get("meetingName"),
            "track_condition": meeting.get("trackCondition"),
            "weather": meeting.get("weatherCondition"),
            "meeting_date": meeting.get("meetingDate"),
            "win_dividend": win_div,
            "place_dividend": place_div,
        }

        for runner in race.get("runners", []):
            fo = runner.get("fixedOdds") or {}
            runner_num = runner.get("runnerNumber")
            rows.append({
                **race_meta,
                "runner_name": runner.get("runnerName"),
                "runner_number": runner_num,
                "finishing_position": finishing.get(runner_num),
                "barrier_draw": runner.get("barrierNumber"),
                "weight_kg": runner.get("handicapWeight"),
                "jockey": runner.get("riderDriverFullName") or runner.get("riderDriverName"),
                "trainer": runner.get("trainerFullName") or runner.get("trainerName"),
                "last_5_starts": runner.get("last5Starts"),
                "form_rating_dfs": runner.get("dfsFormRating"),
                "form_rating_tech": runner.get("techFormRating"),
                "total_rating_pts": runner.get("totalRatingPoints"),
                "early_speed_rating": runner.get("earlySpeedRating"),
                "early_speed_band": runner.get("earlySpeedRatingBand"),
                "blinkers": runner.get("blinkers"),
                "emergency": runner.get("emergency"),
                "fixed_odds_win": fo.get("returnWin"),
                "fixed_odds_place": fo.get("returnPlace"),
                "fixed_odds_win_open": fo.get("returnWinOpen"),
                "fixed_odds_win_open_daily": fo.get("returnWinOpenDaily"),
                "odds_pct_change": fo.get("percentageChange"),
                "is_favourite_win": fo.get("isFavouriteWin"),
                "is_favourite_place": fo.get("isFavouritePlace"),
                "betting_status": fo.get("bettingStatus"),
            })

    return pd.DataFrame(rows)


def parse_last5(series: pd.Series) -> pd.DataFrame:
    """Parse 'last5Starts' strings like '14x32' into win/place counts.

    Each character is a finishing position: 1-9 or 'x' (unplaced/10+).
    Returns columns: wins_last5, places_last5 (top 3), starts_last5.
    """
    def _count(val: str, check):
        if not isinstance(val, str):
            return 0
        return sum(1 for ch in val if ch.isdigit() and check(int(ch)))

    starts = series.fillna("").str.len()
    wins = series.apply(lambda v: _count(v, lambda n: n == 1))
    places = series.apply(lambda v: _count(v, lambda n: 1 <= n <= 3))

    return pd.DataFrame({
        "starts_last5": starts,
        "wins_last5": wins,
        "places_last5": places,
    })


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived columns using vectorized operations."""

    # -- Recent win rate from last5Starts ------------------------------------
    form = parse_last5(df["last_5_starts"])
    df = pd.concat([df, form], axis=1)
    df["win_rate_last5"] = (df["wins_last5"] / df["starts_last5"].replace(0, pd.NA)).round(4)
    df["place_rate_last5"] = (df["places_last5"] / df["starts_last5"].replace(0, pd.NA)).round(4)

    # -- Jockey/Trainer recent win rate (proxy from this dataset) ------------
    # With a single race snapshot we can't compute true historical J/T pair
    # win rates. Instead we derive a combined form score from the runner's
    # recent form as a proxy. With multi-race data from the interceptor,
    # this should be replaced with a proper groupby on historical results.
    df["jt_pair"] = df["jockey"].str.strip() + " / " + df["trainer"].str.strip()
    # Group win rate by J/T pair across all runners in the dataset
    jt_win_rate = (
        df.groupby("jt_pair", observed=True)["win_rate_last5"]
        .transform("mean")
        .round(4)
    )
    df["jt_pair_win_rate"] = jt_win_rate

    # -- Odds movement (weight fluctuation proxy) ----------------------------
    # True weight fluctuation requires the previous race's weight which TAB
    # doesn't include in a single snapshot. We compute odds drift instead:
    # the difference between opening and current fixed odds, which is a
    # direct market signal of perceived chance.
    win = pd.to_numeric(df["fixed_odds_win"], errors="coerce")
    win_open = pd.to_numeric(df["fixed_odds_win_open"], errors="coerce")
    df["odds_drift"] = (win - win_open).round(2)
    raw_pct = ((win - win_open) / win_open.where(win_open != 0)) * 100
    df["odds_drift_pct"] = raw_pct.round(2)

    return df


def build_matrix() -> pd.DataFrame:
    """Main pipeline: load -> flatten -> derive -> clean -> output."""
    print(f"[*] Loading {RAW_FILE}")
    race_entries = load_race_entries(RAW_FILE)
    print(f"    Found {len(race_entries)} race detail entries with runners")

    if not race_entries:
        print("[!] No race data with runners found. Run interceptor.py first.")
        return pd.DataFrame()

    # Deduplicate race entries (interceptor captures polling refreshes)
    seen = set()
    unique_entries = []
    for e in race_entries:
        key = e["url"].split("?")[0]
        if key not in seen:
            seen.add(key)
            unique_entries.append(e)
    print(f"    After dedup: {len(unique_entries)} unique races")

    df = flatten_runners(unique_entries)
    print(f"    Flattened {len(df)} runners")

    # Filter out emergencies and runners that aren't actually running
    EXCLUDE_STATUSES = {"LateScratched", "Scratched"}
    pre = len(df)
    df = df[
        df["emergency"].ne(True) &
        ~df["betting_status"].isin(EXCLUDE_STATUSES)
    ]
    print(f"    After removing emergencies/scratched: {len(df)} (dropped {pre - len(df)})")

    df = compute_derived_metrics(df)

    # Select final columns for the CSV matrix
    output_cols = [
        "meeting_date", "track_name", "race_number", "race_name",
        "race_distance", "track_condition", "race_class",
        "race_status", "win_dividend", "place_dividend",
        "runner_name", "runner_number", "finishing_position",
        "barrier_draw", "weight_kg",
        "jockey", "trainer", "jt_pair",
        "fixed_odds_win", "fixed_odds_place",
        "fixed_odds_win_open", "odds_drift", "odds_drift_pct",
        "is_favourite_win", "betting_status",
        "last_5_starts", "wins_last5", "places_last5",
        "win_rate_last5", "place_rate_last5", "jt_pair_win_rate",
        "form_rating_dfs", "early_speed_rating", "early_speed_band",
    ]
    # Only keep columns that actually exist
    output_cols = [c for c in output_cols if c in df.columns]
    df = df[output_cols]

    # Drop rows with missing core data
    core_cols = [
        "runner_name", "barrier_draw", "weight_kg",
        "jockey", "trainer", "fixed_odds_win",
    ]
    pre = len(df)
    df = df.dropna(subset=core_cols)
    print(f"    After dropping incomplete rows: {len(df)} (dropped {pre - len(df)})")

    return df


if __name__ == "__main__":
    df = build_matrix()
    if df.empty:
        raise SystemExit(1)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n[OK] Wrote {len(df)} rows x {len(df.columns)} cols to {OUTPUT_CSV}")
    print(f"\nPreview:\n{df.to_string(max_rows=10, max_cols=12)}")
