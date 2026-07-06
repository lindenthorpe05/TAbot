"""
tools.py - Ollama-compatible tool wrappers for interceptor, parser, and brain.

Each function returns a JSON-serializable dict summarising what happened.
The TOOL_SCHEMAS list provides Ollama tool-call definitions.
"""

import asyncio
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent

# ============================================================================
# Tool definitions (Ollama format)
# ============================================================================

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "scan_race",
            "description": (
                "Launch the browser interceptor to scrape live race data from "
                "TAB.com.au. Captures runner details, odds, form ratings, and "
                "market data. Use this when the user wants to analyze a specific "
                "race or today's races."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": (
                            "Optional TAB.com.au race URL. If not provided, "
                            "scrapes the default meetings page."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_next_races",
            "description": (
                "Scan ALL races starting within a given time window. "
                "Navigates to the TAB meetings page, finds every race with "
                "a start time in the next N minutes, then visits each race "
                "page to capture full runner and odds data. Use this when "
                "the user asks about upcoming races, best bets across "
                "multiple races, or wants a broad scan of what's coming up."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "minutes": {
                        "type": "integer",
                        "description": (
                            "How many minutes ahead to look for upcoming "
                            "races. Defaults to 30."
                        ),
                    }
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "parse_data",
            "description": (
                "Parse the raw intercepted race data into a structured race "
                "matrix CSV. Extracts runner names, barriers, weights, jockeys, "
                "trainers, odds, form ratings, and computes derived metrics like "
                "odds drift and win rates. Must be called after scan_race."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "predict_winners",
            "description": (
                "Run the XGBoost prediction model on the parsed race matrix. "
                "Trains on historical data, scores each runner's true win "
                "probability, compares against TAB implied odds, and identifies "
                "value bets (EDGE_FOUND). Must be called after parse_data."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_race_matrix",
            "description": (
                "Read the current parsed race matrix and return the full runner "
                "data — names, barriers, weights, jockeys, trainers, odds, form "
                "ratings, and derived metrics. Use this when the user wants to "
                "browse or discuss the parsed data without running predictions, "
                "e.g. 'show me the runners', 'who are the favourites', "
                "'what does the form look like'. Optionally filter to a specific "
                "track or race number."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "track": {
                        "type": "string",
                        "description": "Filter rows to this track name (case-insensitive, partial match). Omit for all tracks.",
                    },
                    "race_number": {
                        "type": "integer",
                        "description": "Filter rows to this race number. Omit for all races.",
                    },
                },
                "required": [],
            },
        },
    },
]


# ============================================================================
# Tool implementations
# ============================================================================

def scan_race(url: str | None = None) -> dict:
    """Run the interceptor and return a summary of captured data."""
    from interceptor import run as interceptor_run, OUTPUT_FILE

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(interceptor_run(target_url=url))
    finally:
        loop.close()

    # Summarise what was captured
    if OUTPUT_FILE.exists():
        data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        racing = data.get("racing_data", [])
        ws = data.get("websocket_messages", [])

        tracks = set()
        race_count = 0
        runner_count = 0
        for entry in racing:
            d = entry.get("data", {})
            if isinstance(d, dict) and "runners" in d:
                race_count += 1
                runner_count += len(d["runners"])
                meeting = d.get("meeting", {})
                if meeting.get("meetingName"):
                    tracks.add(meeting["meetingName"])

        return {
            "status": "success",
            "api_responses": len(racing),
            "ws_messages": len(ws),
            "races_found": race_count,
            "runners_found": runner_count,
            "tracks": sorted(tracks),
        }

    return {"status": "error", "message": "No data captured by interceptor."}


def scan_next_races(minutes: int = 30) -> dict:
    """Scan all races starting within the given time window."""
    from interceptor import run_multi, OUTPUT_FILE

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        loop.run_until_complete(run_multi(minutes=minutes))
    finally:
        loop.close()

    # Summarise what was captured
    if OUTPUT_FILE.exists():
        data = json.loads(OUTPUT_FILE.read_text(encoding="utf-8"))
        racing = data.get("racing_data", [])
        ws = data.get("websocket_messages", [])

        tracks = set()
        race_count = 0
        runner_count = 0
        for entry in racing:
            d = entry.get("data", {})
            if isinstance(d, dict) and "runners" in d:
                race_count += 1
                runner_count += len(d["runners"])
                meeting = d.get("meeting", {})
                if meeting.get("meetingName"):
                    tracks.add(meeting["meetingName"])

        return {
            "status": "success",
            "minutes_window": minutes,
            "api_responses": len(racing),
            "races_found": race_count,
            "runners_found": runner_count,
            "tracks": sorted(tracks),
        }

    return {"status": "error", "message": "No data captured by interceptor."}


def parse_data() -> dict:
    """Run the parser and return a summary of the race matrix."""
    from parser import build_matrix, OUTPUT_CSV

    df = build_matrix()
    if df.empty:
        return {"status": "error", "message": "Parser returned no data. Run scan_race first."}

    df.to_csv(OUTPUT_CSV, index=False)

    tracks = df["track_name"].unique().tolist() if "track_name" in df.columns else []
    races = int(df["race_number"].nunique()) if "race_number" in df.columns else 0

    # Report which races have already run vs are upcoming
    race_statuses = {}
    if "race_status" in df.columns and "race_number" in df.columns:
        for rnum, grp in df.groupby("race_number"):
            race_statuses[int(rnum)] = grp["race_status"].iloc[0]

    finished = [r for r, s in race_statuses.items() if s in ("Paying", "Interim")]
    upcoming = [r for r, s in race_statuses.items() if s not in ("Paying", "Interim", "Abandoned")]

    return {
        "status": "success",
        "runners": len(df),
        "races": races,
        "tracks": tracks,
        "finished_races": finished,
        "upcoming_races": upcoming,
        "columns": list(df.columns),
        "csv_path": str(OUTPUT_CSV),
    }


def predict_winners() -> dict:
    """Train the model and predict current race runners."""
    from brain import (
        train_model, predict_current, load_historical,
        generate_synthetic_historical, HISTORICAL_CSV, CURRENT_CSV,
    )

    if not CURRENT_CSV.exists():
        return {"status": "error", "message": "No race matrix found. Run parse_data first."}

    # Load or generate historical data
    if HISTORICAL_CSV.exists():
        hist_df = load_historical(HISTORICAL_CSV)
    else:
        hist_df = generate_synthetic_historical(n_races=500)
        hist_df.to_csv(HISTORICAL_CSV, index=False)

    model = train_model(hist_df)
    results = predict_current(model, CURRENT_CSV)

    # Build prediction summaries for the LLM to reason over
    predictions = []
    for _, row in results.sort_values("true_win_prob_pct", ascending=False).iterrows():
        odds = row.get("fixed_odds_win", 0)
        pred = {
            "track": row.get("track_name", ""),
            "race_number": int(row.get("race_number", 0)),
            "race_distance": row.get("race_distance", ""),
            "runner_name": row.get("runner_name", ""),
            "runner_number": int(row.get("runner_number", 0)),
            "barrier": int(row.get("barrier_draw", 0)),
            "weight_kg": float(row.get("weight_kg", 0)),
            "jockey": row.get("jockey", ""),
            "trainer": row.get("trainer", ""),
            "odds": float(odds),
            "implied_prob_pct": round(float(row.get("implied_prob", 0)) * 100, 2),
            "model_prob_pct": round(float(row.get("true_win_prob_pct", 0)), 2),
            "edge_pct": round(float(row.get("edge_pct", 0)), 2),
            "signal": row.get("signal", ""),
        }
        predictions.append(pred)

    edge_count = sum(1 for p in predictions if p["signal"] == "EDGE_FOUND")

    return {
        "status": "success",
        "total_runners": len(predictions),
        "edges_found": edge_count,
        "predictions": predictions,
    }


def inspect_race_matrix(
    track: str | None = None,
    race_number: int | None = None,
) -> dict:
    """Return the parsed race matrix rows so the agent can reason over them."""
    from parser import OUTPUT_CSV

    if not OUTPUT_CSV.exists():
        return {
            "status": "error",
            "message": "No race matrix found. Run parse_data first.",
        }

    import pandas as pd

    df = pd.read_csv(OUTPUT_CSV)

    if track:
        mask = df["track_name"].str.contains(track, case=False, na=False)
        df = df[mask]

    if race_number is not None:
        df = df[df["race_number"] == race_number]

    if df.empty:
        return {"status": "error", "message": "No runners matched the filter."}

    runner_cols = [
        "runner_number", "runner_name", "finishing_position",
        "barrier_draw", "weight_kg", "jockey", "trainer",
        "fixed_odds_win", "is_favourite_win", "betting_status",
        "last_5_starts", "win_rate_last5", "place_rate_last5",
        "form_rating_dfs", "early_speed_band", "odds_drift",
    ]
    runner_cols = [c for c in runner_cols if c in df.columns]

    races = []
    for (track_name, rnum), group in df.groupby(
        ["track_name", "race_number"], sort=True
    ):
        race_status = group["race_status"].iloc[0] if "race_status" in group else ""
        finished = race_status in ("Paying", "Interim", "Abandoned")

        runners = group[runner_cols].to_dict(orient="records")

        # Sort by finishing position for completed races, else by runner number
        if finished:
            runners = sorted(
                runners,
                key=lambda r: (r.get("finishing_position") or 99),
            )

        race_entry = {
            "track": track_name,
            "race_number": int(rnum),
            "race_name": group["race_name"].iloc[0] if "race_name" in group else "",
            "distance": group["race_distance"].iloc[0] if "race_distance" in group else "",
            "condition": group["track_condition"].iloc[0] if "track_condition" in group else "",
            "race_status": race_status,
            "runners": runners,
        }

        if finished:
            race_entry["win_dividend"] = group["win_dividend"].iloc[0] if "win_dividend" in group.columns else None
            race_entry["place_dividend"] = group["place_dividend"].iloc[0] if "place_dividend" in group.columns else None

        races.append(race_entry)

    return {
        "status": "success",
        "total_runners": len(df),
        "total_races": len(races),
        "races": races,
    }


# Dispatch map for agent.py
TOOL_DISPATCH = {
    "scan_race": scan_race,
    "scan_next_races": scan_next_races,
    "parse_data": parse_data,
    "predict_winners": predict_winners,
    "inspect_race_matrix": inspect_race_matrix,
}
