"""
brain.py - XGBoost classifier to predict horse race win probability.

Training:   Loads 'historical_results.csv' with labelled outcomes (win_flag).
Inference:  Loads 'current_race_matrix.csv' from the parser module,
            predicts P(win) for each runner, compares against TAB implied
            probability, and flags value bets as EDGE_FOUND.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import log_loss, brier_score_loss

# -- File paths --------------------------------------------------------------
BASE_DIR = Path(__file__).parent
HISTORICAL_CSV = BASE_DIR / "historical_results.csv"
CURRENT_CSV = BASE_DIR / "current_race_matrix.csv"
OUTPUT_CSV = BASE_DIR / "predictions.csv"
MODEL_FILE = BASE_DIR / "xgb_model.json"

# -- Feature configuration ---------------------------------------------------
# Categorical features that need encoding
CAT_FEATURES = ["track_condition", "race_class", "early_speed_band"]

# Numeric features used for training and inference
NUM_FEATURES = [
    "barrier_draw",
    "weight_kg",
    "race_distance",
    "win_rate_last5",
    "place_rate_last5",
    "jt_pair_win_rate",
    "form_rating_dfs",
    "early_speed_rating",
    "odds_drift",
    "field_size",
    "barrier_pct",        # barrier / field_size (relative draw)
    "weight_vs_median",   # weight - race median weight
]

TARGET = "win_flag"


# ============================================================================
# Feature Engineering
# ============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric features and encode categoricals.

    Works identically on both historical and live data.
    """
    df = df.copy()

    # -- Field size per race -------------------------------------------------
    race_key_cols = [c for c in ["meeting_date", "track_name", "race_number"] if c in df.columns]
    if race_key_cols:
        df["field_size"] = df.groupby(race_key_cols)["runner_name"].transform("count")
    else:
        df["field_size"] = len(df)

    # -- Relative barrier position (0-1 scale) -------------------------------
    df["barrier_pct"] = (
        df["barrier_draw"] / df["field_size"].replace(0, np.nan)
    ).round(4)

    # -- Weight relative to race median --------------------------------------
    if race_key_cols:
        race_median_wt = df.groupby(race_key_cols)["weight_kg"].transform("median")
    else:
        race_median_wt = df["weight_kg"].median()
    df["weight_vs_median"] = (df["weight_kg"] - race_median_wt).round(2)

    # -- Encode categoricals as integer codes --------------------------------
    for col in CAT_FEATURES:
        if col in df.columns:
            df[col + "_enc"] = df[col].astype("category").cat.codes
        else:
            df[col + "_enc"] = 0

    # -- Fill odds_drift NaN (runners without opening odds) ------------------
    df["odds_drift"] = df["odds_drift"].fillna(0.0)

    return df


def get_feature_cols() -> list[str]:
    """Return the ordered list of feature columns for the model."""
    encoded_cats = [c + "_enc" for c in CAT_FEATURES]
    return NUM_FEATURES + encoded_cats


# ============================================================================
# Training
# ============================================================================

def load_historical(path: Path) -> pd.DataFrame:
    """Load and validate historical results CSV."""
    df = pd.read_csv(path)
    required = {"runner_name", "barrier_draw", "weight_kg", "win_rate_last5", TARGET}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"historical_results.csv is missing columns: {missing}")
    return df


def train_model(df: pd.DataFrame) -> xgb.XGBClassifier:
    """Train an XGBoost classifier on the historical dataset."""
    df = engineer_features(df)
    feature_cols = get_feature_cols()

    X = df[feature_cols].fillna(0)
    y = df[TARGET]

    print(f"[*] Training on {len(X)} samples, {len(feature_cols)} features")
    print(f"    Win rate in data: {y.mean():.3f}")

    # Class weight to handle imbalance (typically ~1 winner per 10-14 runners)
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos = n_neg / max(n_pos, 1)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos,
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        verbosity=0,
    )

    # Cross-validate before final fit
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_log_loss")
    print(f"    CV log-loss: {-scores.mean():.4f} (+/- {scores.std():.4f})")

    model.fit(X, y)

    # Feature importance
    imp = pd.Series(model.feature_importances_, index=feature_cols)
    imp = imp.sort_values(ascending=False)
    print("\n    Feature importance (top 8):")
    for feat, val in imp.head(8).items():
        print(f"      {feat:25s} {val:.4f}")

    # Save model
    model.save_model(str(MODEL_FILE))
    print(f"\n[OK] Model saved to {MODEL_FILE}")

    return model


# ============================================================================
# Inference
# ============================================================================

def predict_current(model: xgb.XGBClassifier, path: Path = CURRENT_CSV) -> pd.DataFrame:
    """Score the current race matrix and flag value bets."""
    df = pd.read_csv(path)
    print(f"\n[*] Scoring {len(df)} runners from {path.name}")

    df = engineer_features(df)
    feature_cols = get_feature_cols()
    X = df[feature_cols].fillna(0)

    # Predicted P(win)
    df["model_win_prob"] = model.predict_proba(X)[:, 1]

    # -- Implied probability from TAB fixed odds -----------------------------
    # Implied prob = 1 / decimal_odds (includes the bookmaker margin)
    df["implied_prob"] = (1.0 / df["fixed_odds_win"].replace(0, np.nan)).round(4)

    # -- True win probability (model, normalised within race) ----------------
    # Normalise model probabilities within each race so they sum to ~1,
    # giving a cleaner "true" probability that removes model calibration bias.
    race_key_cols = [c for c in ["meeting_date", "track_name", "race_number"] if c in df.columns]
    if race_key_cols:
        race_total = df.groupby(race_key_cols)["model_win_prob"].transform("sum")
    else:
        race_total = df["model_win_prob"].sum()

    df["true_win_prob"] = (df["model_win_prob"] / race_total).round(4)
    df["true_win_prob_pct"] = (df["true_win_prob"] * 100).round(2)

    # -- Edge detection ------------------------------------------------------
    # If our model thinks the horse wins more often than the odds imply,
    # there's an overlay / value bet.
    df["edge"] = (df["true_win_prob"] - df["implied_prob"]).round(4)
    df["edge_pct"] = (df["edge"] * 100).round(2)
    df["signal"] = np.where(df["edge"] > 0, "EDGE_FOUND", "")

    return df


def display_results(df: pd.DataFrame) -> None:
    """Print a formatted results table."""
    show_cols = [
        "runner_number", "runner_name", "barrier_draw", "weight_kg",
        "jockey", "fixed_odds_win", "implied_prob",
        "true_win_prob_pct", "edge_pct", "signal",
    ]
    show_cols = [c for c in show_cols if c in df.columns]

    out = df[show_cols].copy()
    out = out.rename(columns={
        "runner_number": "#",
        "runner_name": "Runner",
        "barrier_draw": "Bar",
        "weight_kg": "Wt",
        "jockey": "Jockey",
        "fixed_odds_win": "Odds",
        "implied_prob": "Impl%",
        "true_win_prob_pct": "Model%",
        "edge_pct": "Edge%",
        "signal": "Signal",
    })
    out = out.sort_values("Model%", ascending=False)

    print("\n" + "=" * 100)
    print("RACE PREDICTIONS")
    print("=" * 100)

    race_info_cols = ["track_name", "race_number", "race_name", "race_distance", "track_condition"]
    for col in race_info_cols:
        if col in df.columns:
            val = df[col].iloc[0]
            print(f"  {col}: {val}")

    print("-" * 100)
    print(out.to_string(index=False))
    print("-" * 100)

    edges = df[df["signal"] == "EDGE_FOUND"]
    if not edges.empty:
        print(f"\n  >>> {len(edges)} VALUE BET(S) DETECTED:")
        for _, row in edges.sort_values("edge", ascending=False).iterrows():
            print(
                f"      {row['runner_name']:25s}  "
                f"Odds ${row['fixed_odds_win']:.2f}  "
                f"Model {row['true_win_prob_pct']:.1f}%  "
                f"vs Implied {row['implied_prob']*100:.1f}%  "
                f"Edge +{row['edge_pct']:.1f}%"
            )
    else:
        print("\n  No value bets found in this race.")

    print()


# ============================================================================
# Bootstrap: generate synthetic historical data for initial training
# ============================================================================

def generate_synthetic_historical(n_races: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic historical race data for bootstrapping.

    This should be replaced with real data from the interceptor once
    enough race results have been collected across multiple days.
    """
    rng = np.random.default_rng(seed)

    track_conditions = ["Good4", "Good3", "Soft5", "Soft6", "Soft7", "HVY8", "HVY9", "HVY10", "Firm1", "Firm2"]
    race_classes = ["CL1", "CL2", "CL3", "CL4", "CL5", "CL6", "BM58", "BM64", "BM70", "BM78", "MDN", "MSW"]
    speed_bands = ["LEADER", "MIDFIELD", "BACKMARKER"]
    distances = [1000, 1100, 1200, 1300, 1400, 1600, 1800, 2000, 2200, 2400]

    rows = []
    for race_id in range(n_races):
        field_size = rng.integers(6, 17)
        track_cond = rng.choice(track_conditions)
        race_class = rng.choice(race_classes)
        distance = rng.choice(distances)
        date = f"2025-{rng.integers(1,13):02d}-{rng.integers(1,29):02d}"

        # Generate runners for this race
        weights = rng.uniform(54.0, 62.0, size=field_size).round(1)
        barriers = rng.permutation(field_size) + 1

        # Simulate "true ability" for each runner to determine winner
        abilities = rng.normal(50, 15, size=field_size)

        # Better barrier (lower) gives slight edge
        abilities += (field_size / 2 - barriers) * 0.5
        # Higher form rating = better
        form_ratings = rng.integers(60, 105, size=field_size)
        abilities += (form_ratings - 80) * 0.3
        # Speed leaders have slight advantage at shorter distances
        speed_ratings = rng.integers(10, 100, size=field_size)
        speed_band_arr = np.array([speed_bands[min(int(s / 34), 2)] for s in speed_ratings])

        # Winner is the horse with highest simulated ability
        winner_idx = np.argmax(abilities)

        for i in range(field_size):
            win_rate = rng.uniform(0.0, 0.4)
            place_rate = min(win_rate + rng.uniform(0.1, 0.4), 1.0)
            jt_win = rng.uniform(0.0, 0.3)

            # Simulate realistic odds based on ability ranking
            rank = np.argsort(-abilities)
            position = np.where(rank == i)[0][0]
            # Favourite gets ~$2-4, longer shots $10-100+
            base_odds = 1.5 + (position / field_size) * 40 + rng.normal(0, 2)
            base_odds = max(1.2, base_odds)
            open_odds = base_odds + rng.normal(0, 1)
            open_odds = max(1.2, open_odds)

            rows.append({
                "meeting_date": date,
                "track_name": f"TRACK_{race_id % 30}",
                "race_number": (race_id % 8) + 1,
                "race_distance": distance,
                "track_condition": track_cond,
                "race_class": race_class,
                "runner_name": f"HORSE_{race_id}_{i}",
                "runner_number": i + 1,
                "barrier_draw": int(barriers[i]),
                "weight_kg": weights[i],
                "jockey": f"JOCKEY_{rng.integers(0, 80)}",
                "trainer": f"TRAINER_{rng.integers(0, 50)}",
                "jt_pair": f"JOCKEY_{rng.integers(0, 80)} / TRAINER_{rng.integers(0, 50)}",
                "fixed_odds_win": round(base_odds, 2),
                "fixed_odds_win_open": round(open_odds, 2),
                "odds_drift": round(base_odds - open_odds, 2),
                "win_rate_last5": round(win_rate, 4),
                "place_rate_last5": round(place_rate, 4),
                "jt_pair_win_rate": round(jt_win, 4),
                "form_rating_dfs": int(form_ratings[i]),
                "early_speed_rating": int(speed_ratings[i]),
                "early_speed_band": speed_band_arr[i],
                "win_flag": 1 if i == winner_idx else 0,
            })

    return pd.DataFrame(rows)


# ============================================================================
# Main
# ============================================================================

def main() -> None:
    # -- Step 1: Load or generate historical training data -------------------
    if HISTORICAL_CSV.exists():
        print(f"[*] Loading historical data from {HISTORICAL_CSV}")
        hist_df = load_historical(HISTORICAL_CSV)
    else:
        print(f"[!] {HISTORICAL_CSV} not found. Generating synthetic training data...")
        hist_df = generate_synthetic_historical(n_races=500)
        hist_df.to_csv(HISTORICAL_CSV, index=False)
        print(f"    Wrote {len(hist_df)} rows to {HISTORICAL_CSV}")
        print(f"    >>> Replace this with real collected results for better accuracy.\n")

    # -- Step 2: Train the model ---------------------------------------------
    model = train_model(hist_df)

    # -- Step 3: Score current race ------------------------------------------
    if not CURRENT_CSV.exists():
        print(f"\n[!] {CURRENT_CSV} not found. Run parser.py first.")
        sys.exit(1)

    results = predict_current(model, CURRENT_CSV)

    # -- Step 4: Output ------------------------------------------------------
    output_cols = [
        "meeting_date", "track_name", "race_number", "race_name",
        "race_distance", "track_condition",
        "runner_number", "runner_name", "barrier_draw", "weight_kg",
        "jockey", "trainer",
        "fixed_odds_win", "implied_prob",
        "true_win_prob_pct", "model_win_prob",
        "edge_pct", "signal",
    ]
    output_cols = [c for c in output_cols if c in results.columns]

    results[output_cols].to_csv(OUTPUT_CSV, index=False)
    print(f"[OK] Predictions saved to {OUTPUT_CSV}")

    display_results(results)


if __name__ == "__main__":
    main()
