"""Streckjusterad favoritfall-modell: logistisk regression för P(häst vinner).

Detta bygger ut den enkla bucket-uppslagningen i ``analysis.py`` (historisk
vinstprocent per streckintervall) med en riktig prediktionsmodell. Modellen
tränas på **varje startande häst** i historiska avgjorda lopp (inte bara
favoriten), med målvariabeln "vann loppet" och följande features:

* ``streck_pct`` - hästens streckprocent
* ``gap_to_rival`` - marginalen till den mest relevanta rivalen: för
  favoriten är det avståndet ner till tvåan (``favorite_streck_margin``),
  för alla andra är det (negativa) avståndet upp till ledaren
* ``post_position`` - startspår
* ``start_method`` - startsätt (volte/auto)
* ``added_distance_m`` - tillägg
* ``num_starters`` - fältstorlek
* ``track_condition`` - banförhållande
* ``shoes_changed`` - skoändring (fram eller bak)
* ``barfota_back`` - barfota bak
* ``driver_win_pct`` - kuskens vinstprocent innevarande år
* ``prize_low`` / ``prize_high`` - loppklass (prissummeintervall), parsat
  ur ``race.terms``
* ``is_mare_race`` / ``is_coldblood`` - loppklass (stolopp/kallblodslopp),
  parsat ur ``race.terms``

Eftersom modellen tränas på varje häst (inte bara favoriten) kan den även
användas för att skatta P(vinst) för icke-favoriter och jämföra det mot
deras streckprocent - det är så "värde" bland icke-favoriter identifieras.

Modellen tränas på data äldre än ``--validation-days`` (standard 365, dvs.
"äldre data") och valideras på de senaste 12 månaderna, med kalibrering
(förutsagd vs. faktisk vinstfrekvens per sannolikhetsbucket) redovisad på
just den valideringsdatan.

Run as a script::

    python -m atg_favorites.favorite_model --raw-dir data/raw --date 2026-08-29
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from atg_favorites.api_client import AtgApiError, AtgClient
from atg_favorites.config import FINISHED_STATUSES, GAME_TYPES, RAW_DIR
from atg_favorites.extraction import start_summary
from atg_favorites.flatten import load_raw_games
from atg_favorites.leg_analysis import find_candidate_games, leg_starters, resolve_leg

logger = logging.getLogger(__name__)

NUMERIC_FEATURES = [
    "streck_pct",
    "gap_to_rival",
    "post_position",
    "added_distance_m",
    "num_starters",
    "driver_win_pct",
    "prize_low",
    "prize_high",
]
CATEGORICAL_FEATURES = ["start_method", "track_condition"]
BOOLEAN_FEATURES = ["shoes_changed", "barfota_back", "is_mare_race", "is_coldblood"]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES + BOOLEAN_FEATURES

#: Default validation window: the most recent 12 months ("senaste 12 månaderna").
DEFAULT_VALIDATION_DAYS = 365
DEFAULT_MIN_TRAINING_ROWS = 200

_MARE_RE = re.compile(r"\bston\b", re.IGNORECASE)
_COLDBLOOD_RE = re.compile(r"kallblod", re.IGNORECASE)
_PRIZE_RANGE_RE = re.compile(r"(\d[\d.]*)\s*-\s*(\d[\d.]*)\s*kr")
_PRIZE_MIN_RE = re.compile(r"lägst\s+(\d[\d.]*)\s*kr", re.IGNORECASE)
_PRIZE_MAX_RE = re.compile(r"högst\s+(\d[\d.]*)\s*kr", re.IGNORECASE)


def _parse_sek(text: str) -> float | None:
    try:
        return float(text.replace(".", ""))
    except ValueError:  # pragma: no cover - defensive
        return None


def parse_race_class(terms: list[str] | None) -> dict[str, Any]:
    """Parse loppklass (prissumma, sto-/kallblodslopp) out of ``race['terms']``.

    ``terms`` is the list of free-text Swedish condition strings ATG attaches
    to each race, e.g. ``"3-åriga och äldre ston 300.001 - 1.950.000 kr. ..."``.
    """
    text = " ".join(terms or [])

    prize_low: float | None = None
    prize_high: float | None = None

    range_match = _PRIZE_RANGE_RE.search(text)
    if range_match:
        prize_low = _parse_sek(range_match.group(1))
        prize_high = _parse_sek(range_match.group(2))
    else:
        min_match = _PRIZE_MIN_RE.search(text)
        if min_match:
            prize_low = _parse_sek(min_match.group(1))
        max_match = _PRIZE_MAX_RE.search(text)
        if max_match:
            prize_high = _parse_sek(max_match.group(1))

    return {
        "prize_low": prize_low,
        "prize_high": prize_high,
        "is_mare_race": bool(_MARE_RE.search(text)),
        "is_coldblood": bool(_COLDBLOOD_RE.search(text)),
    }


def compute_gap_to_rival(streck: pd.Series) -> pd.Series:
    """Signed margin to the "most relevant rival" for every horse in one race.

    For the leader (highest streckprocent) this is its margin over the
    runner-up (positive - the classic "favorite_streck_margin"). For every
    other horse it is the (negative) gap up to the leader.
    """
    if len(streck) < 2:
        return pd.Series(0.0, index=streck.index)
    top_two = streck.nlargest(2)
    leader_value = top_two.iloc[0]
    runner_up_value = top_two.iloc[1]
    leader_mask = streck == leader_value
    gap = streck - leader_value
    gap = gap.mask(leader_mask, leader_value - runner_up_value)
    return gap


def build_start_dataset(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """One row per non-scratched starter (with streck data) in every finished race."""
    rows: list[dict[str, Any]] = []
    for _path, game in load_raw_games(raw_dir):
        pool_key = game.get("type")
        for race in game.get("races") or []:
            if race.get("status") not in FINISHED_STATUSES or not pool_key:
                continue
            starts = race.get("starts") or []
            if not starts:
                continue

            summaries = [start_summary(s, race, pool_key) for s in starts]
            runners = [s for s in summaries if not s["scratched"] and s["streck_pct"] is not None]
            if len(runners) < 2:
                continue

            streck_series = pd.Series({s["number"]: s["streck_pct"] for s in runners})
            gaps = compute_gap_to_rival(streck_series)
            race_class = parse_race_class(race.get("terms"))
            track = race.get("track") or {}

            for s in runners:
                rows.append(
                    {
                        "game_id": game.get("id"),
                        "race_id": race.get("id"),
                        "game_date": race.get("date"),
                        "game_type": pool_key,
                        "track_name": track.get("name"),
                        "horse_id": s["horse_id"],
                        "horse_name": s["horse_name"],
                        "number": s["number"],
                        "streck_pct": s["streck_pct"],
                        "gap_to_rival": float(gaps.loc[s["number"]]),
                        "post_position": s["post_position"],
                        "start_method": race.get("startMethod"),
                        "added_distance_m": s["added_distance_m"] or 0,
                        "num_starters": len(runners),
                        "track_condition": track.get("condition"),
                        "shoes_changed": float(bool(s["shoes_changed"])),
                        "barfota_back": float(s["shoes_back"] is False),
                        "driver_win_pct": s["driver_win_pct"],
                        **{k: (float(v) if k in ("is_mare_race", "is_coldblood") else v) for k, v in race_class.items()},
                        "won": s["place"] == 1,
                    }
                )
    return pd.DataFrame(rows)


def build_pipeline() -> Pipeline:
    """A logistic regression with proper preprocessing for the feature set above.

    No ``class_weight="balanced"`` on purpose: reweighting classes would
    distort the predicted probabilities away from true win frequencies,
    which is exactly what the calibration check below is meant to verify.
    """
    numeric_transformer = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    categorical_transformer = Pipeline(
        [
            ("impute", SimpleImputer(strategy="constant", fill_value="unknown")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    boolean_transformer = SimpleImputer(strategy="constant", fill_value=0)

    preprocessor = ColumnTransformer(
        [
            ("numeric", numeric_transformer, NUMERIC_FEATURES),
            ("categorical", categorical_transformer, CATEGORICAL_FEATURES),
            ("boolean", boolean_transformer, BOOLEAN_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            ("classify", LogisticRegression(max_iter=1000)),
        ]
    )


def time_split(
    df: pd.DataFrame,
    validation_days: int = DEFAULT_VALIDATION_DAYS,
    as_of: pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into (older) training data and the most recent ``validation_days``."""
    dates = pd.to_datetime(df["game_date"])
    as_of = pd.Timestamp(as_of) if as_of is not None else dates.max()
    cutoff = as_of - pd.Timedelta(days=validation_days)
    train = df.loc[dates < cutoff].reset_index(drop=True)
    validation = df.loc[dates >= cutoff].reset_index(drop=True)
    return train, validation


def train_model(train_df: pd.DataFrame) -> Pipeline:
    pipeline = build_pipeline()
    X = train_df[ALL_FEATURES]
    y = train_df["won"].astype(int)
    pipeline.fit(X, y)
    return pipeline


def evaluate(pipeline: Pipeline, validation_df: pd.DataFrame) -> dict[str, float]:
    """Headline metrics for ``pipeline`` on ``validation_df`` (log-loss, Brier, AUC)."""
    X = validation_df[ALL_FEATURES]
    y = validation_df["won"].astype(int)
    proba = pipeline.predict_proba(X)[:, 1]
    metrics = {
        "n": int(len(validation_df)),
        "base_rate": float(y.mean()) if len(y) else float("nan"),
        "log_loss": float(log_loss(y, proba, labels=[0, 1])) if len(y) else float("nan"),
        "brier_score": float(brier_score_loss(y, proba)) if len(y) else float("nan"),
    }
    metrics["roc_auc"] = float(roc_auc_score(y, proba)) if y.nunique() > 1 else float("nan")
    return metrics


def calibration_table(pipeline: Pipeline, validation_df: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Predicted-vs-actual win rate per probability bucket, on the validation set."""
    X = validation_df[ALL_FEATURES]
    y = validation_df["won"].astype(int).to_numpy()
    proba = pipeline.predict_proba(X)[:, 1]

    table = pd.DataFrame({"predicted": proba, "actual": y})
    try:
        table["bucket"] = pd.qcut(table["predicted"], q=n_bins, duplicates="drop")
    except ValueError:
        table["bucket"] = pd.cut(table["predicted"], bins=1)

    grouped = (
        table.groupby("bucket", observed=True)
        .agg(n=("actual", "size"), predicted_mean=("predicted", "mean"), actual_rate=("actual", "mean"))
        .reset_index()
    )
    grouped["calibration_gap"] = grouped["actual_rate"] - grouped["predicted_mean"]
    return grouped.sort_values("predicted_mean").reset_index(drop=True)


def build_features_for_leg(starters: pd.DataFrame, race: dict[str, Any]) -> pd.DataFrame:
    """Build the same feature columns used in training, for one live/upcoming leg."""
    df = starters.loc[~starters["scratched"] & starters["streck_pct"].notna()].copy()
    if df.empty:
        return df

    gaps = compute_gap_to_rival(df.set_index("number")["streck_pct"])
    df["gap_to_rival"] = df["number"].map(gaps).astype(float)
    df["num_starters"] = len(df)
    df["start_method"] = race.get("startMethod")

    track = race.get("track") or {}
    df["track_condition"] = track.get("condition")

    race_class = parse_race_class(race.get("terms"))
    for key, value in race_class.items():
        if key in ("is_mare_race", "is_coldblood"):
            value = float(value)
        df[key] = value

    df["barfota_back"] = (df["shoes_back"] == False).astype(float)  # noqa: E712 - explicit tri-state comparison
    df["shoes_changed"] = pd.to_numeric(df["shoes_changed"], errors="coerce").fillna(0.0)
    df["added_distance_m"] = df["added_distance_m"].fillna(0)
    return df


def score_leg(pipeline: Pipeline, starters: pd.DataFrame, race: dict[str, Any]) -> pd.DataFrame:
    """Score every non-scratched starter in one leg; adds model_prob (field-normalised)."""
    features_df = build_features_for_leg(starters, race)
    if features_df.empty:
        return features_df

    X = features_df[ALL_FEATURES]
    features_df["model_prob_raw"] = pipeline.predict_proba(X)[:, 1]
    total = features_df["model_prob_raw"].sum()
    features_df["model_prob"] = (
        features_df["model_prob_raw"] / total if total > 0 else features_df["model_prob_raw"]
    )
    features_df["implied_prob"] = features_df["streck_pct"] / 100.0
    features_df["value_diff"] = features_df["model_prob"] - features_df["implied_prob"]
    features_df["value_ratio"] = features_df["model_prob"] / features_df["implied_prob"].replace(0, pd.NA)
    return features_df.sort_values("model_prob", ascending=False)


def top_value_non_favorites(scored: pd.DataFrame, top_n: int = 3) -> pd.DataFrame:
    """The ``top_n`` non-favorite starters the model values highest relative to their streck."""
    if scored.empty:
        return scored
    favorite_number = scored.loc[scored["streck_pct"].idxmax(), "number"]
    non_favorites = scored.loc[scored["number"] != favorite_number]
    return non_favorites.sort_values("value_ratio", ascending=False).head(top_n)


def iter_todays_legs(
    client: AtgClient,
    date_str: str,
    game_type: str | None = None,
) -> list[tuple[str, int]]:
    """Every ``(game_type, avd)`` pair scheduled on ``date_str``."""
    candidates = find_candidate_games(client, date_str, game_type=game_type)
    legs: list[tuple[str, int]] = []
    for candidate in candidates:
        num_legs = len(candidate.get("races") or [])
        legs.extend((candidate["type"], avd) for avd in range(1, num_legs + 1))
    return legs


def format_calibration_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "  (inget valideringsunderlag)"
    lines = ["  Predikterad %  Faktisk %  Gap (pp)  N"]
    for _, row in table.iterrows():
        lines.append(
            f"  {row['predicted_mean'] * 100:11.1f}  {row['actual_rate'] * 100:9.1f}  "
            f"{row['calibration_gap'] * 100:+8.1f}  {int(row['n']):3d}"
        )
    return "\n".join(lines)


def format_value_picks(client: AtgClient, date_str: str, pipeline: Pipeline, game_type: str | None = None) -> str:
    legs = iter_todays_legs(client, date_str, game_type=game_type)
    if not legs:
        return f"Inga V75/V85/V86-avdelningar hittades för {date_str}."

    lines = [f"Värdespel bland icke-favoriter, {date_str}:"]
    for gtype, avd in legs:
        try:
            target = resolve_leg(client, date_str, avd, game_type=gtype)
            starters = leg_starters(target)
            scored = score_leg(pipeline, starters, target.race)
        except AtgApiError as exc:
            logger.warning("Kunde inte analysera %s avd %d: %s", gtype, avd, exc)
            continue
        if scored.empty:
            continue

        favorite = scored.loc[scored["streck_pct"].idxmax()]
        picks = top_value_non_favorites(scored, top_n=3)

        track_name = (target.race.get("track") or {}).get("name", "?")
        lines.append(f"\n{gtype} avd {avd} - {track_name} (favorit: {favorite['horse_name']} {favorite['streck_pct']:.1f}%)")
        for _, pick in picks.iterrows():
            lines.append(
                f"  {int(pick['number']):>2}. {pick['horse_name']:<20} streck {pick['streck_pct']:5.1f}%  "
                f"modell {pick['model_prob'] * 100:5.1f}%  value x{pick['value_ratio']:.2f}"
            )
    return "\n".join(lines)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validation-days", type=int, default=DEFAULT_VALIDATION_DAYS)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--date", type=str, default=None, help="YYYY-MM-DD, default: idag.")
    parser.add_argument("--game-type", choices=list(GAME_TYPES), default=None)
    parser.add_argument("--skip-today", action="store_true", help="Skippa värdespels-delen (bara träna/validera).")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    import datetime as _dt

    args = _parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s %(message)s")

    dataset = build_start_dataset(args.raw_dir)
    if dataset.empty:
        logger.warning("Inga startande hittades i %s - kör fetch/flatten först.", args.raw_dir)
        return

    train_df, validation_df = time_split(dataset, validation_days=args.validation_days)
    logger.info(
        "Träningsdata: %d startande (%s till %s), valideringsdata (senaste %d dagarna): %d startande",
        len(train_df),
        train_df["game_date"].min() if not train_df.empty else "-",
        train_df["game_date"].max() if not train_df.empty else "-",
        args.validation_days,
        len(validation_df),
    )

    if len(train_df) < DEFAULT_MIN_TRAINING_ROWS or train_df["won"].nunique() < 2:
        logger.warning(
            "För lite (eller för homogen) träningsdata (%d rader) - hämta mer historik med "
            "`atg_favorites.fetch` för en meningsfull modell. Fortsätter i alla fall.",
            len(train_df),
        )
    if train_df.empty or train_df["won"].nunique() < 2:
        logger.error("Kan inte träna: ingen träningsdata äldre än valideringsfönstret.")
        return

    pipeline = train_model(train_df)

    if not validation_df.empty and validation_df["won"].nunique() > 1:
        metrics = evaluate(pipeline, validation_df)
        logger.info("Valideringsmått: %s", metrics)
        print("Kalibrering på valideringsdata (senaste 12 månaderna):")
        print(format_calibration_table(calibration_table(pipeline, validation_df, n_bins=args.bins)))
    else:
        logger.warning("Inget (eller endimensionellt) valideringsunderlag - kalibrering kan inte redovisas.")

    if args.skip_today:
        return

    date_str = args.date or _dt.date.today().isoformat()
    client = AtgClient()
    print()
    print(format_value_picks(client, date_str, pipeline, game_type=args.game_type))


if __name__ == "__main__":
    main()
