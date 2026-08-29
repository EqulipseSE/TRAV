"""Streamlit-sida för att filtrera V75/V85/V86-omgångar och streckgrupper.

Kör med::

    streamlit run streamlit_app/app.py

Sidan läser den flatta CSV:n som skapas av ``atg_favorites.flatten``
(standard: ``data/processed/races.csv``) - en rad per lopp - och låter dig:

1. Filtrera omgångarna (spelform, bana, datum, antal startande, streck-%
   på favoriten, skor m.m.) i sidopanelen.
2. Justera streck-grupperna (bucketar) för favoritfall-analysen och se hur
   favoritens faktiska vinstchans jämfört med den streckimplicerade chansen
   ("streckjusterad favoritfall-analys") beror på gruppindelningen, direkt
   på det filtrerade urvalet.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from atg_favorites.analysis import favorite_bucket_analysis, favorite_surprises  # noqa: E402
from atg_favorites.config import DEFAULT_BUCKET_EDGES, RACES_CSV  # noqa: E402

st.set_page_config(page_title="V75/V85/V86 - Favoritanalys", page_icon="🐎", layout="wide")


@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["game_date"] = pd.to_datetime(df["game_date"])
    for col in ("favorite_won", "favorite_top3", "winner_is_favorite"):
        if col in df.columns:
            df[col] = df[col].astype(bool)
    return df


def _multiselect_all(label: str, options: list[str], key: str) -> list[str]:
    options = sorted(o for o in options if pd.notna(o))
    selected = st.sidebar.multiselect(label, options, default=options, key=key)
    return selected if selected else options


st.title("🐎 Favoritanalys för V75 / V85 / V86")
st.caption(
    "Streckjusterad favoritfall-analys: hur ofta vinner favoriten (högst streckprocent) "
    "jämfört med vad streckprocenten implicerar - grupperat på streck och andra filter."
)

st.sidebar.header("Datakälla")
csv_path = st.sidebar.text_input("Sökväg till races.csv", value=str(RACES_CSV))

if not Path(csv_path).exists():
    st.warning(
        f"Hittar ingen fil på `{csv_path}`. Kör först:\n\n"
        "```bash\npython -m atg_favorites.fetch --days-back 30\n"
        "python -m atg_favorites.flatten\n```"
    )
    st.stop()

races = load_data(csv_path)

st.sidebar.header("Filtrera omgångar")
game_types = _multiselect_all("Spelform", races["game_type"].unique().tolist(), "game_types")
tracks = _multiselect_all("Bana", races["track_name"].unique().tolist(), "tracks")

min_date, max_date = races["game_date"].min().date(), races["game_date"].max().date()
date_range = st.sidebar.date_input(
    "Datumintervall",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date, end_date = min_date, max_date

streck_range = st.sidebar.slider("Favoritens streckprocent", 0, 100, (0, 100))
min_starters, max_starters = int(races["num_starters"].min()), int(races["num_starters"].max())
starters_range = st.sidebar.slider(
    "Antal startande",
    min_starters,
    max_starters,
    (min_starters, max_starters),
)

only_favoritfall = st.sidebar.checkbox("Visa endast favoritfall (favoriten vann inte)", value=False)
shoe_options = st.sidebar.selectbox(
    "Favoritens skor (bak)",
    ["Alla", "Med skor bak", "Barfota bak", "Skoombyte"],
)

mask = (
    races["game_type"].isin(game_types)
    & races["track_name"].isin(tracks)
    & (races["game_date"].dt.date >= start_date)
    & (races["game_date"].dt.date <= end_date)
    & races["favorite_streck_pct"].between(*streck_range)
    & races["num_starters"].between(*starters_range)
)
if only_favoritfall:
    mask &= ~races["favorite_won"]
if shoe_options == "Med skor bak":
    mask &= races["favorite_shoes_back"] == True  # noqa: E712
elif shoe_options == "Barfota bak":
    mask &= races["favorite_shoes_back"] == False  # noqa: E712
elif shoe_options == "Skoombyte":
    mask &= races["favorite_shoes_changed"] == True  # noqa: E712

filtered = races.loc[mask].copy()

st.subheader(f"Filtrerade lopp ({len(filtered)} av {len(races)})")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Antal lopp", len(filtered))
col2.metric("Favoritens vinstfrekvens", f"{filtered['favorite_won'].mean() * 100:.1f}%" if len(filtered) else "-")
col3.metric("Favoritens topp-3-frekvens", f"{filtered['favorite_top3'].mean() * 100:.1f}%" if len(filtered) else "-")
col4.metric("Favoritfall (förlust)", f"{(1 - filtered['favorite_won'].mean()) * 100:.1f}%" if len(filtered) else "-")

with st.expander("Visa lopptabell", expanded=False):
    display_cols = [
        "game_date",
        "game_type",
        "track_name",
        "race_number",
        "favorite_horse_name",
        "favorite_post_position",
        "favorite_added_distance_m",
        "favorite_streck_pct",
        "favorite_shoes_front",
        "favorite_shoes_back",
        "favorite_sulky_type",
        "favorite_place",
        "favorite_won",
        "winner_horse_name",
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]
    st.dataframe(filtered[display_cols].sort_values("game_date", ascending=False), use_container_width=True)

st.sidebar.header("Streckgrupper (bucketar)")
bucket_width = st.sidebar.slider("Gruppbredd (procentenheter)", 5, 25, 10, step=5)
split_by_type = st.sidebar.checkbox("Dela upp per spelform", value=False)

bucket_edges = tuple(range(0, 101, bucket_width))
if bucket_edges[-1] != 100:
    bucket_edges = bucket_edges + (100,)
if len(bucket_edges) < 2:
    bucket_edges = DEFAULT_BUCKET_EDGES

st.subheader("Streckjusterad favoritfall-analys per grupp")
if filtered.empty:
    st.info("Inga lopp matchar filtren.")
else:
    group_by = ["game_type"] if split_by_type else None
    bucket_df = favorite_bucket_analysis(filtered, bucket_edges=bucket_edges, group_by=group_by)

    display_bucket_df = bucket_df.copy()
    for pct_col in ("win_rate", "top3_rate", "favoritfall_rate", "implied_win_prob"):
        display_bucket_df[pct_col] = (display_bucket_df[pct_col] * 100).round(1)
    display_bucket_df["avg_streck_pct"] = display_bucket_df["avg_streck_pct"].round(1)
    display_bucket_df["edge_vs_streck"] = (display_bucket_df["edge_vs_streck"] * 100).round(1)

    st.dataframe(display_bucket_df, use_container_width=True)

    chart_df = bucket_df.set_index("streck_bucket")[["win_rate", "implied_win_prob"]] * 100
    chart_df = chart_df.rename(
        columns={"win_rate": "Faktisk vinstfrekvens (%)", "implied_win_prob": "Streckimplicerad chans (%)"}
    )
    st.bar_chart(chart_df)

    st.caption(
        "`edge_vs_streck` = faktisk vinstfrekvens minus den streckimplicerade chansen. "
        "Positivt = favoriten vinner oftare än streckprocenten antyder i den gruppen, "
        "negativt = favoriten är överspelad (\"overbet\") i den gruppen."
    )

    st.subheader("Favoritfall att titta närmare på")
    min_surprise = st.slider("Minsta streckprocent för \"stark favorit\"", 20, 60, 35)
    surprises = favorite_surprises(filtered, min_streck_pct=min_surprise)
    if surprises.empty:
        st.info("Inga favoritfall med streck ≥ denna nivå i det filtrerade urvalet.")
    else:
        st.dataframe(surprises, use_container_width=True)
