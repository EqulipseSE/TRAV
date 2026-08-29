"""Streamlit-sida: analysera en avdelning (dagens omgång, avdelning för avdelning).

Väljer man datum, spelform, omgång och avdelning hämtas loppet live från
ATG (fungerar för kommande, spelbara, pågående och redan avgjorda lopp) och
jämförs mot jämförelselopp i det lokala rå-arkivet (``data/raw``) - hårda
villkor (hästtyp/köns­restriktion/körsätt) och stegvis uppslappnade mjuka
villkor (bana/distans/klass/startmetod), se ``leg_analysis.py``.
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atg_favorites import favorite_model, leg_analysis  # noqa: E402
from atg_favorites.api_client import AtgApiError, AtgClient  # noqa: E402
from atg_favorites.config import GAME_TYPES, RAW_DIR  # noqa: E402
from atg_favorites.leg_analysis import LegNotFoundError, build_leg_report  # noqa: E402

st.set_page_config(page_title="Analysera avdelning", page_icon="🔎", layout="wide")


@st.cache_resource
def get_client() -> AtgClient:
    return AtgClient(request_delay=0.2)


@st.cache_resource
def get_comparison_pool(raw_dir: str):
    """Bygg jämförelselopps-poolen (data/raw) en gång per session (cachad)."""
    return leg_analysis.build_comparison_pool(Path(raw_dir))


@st.cache_resource
def get_favorite_model(raw_dir: str):
    """Train the favoritfall-modellen once per session (cached)."""
    dataset = favorite_model.build_start_dataset(Path(raw_dir))
    if dataset.empty or dataset["won"].nunique() < 2:
        return None
    train_df, _ = favorite_model.time_split(dataset)
    if train_df.empty or train_df["won"].nunique() < 2:
        return None
    return favorite_model.train_model(train_df)


@st.cache_data(ttl=30, show_spinner="Hämtar dagens kalender från ATG...")
def load_calendar(date_str: str) -> dict:
    return get_client().get_calendar_day(date_str)


st.title("🔎 Analysera en avdelning")
st.caption(
    "Skicka in dagens (eller valfri annan) omgång, avdelning för avdelning, och få en "
    "grundlig analys: streckprocent, spår, tillägg, skor m.m. för alla startande, samt hur "
    "favoriten klarat sig i jämförelselopp - hästtyp/köns­restriktion/körsätt matchas alltid "
    "exakt, övriga villkor (bana/distans/klass) släpps stegvis om underlaget är för litet."
)

st.sidebar.header("Välj lopp")
selected_date = st.sidebar.date_input("Datum", value=date.today())
date_str = selected_date.isoformat()

try:
    calendar = load_calendar(date_str)
except AtgApiError as exc:
    st.error(f"Kunde inte hämta kalendern för {date_str}: {exc}")
    st.stop()

games_by_type = calendar.get("games", {})
available_types = [gt for gt in GAME_TYPES if games_by_type.get(gt)]

if not available_types:
    st.warning(f"Inga V75/V85/V86-omgångar hittades för {date_str}.")
    st.stop()

game_type = st.sidebar.selectbox("Spelform", available_types)
games_of_type = games_by_type[game_type]
game_options = {f"{g['id']} ({g.get('status', '?')})": g for g in games_of_type}
game_label = st.sidebar.selectbox("Omgång", list(game_options.keys()))
game_summary = game_options[game_label]
num_legs = len(game_summary.get("races", [])) or 1

avd = st.sidebar.number_input("Avdelning", min_value=1, max_value=num_legs, value=1, step=1)

with st.sidebar.expander("Avancerat"):
    raw_dir_path = st.text_input("Sökväg till rå-JSON (data/raw)", value=str(RAW_DIR))

analyze_clicked = st.sidebar.button("Analysera avdelning", type="primary", use_container_width=True)

if not analyze_clicked:
    st.info("Välj datum, spelform, omgång och avdelning i sidopanelen och klicka på **Analysera avdelning**.")
    st.stop()

try:
    with st.spinner(f"Hämtar och analyserar {game_type} avd {avd}..."):
        comparison_pool = get_comparison_pool(raw_dir_path)
        report = build_leg_report(
            Path(raw_dir_path),
            date_str=date_str,
            avd=int(avd),
            game_type=game_type,
            client=get_client(),
            pool=comparison_pool,
        )
except LegNotFoundError as exc:
    st.error(str(exc))
    st.stop()
except AtgApiError as exc:
    st.error(f"ATG-API-fel: {exc}")
    st.stop()

race = report.target.race
track = race.get("track") or {}

st.subheader(f"{report.target.game_type} avd {report.target.avd} - {track.get('name', '?')}")
st.write(race.get("name", ""))

meta_cols = st.columns(4)
meta_cols[0].metric("Distans", f"{race.get('distance', '?')} m")
meta_cols[1].metric("Startsätt", race.get("startMethod", "?"))
meta_cols[2].metric("Bankondition", track.get("condition", "?"))
meta_cols[3].metric("Antal startande", len(report.starters))

st.markdown("### Startlista")
column_labels = {
    "number": "Nr",
    "horse_name": "Häst",
    "streck_pct": "Streck %",
    "post_position": "Spår",
    "added_distance_m": "Tillägg (m)",
    "shoes_front": "Sko fram",
    "shoes_back": "Sko bak",
    "sulky_type": "Sulky",
    "driver_name": "Kusk",
    "driver_win_pct": "Kusk vinst% (år)",
    "trainer_name": "Tränare",
    "trainer_win_pct": "Tränare vinst% (år)",
    "record_time": "Rekord",
    "life_starts": "Starter (liv)",
    "life_win_pct": "Vinst% (liv)",
    "scratched": "Struken",
}
display_cols = [c for c in column_labels if c in report.starters.columns]
starters_display = report.starters[display_cols].rename(columns=column_labels)
st.dataframe(starters_display, use_container_width=True, hide_index=True)

if report.favorite:
    st.markdown(
        f"**Favorit:** {report.favorite.get('horse_name')} - streck "
        f"{report.favorite.get('streck_pct', 0):.1f}%, kusk {report.favorite.get('driver_name')}, "
        f"tränare {report.favorite.get('trainer_name')}"
    )
else:
    st.warning("Kunde inte avgöra favorit (ingen streckdata ännu).")

st.markdown("### Favoritfall-modellen: värdespel bland icke-favoriter")
model = get_favorite_model(raw_dir_path)
if model is None:
    st.info(
        "Ingen tränad favoritfall-modell tillgänglig än (för lite historik i "
        f"`{raw_dir_path}`). Kör `python -m atg_favorites.fetch` för att bygga upp arkivet."
    )
else:
    scored = favorite_model.score_leg(model, report.starters, race)
    if scored.empty:
        st.info("Kunde inte skatta modellen för denna avdelning (ingen streckdata).")
    else:
        fav_row = scored.loc[scored["streck_pct"].idxmax()]
        favoritfall_prob = 1 - fav_row["model_prob"]
        cols = st.columns(2)
        cols[0].metric("Modellens skattning för favoriten", f"{fav_row['model_prob'] * 100:.1f}%")
        cols[1].metric("Modellens favoritfall-sannolikhet", f"{favoritfall_prob * 100:.1f}%")

        picks = favorite_model.top_value_non_favorites(scored, top_n=3)
        st.caption(
            "De tre icke-favoriter modellen värderar högst relativt sin egen streckprocent "
            "(`value_ratio` = modellens sannolikhet delat med streckimplicerad sannolikhet)."
        )
        picks_display = picks[["number", "horse_name", "streck_pct", "model_prob", "value_ratio"]].rename(
            columns={
                "number": "Nr",
                "horse_name": "Häst",
                "streck_pct": "Streck %",
                "model_prob": "Modell %",
                "value_ratio": "Value (x streck)",
            }
        )
        picks_display["Modell %"] = (picks_display["Modell %"] * 100).round(1)
        picks_display["Value (x streck)"] = picks_display["Value (x streck)"].round(2)
        st.dataframe(picks_display, use_container_width=True, hide_index=True)

st.markdown("### Jämförelselopp")
stats = report.similarity_stats

st.caption(
    "Villkor som användes (hästtyp/köns­restriktion/körsätt matchas alltid exakt): "
    f"**{stats.get('description', report.similarity_description)}**"
)

if stats.get("n", 0) == 0:
    st.info(
        "Inget historiskt underlag hittades ännu. Kör `python -m atg_favorites.fetch` för att "
        "bygga upp arkivet i `data/raw`."
    )
elif stats["display"] == "insufficient":
    st.warning(f"Otillräckligt underlag (n={stats['n']}) - ingen procentsiffra visas.")
else:
    if stats["display"] == "uncertain":
        st.warning(f"⚠️ Osäkert underlag (n={stats['n']}, mellan 100 och 300 lopp) - tolka siffran med försiktighet.")
    if stats.get("is_baseline"):
        st.warning(
            "Detta är en **global baslinje** (bucketerad enbart på favoritens streckprocent) - "
            "inte lopptyps-matchade jämförelselopp, eftersom underlaget för de riktiga "
            "hård-/mjuk-villkoren var för litet."
        )

    cols = st.columns(3)
    cols[0].metric(
        "Favoritens vinstfrekvens",
        f"{stats['win_rate'] * 100:.1f}%",
        help=f"95% konfidensintervall: {stats['ci_low'] * 100:.1f}-{stats['ci_high'] * 100:.1f}%",
    )
    if stats.get("top3_rate") is not None:
        cols[1].metric("Favoritens topp-3-frekvens", f"{stats['top3_rate'] * 100:.1f}%")
    cols[2].metric("n (jämförelselopp)", stats["n"])
    st.caption(f"95% konfidensintervall för vinstfrekvensen: {stats['ci_low'] * 100:.1f}-{stats['ci_high'] * 100:.1f}%")

    with st.expander(f"Visa alla {len(report.similar_races)} jämförelselopp"):
        show_cols = [
            c
            for c in (
                "game_date",
                "track_name",
                "race_distance_m",
                "start_method",
                "sport",
                "is_coldblood",
                "is_mare_race",
                "favorite_horse_name",
                "favorite_streck_pct",
                "favorite_won",
                "winner_horse_name",
            )
            if c in report.similar_races.columns
        ]
        st.dataframe(
            report.similar_races[show_cols].sort_values("game_date", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

    if not report.favoritfall_examples.empty:
        st.markdown("#### Tidigare favoritfall bland jämförelselopp")
        st.dataframe(report.favoritfall_examples, use_container_width=True, hide_index=True)
