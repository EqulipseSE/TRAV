"""Streamlit-sida: analysera en avdelning (dagens omgång, avdelning för avdelning).

Väljer man datum, spelform, omgång och avdelning hämtas loppet live från
ATG (fungerar för kommande, spelbara, pågående och redan avgjorda lopp) och
jämförs mot liknande historiska lopp i det lokala arkivet
(``data/processed/races.csv``).
"""

from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from atg_favorites.api_client import AtgApiError, AtgClient  # noqa: E402
from atg_favorites.config import GAME_TYPES, RACES_CSV  # noqa: E402
from atg_favorites.leg_analysis import LegNotFoundError, build_leg_report  # noqa: E402

st.set_page_config(page_title="Analysera avdelning", page_icon="🔎", layout="wide")


@st.cache_resource
def get_client() -> AtgClient:
    return AtgClient(request_delay=0.2)


@st.cache_data(ttl=30, show_spinner="Hämtar dagens kalender från ATG...")
def load_calendar(date_str: str) -> dict:
    return get_client().get_calendar_day(date_str)


st.title("🔎 Analysera en avdelning")
st.caption(
    "Skicka in dagens (eller valfri annan) omgång, avdelning för avdelning, och få en "
    "grundlig analys: streckprocent, spår, tillägg, skor m.m. för alla startande, samt hur "
    "favoriten historiskt klarat sig i liknande lopp (samma bana/distans/startsätt/fältstorlek, "
    "eller annat relevant om underlaget är för litet)."
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
    races_csv_path = st.text_input("Sökväg till historisk races.csv", value=str(RACES_CSV))
    distance_tolerance = st.number_input("Distanstolerans (m)", min_value=0, value=150, step=25)
    field_size_tolerance = st.number_input("Fältstorlekstolerans", min_value=0, value=2, step=1)

analyze_clicked = st.sidebar.button("Analysera avdelning", type="primary", use_container_width=True)

if not analyze_clicked:
    st.info("Välj datum, spelform, omgång och avdelning i sidopanelen och klicka på **Analysera avdelning**.")
    st.stop()

try:
    with st.spinner(f"Hämtar och analyserar {game_type} avd {avd}..."):
        report = build_leg_report(
            Path(races_csv_path),
            date_str=date_str,
            avd=int(avd),
            game_type=game_type,
            client=get_client(),
            distance_tolerance=int(distance_tolerance),
            field_size_tolerance=int(field_size_tolerance),
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

st.markdown("### Liknande historiska lopp")
st.caption(f"Matchningskriterier som användes: {report.similarity_description}")

stats = report.similarity_stats
if stats.get("races"):
    cols = st.columns(4)
    cols[0].metric("Liknande lopp", stats["races"])
    cols[1].metric("Favoritens vinstfrekvens", f"{stats['favorite_win_rate'] * 100:.1f}%")
    cols[2].metric("Favoritens topp-3-frekvens", f"{stats['favorite_top3_rate'] * 100:.1f}%")
    cols[3].metric("Favoritfall-frekvens", f"{stats['favoritfall_rate'] * 100:.1f}%")

    if "close_streck_band_win_rate" in stats:
        st.info(
            f"Bland de {stats['close_streck_band_races']} liknande loppen där favoriten hade "
            f"ungefär samma streckprocent (±10 %-enheter) som i det aktuella loppet vann favoriten "
            f"{stats['close_streck_band_win_rate'] * 100:.1f}% - en edge på "
            f"{stats['close_streck_band_edge_vs_streck'] * 100:+.1f} %-enheter jämfört med vad "
            f"streckprocenten implicerar."
        )

    with st.expander(f"Visa alla {len(report.similar_races)} liknande lopp"):
        show_cols = [
            c
            for c in (
                "game_date",
                "track_name",
                "race_distance_m",
                "start_method",
                "num_starters",
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
        st.markdown("#### Tidigare favoritfall i liknande lopp")
        st.dataframe(report.favoritfall_examples, use_container_width=True, hide_index=True)
else:
    st.info(
        "Inget historiskt underlag hittades ännu. Kör `python -m atg_favorites.fetch` och "
        "`python -m atg_favorites.flatten` för att bygga upp arkivet - ju mer historik, desto "
        "bättre analys."
    )
