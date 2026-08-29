# V75/V85/V86 Favoritanalys

Ett litet Python-projekt som

1. **hämtar avgjorda V75/V85/V86-omgångar** från ATG:s (odokumenterade men
   publika) `racinginfo`-JSON-API och sparar rå-JSON lokalt,
2. **plattar ut** den rå-JSON:en till en CSV med **en rad per lopp**
   (favoritens streckprocent, spår, tillägg, skor, sulky, odds, resultat
   m.m.), och
3. gör en **streckjusterad favoritfall-analys**: hur ofta vinner favoriten
   (den häst som fått mest streck/pengar i V-poolen) jämfört med vad
   streckprocenten faktiskt implicerar - grupperat i streck-bucketar.

Det finns även en **Streamlit-sida** för att filtrera omgångarna och
streck-grupperna interaktivt.

> **Ansvarsfriskrivning**: API:et (`https://www.atg.se/services/racinginfo/v1/api/...`)
> är samma som atg.se:s egen webbplats använder, men det är inte officiellt
> dokumenterat eller supportat av ATG. Använd det ansvarsfullt (projektet
> har inbyggd fördröjning mellan anrop) och var beredd på att fältnamn kan
> ändras. Detta är ett analys-/forskningsprojekt, inte ett spelverktyg med
> garanterad träffsäkerhet.

## Hur det fungerar

### 1. Datakälla

Två endpoints i ATG:s API används:

| Endpoint | Vad den ger |
|---|---|
| `GET /calendar/day/{YYYY-MM-DD}` | Dagens banor och, per spelform (`V75`/`V85`/`V86`/...), listan av omgångar den dagen med status (`upcoming`, `bettable`, `ongoing`, `results`). |
| `GET /games/{game_id}` | Fullständig information om en omgång: pooler, samtliga lopp och för varje lopp samtliga startande (häst, kusk, skor, sulky, streckprocent/`betDistribution`, odds) samt - när loppet är avgjort - resultat. |

`fetch.py` scannar kalendern dag för dag, plockar ut varje V75/V85/V86-omgång
med status `results` ("avgjord") och sparar hela `games/{game_id}`-svaret
oförändrat som `data/raw/{game_id}.json`.

### 2. Flatten till CSV

`flatten.py` läser alla `data/raw/*.json`, går igenom varje avgjort lopp och
bygger **en rad per lopp** med bland annat:

* `favorite_streck_pct` - favoritens andel av V-poolen (streckprocent)
* `favorite_post_position` - spår
* `favorite_added_distance_m` - tillägg (extra startsträcka, t.ex. för
  autostartsgrupper)
* `favorite_shoes_front` / `favorite_shoes_back` / `favorite_shoes_changed` - skor
* `favorite_sulky_type`, `favorite_driver`, `favorite_trainer`
* `favorite_final_odds`, `favorite_place`, `favorite_won`, `favorite_top3`
* `second_favorite_streck_pct`, `favorite_streck_margin` - hur dominant favoriten var
* `winner_horse_name`, `winner_is_favorite`

Favoriten definieras som den icke-strukna häst som har högst streckprocent
i loppets V-pool.

### 3. Streckjusterad favoritfall-analys

"Favoritfall" = loppet där favoriten **inte** vinner. `analysis.py` grupperar
lopp i streck-%-bucketar (t.ex. 0-15%, 15-20%, ..., 50-100%) och beräknar per
bucket:

* `win_rate` / `top3_rate` - favoritens faktiska vinst- och pallfrekvens
* `implied_win_prob` - streckprocenten som andel (crowdens implicerade sannolikhet)
* `favoritfall_rate` = `1 - win_rate`
* `edge_vs_streck` = `win_rate - implied_win_prob` - positivt betyder att
  favoriten i den gruppen vinner *oftare* än streckprocenten antyder,
  negativt att den är överspelad.

Resultatet skrivs till `data/processed/favorite_bucket_analysis.csv`, och
kraftiga favoritfall (stark favorit som ändå förlorar) skrivs till
`data/processed/favorite_surprises.csv`.

### 4. Streamlit-sida

`streamlit_app/app.py` läser `data/processed/races.csv` och låter dig
filtrera omgångarna (spelform, bana, datum, antal startande, favoritens
streckprocent, skor) samt justera streck-gruppernas bredd och se
favoritfall-analysen och en stapeldiagram-jämförelse live på det filtrerade
urvalet.

## Installation

Kräver Python 3.10+.

```bash
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Körinstruktioner

Allt körs från repo-roten så att `atg_favorites`-paketet hittas.

### Hämta rå-data

```bash
# De senaste 30 dagarnas avgjorda V75/V85/V86-omgångar
python -m atg_favorites.fetch --days-back 30

# Ett specifikt datumintervall
python -m atg_favorites.fetch --start-date 2026-01-01 --end-date 2026-03-31

# Bara en spelform
python -m atg_favorites.fetch --days-back 14 --game-types V75

# Ladda om redan hämtade omgångar
python -m atg_favorites.fetch --days-back 30 --overwrite
```

Rå-JSON hamnar i `data/raw/<game_id>.json`, t.ex. `data/raw/V85_2026-08-22_23_5.json`.
Redan hämtade omgångar hoppas över nästa gång, så kommandot är säkert att
köra regelbundet (t.ex. i ett cron-jobb) för att gradvis bygga upp ett
historiskt dataset.

### Platta ut till CSV

```bash
python -m atg_favorites.flatten
# eller med egna sökvägar
python -m atg_favorites.flatten --raw-dir data/raw --out data/processed/races.csv
```

### Kör favoritfall-analysen

```bash
python -m atg_favorites.analysis
python -m atg_favorites.analysis --by-game-type          # dela upp per V75/V85/V86
python -m atg_favorites.analysis --min-surprise-streck 40 # tröskel för "stark favorit som föll"
```

### Allt i ett kommando

Ett samlat CLI (`atg_favorites.cli`) kör hela pipelinen:

```bash
python -m atg_favorites.cli pipeline --days-back 30 --by-game-type
```

eller styckvis: `python -m atg_favorites.cli fetch|flatten|analyze [...]`.

### Streamlit-sidan

```bash
streamlit run streamlit_app/app.py
```

Öppna länken som skrivs ut (normalt `http://localhost:8501`). Sidan läser
`data/processed/races.csv` som standard - kör fetch + flatten först (se
ovan) om filen inte finns än. Sökvägen till CSV:n kan även ändras direkt i
sidopanelen.

## Projektstruktur

```
atg_favorites/
  api_client.py   HTTP-klient mot ATG:s racinginfo-API (med retry/backoff)
  config.py       Sökvägar, endpoints, konstanter (spelformer, bucket-gränser m.m.)
  fetch.py        Hämtar avgjorda omgångar -> data/raw/*.json
  flatten.py      Rå-JSON -> en rad per lopp -> data/processed/races.csv
  analysis.py     Streckjusterad favoritfall-analys, bucketgruppering
  cli.py          Samlat CLI: fetch / flatten / analyze / pipeline
streamlit_app/
  app.py          Streamlit-sida för att filtrera omgångar och streckgrupper
data/
  raw/            Rå-JSON per omgång (skapas av fetch.py, ej incheckad)
  processed/      Genererade CSV:er (skapas av flatten.py/analysis.py, ej incheckade)
tests/
  test_flatten.py, test_analysis.py, fixtures/sample_game.json
```

## Tester

```bash
pytest
```

Testerna körs helt lokalt mot en handgjord fixture (`tests/fixtures/sample_game.json`)
som är formad efter det verkliga API-svaret och kräver ingen nätverksåtkomst.
