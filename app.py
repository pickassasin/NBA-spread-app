import streamlit as st
import requests
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="NBA & NHL Betting Edge", layout="wide")

API_KEY = st.secrets["ODDS_API_KEY"]

NBA_CSV = "nba_bet_results.csv"
NHL_CSV = "nhl_bet_results.csv"

# ======================
# HELPERS
# ======================
def odds_to_prob(odds):
    try:
        return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)
    except:
        return 0.5

def prob_to_odds(p):
    try:
        return int(-100 * p / (1 - p)) if p > 0.5 else int(100 * (1 - p) / p)
    except:
        return 0

def edge_confidence(edge):
    if edge >= 6:
        return "HIGH"
    if edge >= 3:
        return "MEDIUM"
    return "LOW"

def safe_request(url, params):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        return r.json()
    except:
        return []

# ======================
# SIDEBAR
# ======================
screen = st.sidebar.radio("Select Sport", ["NBA", "NHL"])
st.title(f"{screen} Betting Edge Finder")

# ======================
# NBA
# ======================
def fetch_nba_odds():
    url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "spreads",
        "oddsFormat": "american"
    }

    data = safe_request(url, params)
    games = []

    for g in data:
        try:
            bm = g["bookmakers"][0]
            market = bm["markets"][0]["outcomes"][0]
            games.append({
                "Home Team": g["home_team"],
                "Away Team": g["away_team"],
                "Book Odds": market["price"]
            })
        except:
            continue

    return pd.DataFrame(games)

def nba_model(df):
    if df.empty:
        return df.assign(Model_Prob=0.52)

    if not os.path.exists(NBA_CSV):
        return df.assign(Model_Prob=0.52)

    hist = pd.read_csv(NBA_CSV)
    if hist.empty or "Result" not in hist.columns:
        return df.assign(Model_Prob=0.52)

    try:
        X = hist[["Book Odds"]]
        y = hist["Result"]
        model = RandomForestClassifier()
        model.fit(X, y)
        df["Model_Prob"] = model.predict_proba(df[["Book Odds"]])[:, 1]
    except:
        df["Model_Prob"] = 0.52

    return df

# ======================
# NHL
# ======================
def fetch_nhl_odds():
    url = "https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "h2h",
        "oddsFormat": "american"
    }

    data = safe_request(url, params)
    games = []

    for g in data:
        try:
            bm = g["bookmakers"][0]
            home = g["home_team"]
            for o in bm["markets"][0]["outcomes"]:
                if o["name"] == home:
                    games.append({
                        "Home Team": home,
                        "Away Team": g["away_team"],
                        "Book Odds": o["price"]
                    })
        except:
            continue

    return pd.DataFrame(games)

def nhl_model(df):
    if df.empty:
        return df.assign(Model_Prob=0.51)

    if not os.path.exists(NHL_CSV):
        return df.assign(Model_Prob=0.51)

    hist = pd.read_csv(NHL_CSV)
    if hist.empty or "Result" not in hist.columns:
        return df.assign(Model_Prob=0.51)

    try:
        X = hist[["Book Odds"]]
        y = hist["Result"]
        model = RandomForestClassifier()
        model.fit(X, y)
        df["Model_Prob"] = model.predict_proba(df[["Book Odds"]])[:, 1]
    except:
        df["Model_Prob"] = 0.51

    return df

# ======================
# RUN
# ======================
if screen == "NBA":
    df = fetch_nba_odds()
    df = nba_model(df)
else:
    df = fetch_nhl_odds()
    df = nhl_model(df)

if df.empty:
    st.info("No games currently available.")
    st.stop()

df["Book Prob"] = df["Book Odds"].apply(odds_to_prob)
df["Edge %"] = (df["Model_Prob"] - df["Book Prob"]) * 100
df["Fair Odds"] = df["Model_Prob"].apply(prob_to_odds)
df["Confidence"] = df["Edge %"].apply(edge_confidence)

st.dataframe(
    df[[
        "Home Team",
        "Away Team",
        "Book Odds",
        "Fair Odds",
        "Edge %",
        "Confidence"
    ]].sort_values("Edge %", ascending=False),
    use_container_width=True
)
