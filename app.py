import streamlit as st
import requests
import pandas as pd
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIG
# ======================
API_KEY = st.secrets["ODDS_API_KEY"]

NBA_CSV = "nba_bet_results.csv"
NHL_CSV = "nhl_bet_results.csv"

st.set_page_config(page_title="NBA & NHL Betting Edge", layout="wide")

# ======================
# HELPERS
# ======================
def odds_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def prob_to_odds(p):
    return int(-100 * p / (1 - p)) if p > 0.5 else int(100 * (1 - p) / p)

def edge_confidence(edge):
    if edge >= 6:
        return "HIGH"
    if edge >= 3:
        return "MEDIUM"
    return "LOW"

# ======================
# SIDEBAR
# ======================
screen = st.sidebar.radio("Select Sport", ["NBA", "NHL"])

# ======================
# NBA LOGIC
# ======================
def fetch_nba_odds():
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
    params = {
        "apiKey": API_KEY,
        "markets": "spreads",
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    games = []
    for g in data:
        bm = g["bookmakers"][0]
        market = bm["markets"][0]["outcomes"][0]
        games.append({
            "Home Team": g["home_team"],
            "Away Team": g["away_team"],
            "Book Odds": market["price"]
        })
    return pd.DataFrame(games)

def nba_model(df):
    if not os.path.exists(NBA_CSV):
        df["Model Prob"] = 0.52
        return df

    hist = pd.read_csv(NBA_CSV)
    if hist.empty:
        df["Model Prob"] = 0.52
        return df

    X = hist[["Book Odds"]]
    y = hist["Result"]
    model = RandomForestClassifier()
    model.fit(X, y)

    df["Model Prob"] = model.predict_proba(df[["Book Odds"]])[:, 1]
    return df

# ======================
# NHL LOGIC
# ======================
def fetch_nhl_odds():
    url = f"https://api.the-odds-api.com/v4/sports/icehockey_nhl/odds"
    params = {
        "apiKey": API_KEY,
        "markets": "h2h",
        "oddsFormat": "american"
    }
    r = requests.get(url, params=params)
    r.raise_for_status()
    data = r.json()

    games = []
    for g in data:
        bm = g["bookmakers"][0]
        home = g["home_team"]
        for o in bm["markets"][0]["outcomes"]:
            if o["name"] == home:
                games.append({
                    "Home Team": home,
                    "Away Team": g["away_team"],
                    "Book Odds": o["price"]
                })
    return pd.DataFrame(games)

def nhl_model(df):
    if not os.path.exists(NHL_CSV):
        df["Model Prob"] = 0.51
        return df

    hist = pd.read_csv(NHL_CSV)
    if hist.empty:
        df["Model Prob"] = 0.51
        return df

    X = hist[["Book Odds"]]
    y = hist["Result"]
    model = RandomForestClassifier()
    model.fit(X, y)

    df["Model Prob"] = model.predict_proba(df[["Book Odds"]])[:, 1]
    return df

# ======================
# UI
# ======================
st.title(f"{screen} Betting Edge Finder")

if screen == "NBA":
    df = fetch_nba_odds()
    df = nba_model(df)

elif screen == "NHL":
    df = fetch_nhl_odds()
    df = nhl_model(df)

df["Book Prob"] = df["Book Odds"].apply(odds_to_prob)
df["Edge %"] = (df["Model Prob"] - df["Book Prob"]) * 100
df["Fair Odds"] = df["Model Prob"].apply(prob_to_odds)
df["Confidence"] = df["Edge %"].apply(edge_confidence)

st.dataframe(
    df[[
        "Home Team", "Away Team",
        "Book Odds", "Fair Odds",
        "Edge %", "Confidence"
    ]].sort_values("Edge %", ascending=False),
    use_container_width=True
)
