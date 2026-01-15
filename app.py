import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="NBA & NHL Betting Edge", layout="wide")

API_KEY = st.secrets["ODDS_API_KEY"]

NBA_LOG = "nba_bets.csv"
NHL_LOG = "nhl_bets.csv"

HIGH_EDGE_THRESHOLD = 6  # %

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

def safe_request(url, params):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return []
        return r.json()
    except:
        return []

def pick_team(row):
    return row["Home Team"] if row["Edge %"] > 0 else row["Away Team"]

def init_log(path):
    if not os.path.exists(path):
        pd.DataFrame(columns=[
            "Date","Sport","Home Team","Away Team",
            "Bet On","Book Odds","Fair Odds",
            "Edge %","Result"
        ]).to_csv(path, index=False)

def log_bets(df, sport, path):
    init_log(path)
    existing = pd.read_csv(path)

    new = df.copy()
    new["Date"] = datetime.utcnow().date()
    new["Sport"] = sport
    new["Result"] = "PENDING"

    merged = pd.concat([existing, new], ignore_index=True)
    merged.drop_duplicates(
        subset=["Date","Sport","Home Team","Away Team","Bet On"],
        inplace=True
    )
    merged.to_csv(path, index=False)

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
        df["Model Prob"] = 0.52
        return df

    if not os.path.exists(NBA_LOG):
        df["Model Prob"] = 0.52
        return df

    hist = pd.read_csv(NBA_LOG)
    hist = hist[hist["Result"].isin(["WIN","LOSS"])]

    if hist.empty:
        df["Model Prob"] = 0.52
        return df

    try:
        X = hist[["Book Odds"]]
        y = (hist["Result"] == "WIN").astype(int)
        model = RandomForestClassifier()
        model.fit(X, y)
        df["Model Prob"] = model.predict_proba(df[["Book Odds"]])[:,1]
    except:
        df["Model Prob"] = 0.52

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
        df["Model Prob"] = 0.51
        return df

    if not os.path.exists(NHL_LOG):
        df["Model Prob"] = 0.51
        return df

    hist = pd.read_csv(NHL_LOG)
    hist = hist[hist["Result"].isin(["WIN","LOSS"])]

    if hist.empty:
        df["Model Prob"] = 0.51
        return df

    try:
        X = hist[["Book Odds"]]
        y = (hist["Result"] == "WIN").astype(int)
        model = RandomForestClassifier()
        model.fit(X, y)
        df["Model Prob"] = model.predict_proba(df[["Book Odds"]])[:,1]
    except:
        df["Model Prob"] = 0.51

    return df

# ======================
# RUN
# ======================
if screen == "NBA":
    df = fetch_nba_odds()
    df = nba_model(df)
    LOG_PATH = NBA_LOG
else:
    df = fetch_nhl_odds()
    df = nhl_model(df)
    LOG_PATH = NHL_LOG

if df.empty:
    st.info("No games available right now.")
    st.stop()

df["Book Prob"] = df["Book Odds"].apply(odds_to_prob)
df["Edge %"] = (df["Model Prob"] - df["Book Prob"]) * 100
df["Fair Odds"] = df["Model Prob"].apply(prob_to_odds)
df["Confidence"] = df["Edge %"].apply(edge_confidence)
df["Bet On"] = df.apply(pick_team, axis=1)
df["Abs Edge %"] = df["Edge %"].abs()

# ======================
# NOTIFICATIONS
# ======================
high_edge = df[df["Abs Edge %"] >= HIGH_EDGE_THRESHOLD]
if not high_edge.empty:
    st.toast("🔥 HIGH EDGE BETS AVAILABLE", icon="🔥")
    st.warning(f"{len(high_edge)} high-edge opportunities detected.")

# ======================
# LOG BETS
# ======================
log_bets(
    df[[
        "Home Team","Away Team","Bet On",
        "Book Odds","Fair Odds","Edge %"
    ]],
    screen,
    LOG_PATH
)

# ======================
# DISPLAY
# ======================
st.dataframe(
    df[[
        "Bet On",
        "Home Team",
        "Away Team",
        "Book Odds",
        "Fair Odds",
        "Abs Edge %",
        "Confidence"
    ]].sort_values("Abs Edge %", ascending=False),
    use_container_width=True
)
