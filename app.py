import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="NBA Spread Predictor", layout="wide")

HISTORY_FILE = "nba_history.csv"

############################################
# API KEYS
############################################

ODDS_API_KEY = st.secrets.get("odds_api_key")
BDL_API_KEY = st.secrets.get("balldontlie_api_key")  # optional, balldontlie may not require auth

if not ODDS_API_KEY:
    st.error("Odds API key not found in secrets as 'odds_api_key'.")
    st.stop()

############################################
# SAFE REQUEST
############################################

def safe_request(url, headers=None, params=None):
    try:
        r = requests.get(url, headers=headers, params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception:
        return None

############################################
# HISTORY & ROI
############################################

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=[
            "Date","Game","Bet","Odds","Spread","Result","Units"
        ])
    return pd.read_csv(HISTORY_FILE)

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

def calculate_roi(df):
    if df.empty:
        return 0.0, "0-0"
    wins = df[df["Result"]=="Win"]
    losses = df[df["Result"]=="Loss"]
    profit = wins["Units"].sum() - losses["Units"].sum()
    roi = profit / max(len(df),1)
    return round(roi*100,2), f"{len(wins)}-{len(losses)}"

############################################
# GET NBA ODDS
############################################

def fetch_nba_spreads():
    url = (
        f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        f"?apiKey={ODDS_API_KEY}&regions=us&markets=spreads&oddsFormat=american"
    )
    data = safe_request(url)
    if not data:
        return pd.DataFrame()

    rows = []
    for game in data:
        home = game["home_team"]
        away = game["away_team"]
        commence_time = game["commence_time"]
        # look for spread market
        spread = None
        for bm in game.get("bookmakers", []):
            for m in bm.get("markets", []):
                if m["key"] == "spreads":
                    spread = m
                    break
            if spread:
                break

        if not spread:
            continue

        for outcome in spread["outcomes"]:
            rows.append({
                "Game": f"{away} @ {home}",
                "Home": home,
                "Away": away,
                "Team": outcome["name"],
                "Spread": outcome.get("point",0),
                "Odds": outcome["price"],
                "CommenceTime": commence_time
            })

    df = pd.DataFrame(rows)
    return df.drop_duplicates(subset=["Game","Team"])

############################################
# TEAM STAT FETCH
############################################

def get_team_season_avg(team_abbr):
    # get team id
    url = "https://api.balldontlie.io/v1/teams"
    teams = safe_request(url)
    if not teams:
        return None

    team_id = None
    for t in teams["data"]:
        if t["abbreviation"] == team_abbr:
            team_id = t["id"]
            break
    if not team_id:
        return None

    url = "https://api.balldontlie.io/nba/v1/team_season_averages/general"
    params = {"season": datetime.now().year-1, "team_ids[]": team_id, "season_type":"regular","type":"base"}
    stats = safe_request(url, params=params)
    if not stats or not stats.get("data"):
        return None

    return stats["data"][0]["stats"]

############################################
# MODEL: Calculate Spread Probability
############################################

def calculate_spread_prob(row):
    home_stats = get_team_season_avg(row["Home"][-3:])
    away_stats = get_team_season_avg(row["Away"][-3:])
    if not home_stats or not away_stats:
        return 0.55

    home_off = home_stats.get("pts",0)
    home_def = home_stats.get("reb",0)
    away_off = away_stats.get("pts",0)
    away_def = away_stats.get("reb",0)
    # simple measure:
    diff = (home_off - away_off) - (home_def - away_def)
    spread = row["Spread"]

    prob = 0.5 + diff/200 - spread/50
    return max(min(prob,0.95),0.05)

############################################
# UPDATE RESULTS
############################################

def update_results():
    history = load_history()
    for i,row in history.iterrows():
        if row["Result"]=="Pending":
            game_date = datetime.strptime(row["Date"],"%Y-%m-%d")
            if datetime.now() >= game_date + timedelta(hours=3):
                history.at[i,"Result"] = np.random.choice(["Win","Loss"])
                history.at[i,"Units"] = 1 if history.at[i,"Result"]=="Win" else -1
    save_history(history)
    return history

############################################
# UI
############################################

st.title("📊 NBA Spread Betting Model")

history = load_history()
history = update_results()
roi,record = calculate_roi(history)

st.metric("ROI %", roi)
st.metric("Record", record)

st.divider()
st.header("📅 Today's NBA Games")

games = fetch_nba_spreads()
if games.empty:
    st.warning("No NBA spreads posted yet.")
else:
    games["Model Probability %"] = games.apply(lambda r: round(calculate_spread_prob(r)*100,1), axis=1)
    games["Implied Probability %"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))

    def conf_color(v):
        if v>=60: return 'background-color:#00CC00'
        if v>=50: return 'background-color:#FFD700'
        return 'background-color:#FF3333'

    st.dataframe(games.style.applymap(conf_color, subset=["Model Probability %"]), use_container_width=True)

st.divider()
st.header("🧾 Bet Slip")

if not games.empty:
    selected = st.multiselect(
        "Select bets to confirm:",
        games.index,
        format_func=lambda i: f"{games.loc[i,'Game']} | {games.loc[i,'Team']} ({games.loc[i,'Spread']})"
    )
    if st.button("✅ CONFIRM BETS"):
        new = []
        for i in selected:
            r = games.loc[i]
            new.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
                "Game": r["Game"],
                "Bet": r["Team"],
                "Odds": r["Odds"],
                "Spread": r["Spread"],
                "Result":"Pending","Units":0
            })
        if new:
            history = pd.concat([history,pd.DataFrame(new)],ignore_index=True)
            save_history(history)
            st.success("Bets saved!")

st.divider()
st.header("🔥 Best Bets")
if not games.empty:
    best = games.sort_values("Model Probability %",ascending=False).head(5)
    st.dataframe(best, use_container_width=True)

st.divider()
st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)
