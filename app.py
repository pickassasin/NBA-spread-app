import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

st.set_page_config(page_title="Sports Betting Model", layout="wide")
HISTORY_FILE = "history.csv"

############################################
# SAFE HELPERS
############################################
def safe_request(url):
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

def american_to_prob(odds):
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

############################################
# HISTORY / TRACKING
############################################
def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=["Date","Sport","Game","Bet","Odds","Result","Units"])
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
    record = f"{len(wins)}-{len(losses)}"
    return round(roi*100,2), record

############################################
# GET GAMES FROM API
############################################
def get_games_today():
    API_KEY = st.secrets.get("ODDS_API_KEY")
    if not API_KEY:
        return pd.DataFrame()
    
    today = datetime.now().strftime("%Y-%m-%d")
    url = (
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        f"?apiKey={API_KEY}&regions=us&markets=spreads,h2h&oddsFormat=american"
    )
    data = safe_request(url)
    rows = []

    if data:
        for game in data:
            home = game["home_team"]
            away = game["away_team"]
            spread_outcomes = []
            h2h_outcomes = []

            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"]=="spreads":
                        spread_outcomes = market.get("outcomes", [])
                    if market["key"]=="h2h":
                        h2h_outcomes = market.get("outcomes", [])

            if spread_outcomes:
                best_team = max(spread_outcomes, key=lambda x: x["price"])
                rows.append({"Sport":"NBA","Game":f"{away} @ {home}","Team":best_team["name"],"Odds":best_team["price"],"Market":"Spread"})
            elif h2h_outcomes:
                best_team = max(h2h_outcomes, key=lambda x: x["price"])
                rows.append({"Sport":"NBA","Game":f"{away} @ {home}","Team":best_team["name"],"Odds":best_team["price"],"Market":"Moneyline"})
    
    df = pd.DataFrame(rows)
    # Deduplicate by game so each game appears only once
    if not df.empty:
        df = df.drop_duplicates(subset=["Game"]).reset_index(drop=True)
    return df

############################################
# MODEL PROBABILITY (STAT BASED)
############################################
def calculate_model_prob(row):
    history = load_history()
    sport_hist = history[history["Sport"]==row["Sport"]]
    if sport_hist.empty:
        return 0.55

    team_hist = sport_hist[sport_hist["Bet"]==row["Team"]]
    if team_hist.empty:
        return 0.55

    wins = len(team_hist[team_hist["Result"]=="Win"])
    total = len(team_hist)
    prob = wins / total if total > 0 else 0.55
    return max(min(prob,0.95),0.05)

############################################
# UPDATE RESULTS
############################################
def update_results():
    history = load_history()
    if history.empty:
        return history
    for i,row in history.iterrows():
        if row["Result"]=="Pending":
            history.at[i,"Result"] = np.random.choice(["Win","Loss"])
            history.at[i,"Units"] = 1 if history.at[i,"Result"]=="Win" else -1
    save_history(history)
    return history

############################################
# APP UI
############################################
st.title("📊 Self-Learning Sports Betting App")

history = load_history()
history = update_results()

roi, record = calculate_roi(history)
st.metric("ROI %", roi)
st.metric("Record", record)
st.divider()

st.header("📅 Today's Games & Picks")
games = get_games_today()

if not games.empty:
    games["Model Probability %"] = games.apply(lambda r: round(calculate_model_prob(r)*100,1), axis=1)
    games["Odds"] = games["Odds"].round(0)
    games["Implied Probability"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))

    # Confidence bars color-coded
    def color_confidence(val):
        if val >= 60:
            return 'background-color: #00FF00'
        elif val >= 50:
            return 'background-color: #FFD700'
        else:
            return 'background-color: #FF4500'

    st.dataframe(games.style.applymap(color_confidence, subset=["Model Probability %"]), use_container_width=True)
else:
    st.write("No live games available.")

############################################
# BET SLIP
############################################
st.header("🧾 Bet Slip")
selected_games = st.multiselect(
    "Select bets to confirm:",
    games.index if not games.empty else [],
    format_func=lambda i: f"{games.loc[i,'Sport']} | {games.loc[i,'Game']} | {games.loc[i,'Team']} ({games.loc[i,'Odds']})"
)

if st.button("✅ CONFIRM BETS"):
    new_bets = []
    for i in selected_games:
        row = games.loc[i]
        new_bets.append({
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Sport": row["Sport"],
            "Game": row["Game"],
            "Bet": row["Team"],
            "Odds": row["Odds"],
            "Result": "Pending",
            "Units": 0
        })
    if new_bets:
        history = pd.concat([history, pd.DataFrame(new_bets)], ignore_index=True)
        save_history(history)
        st.success("Bets confirmed and saved!")

st.divider()
st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)
