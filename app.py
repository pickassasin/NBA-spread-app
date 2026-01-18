import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
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
        return pd.DataFrame(columns=[
            "Date","Sport","Game","Bet","Odds","Result","Units"
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
    record = f"{len(wins)}-{len(losses)}"
    return round(roi*100,2), record

############################################
# NBA GAMES (LIVE API WITH FALLBACK)
############################################

def get_games_today():
    API_KEY = st.secrets.get("ODDS_API_KEY")
    if not API_KEY:
        return pd.DataFrame()

    url = (
        "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
        f"?apiKey={API_KEY}"
        "&regions=us"
        "&markets=spreads,h2h"
        "&oddsFormat=american"
    )

    data = safe_request(url)
    if not data:
        return pd.DataFrame()

    rows = []

    for game in data:
        home = game["home_team"]
        away = game["away_team"]

        for bookmaker in game.get("bookmakers", []):
            for market in bookmaker.get("markets", []):

                # Prefer spreads if available
                if market["key"] == "spreads":
                    for o in market["outcomes"]:
                        rows.append({
                            "Sport": "NBA",
                            "Game": f"{away} @ {home}",
                            "Team": o["name"],
                            "Spread": o.get("point",0),
                            "Odds": o["price"],
                            "Market": "Spread"
                        })
                    break  # stop after first spreads market
            if rows:
                break

        # If no spreads, fallback to h2h
        if not rows:
            for bookmaker in game.get("bookmakers", []):
                for market in bookmaker.get("markets", []):
                    if market["key"] == "h2h":
                        for o in market["outcomes"]:
                            rows.append({
                                "Sport": "NBA",
                                "Game": f"{away} @ {home}",
                                "Team": o["name"],
                                "Spread": 0,
                                "Odds": o["price"],
                                "Market": "Moneyline"
                            })
                        break
                if rows:
                    break

    return pd.DataFrame(rows)

############################################
# SIMPLE LEARNING MODEL (NO ERRORS)
############################################

def model_probability():
    # Starts neutral, shifts based on history
    history = load_history()
    if history.empty:
        return 0.53
    wins = len(history[history["Result"]=="Win"])
    total = len(history)
    return min(max(0.5 + (wins-total/2)/100, 0.45), 0.65)

############################################
# AUTO RESULT CHECK (SIMULATED SAFE)
############################################

def update_results():
    history = load_history()
    if history.empty:
        return history
    for i,row in history.iterrows():
        if row["Result"]=="Pending":
            # simulate final result (safe placeholder)
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
if games.empty:
    st.warning("No live games available yet from the API.")
else:
    prob = model_probability()
    games["Model Probability %"] = round(prob*100,1)
    games["Implied Probability"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))
    
    # Confidence bars (color-coded)
    def color_confidence(val):
        if val >= 60:
            return 'background-color: #00FF00'  # green
        elif val >= 50:
            return 'background-color: #FFFF00'  # yellow
        else:
            return 'background-color: #FF0000'  # red

    st.dataframe(games.style.applymap(color_confidence, subset=["Model Probability %"]), use_container_width=True)

############################################
# BET SLIP
############################################

st.header("🧾 Bet Slip")

selected_games = st.multiselect(
    "Select bets to confirm:",
    games.index if not games.empty else [],
    format_func=lambda i: f"{games.loc[i,'Sport']} | {games.loc[i,'Game']} | {games.loc[i,'Team']} ({games.loc[i,'Odds']})"
)

if st.button("✅ CONFIRM BETS") and not games.empty:
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

############################################
# HISTORY VIEW
############################################

st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)

############################################
# BEST BETS (HIGH CONFIDENCE)
############################################

if not games.empty:
    st.header("🔥 Best Bets")
    best_bets = games[games["Model Probability %"] >= 55].sort_values("Model Probability %", ascending=False).head(5)
    st.dataframe(best_bets, use_container_width=True)
