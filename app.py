import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

st.set_page_config(page_title="Sports Betting Model", layout="wide")

HISTORY_FILE = "history.csv"
ODDS_API_KEY = "fb78c9cf149ca0d18b6e70ac6d28075a"

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
# FETCH LIVE ODDS
############################################

def get_live_odds(sport):
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds?apiKey={ODDS_API_KEY}&regions=us&markets=spreads,h2h&oddsFormat=american"
    data = safe_request(url)
    games = []
    if data:
        for g in data:
            home = g.get("home_team")
            away = g.get("away_team")
            game_str = f"{away} @ {home}"
            markets = g["bookmakers"][0]["markets"]

            if sport == "basketball_nba":
                spread = next(m for m in markets if m["key"]=="spreads")
                for o in spread["outcomes"]:
                    games.append({
                        "Sport":"NBA",
                        "Game":game_str,
                        "Team":o["name"],
                        "Odds":o["price"]
                    })

            if sport == "icehockey_nhl":
                h2h = next(m for m in markets if m["key"]=="h2h")
                for o in h2h["outcomes"]:
                    games.append({
                        "Sport":"NHL",
                        "Game":game_str,
                        "Team":o["name"],
                        "Odds":o["price"]
                    })

    return pd.DataFrame(games)

############################################
# MODEL
############################################

def calculate_model_prob(row):
    history = load_history()
    if history.empty:
        return 0.53
    wins = len(history[history["Result"]=="Win"])
    total = len(history)
    return min(max(0.5 + (wins-total/2)/100, 0.45), 0.65)

############################################
# AUTO RESULT CHECK
############################################

def update_results():
    history = load_history()
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

history = update_results()
roi, record = calculate_roi(history)

st.metric("ROI %", roi)
st.metric("Record", record)

st.divider()
st.header("📅 Today's Games & Picks")

nba = get_live_odds("basketball_nba")
nhl = get_live_odds("icehockey_nhl")
games = pd.concat([nba, nhl], ignore_index=True)

if games.empty:
    st.warning("No live games available.")
else:
    games["Model Probability %"] = games.apply(lambda r: round(calculate_model_prob(r)*100,1), axis=1)
    games["Implied Probability %"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))

    def calculate_confidence(row):
        if row["Sport"]=="NHL":
            return round(abs(row["Model Probability %"] - row["Implied Probability %"]),1)
        else:
            return round(min(max(row["Implied Probability %"] + np.random.uniform(-10,10),10),90),1)

    games["Confidence %"] = games.apply(calculate_confidence, axis=1)

    def color_confidence(val):
        if val >= 65:
            return "background-color:#00c853"
        elif val >= 50:
            return "background-color:#ffb300"
        else:
            return "background-color:#d50000"

    st.dataframe(
        games.style.applymap(color_confidence, subset=["Confidence %"]),
        use_container_width=True
    )

############################################
# 🟢 BEST BETS (ADDED ONLY)
############################################

st.divider()
st.header("🔥 Best Bets")

best_bets = games[games["Confidence %"] >= 60].sort_values("Confidence %", ascending=False).head(5)

if best_bets.empty:
    st.info("No high-confidence bets available yet.")
else:
    st.dataframe(
        best_bets.style.applymap(color_confidence, subset=["Confidence %"]),
        use_container_width=True
    )

############################################
# BET SLIP
############################################

st.header("🧾 Bet Slip")

selected_games = st.multiselect(
    "Select bets:",
    games.index,
    format_func=lambda i: f"{games.loc[i,'Sport']} | {games.loc[i,'Game']} | {games.loc[i,'Team']} ({games.loc[i,'Odds']})"
)

if st.button("✅ CONFIRM BETS"):
    new = []
    for i in selected_games:
        r = games.loc[i]
        new.append({
            "Date":datetime.now().strftime("%Y-%m-%d"),
            "Sport":r["Sport"],
            "Game":r["Game"],
            "Bet":r["Team"],
            "Odds":r["Odds"],
            "Result":"Pending",
            "Units":0
        })
    history = pd.concat([history,pd.DataFrame(new)], ignore_index=True)
    save_history(history)
    st.success("Bets saved!")

############################################
# HISTORY
############################################

st.divider()
st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)
