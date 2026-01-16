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
# LIVE GAME DATA
############################################

def get_games_today():
    # NBA games
    nba_games = []
    date = datetime.now().strftime("%Y-%m-%d")
    nba_url = f"https://www.balldontlie.io/api/v1/games?start_date={date}&end_date={date}&per_page=100"
    nba_data = safe_request(nba_url)
    if nba_data and "data" in nba_data:
        for g in nba_data["data"]:
            home = g["home_team"]["full_name"]
            away = g["visitor_team"]["full_name"]
            nba_games.append({"Sport":"NBA","Game":f"{away} @ {home}","Team":home,"Odds":-110})
            nba_games.append({"Sport":"NBA","Game":f"{away} @ {home}","Team":away,"Odds":100})

    # NHL games
    nhl_games = []
    nhl_url = f"https://statsapi.web.nhl.com/api/v1/schedule?date={date}"
    nhl_data = safe_request(nhl_url)
    if nhl_data and nhl_data.get("dates"):
        for date_block in nhl_data["dates"]:
            for g in date_block["games"]:
                home = g["teams"]["home"]["team"]["name"]
                away = g["teams"]["away"]["team"]["name"]
                nhl_games.append({"Sport":"NHL","Game":f"{away} @ {home}","Team":home,"Odds":120})
                nhl_games.append({"Sport":"NHL","Game":f"{away} @ {home}","Team":away,"Odds":-130})

    games = pd.DataFrame(nba_games + nhl_games)
    if games.empty:
        st.warning("No games found for today.")
    return games

############################################
# SIMPLE LEARNING MODEL
############################################

def calculate_model_prob(row):
    history = load_history()
    if history.empty:
        return 0.53
    wins = len(history[history["Result"]=="Win"])
    total = len(history)
    return min(max(0.5 + (wins-total/2)/100, 0.45), 0.65)

############################################
# AUTO RESULT CHECK (SIMULATED)
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
    games["Implied Probability %"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))
    # Confidence bar (color-coded)
    games["Confidence %"] = games["Model Probability %"] - games["Implied Probability %"]
    games["Confidence %"] = games["Confidence %"].apply(lambda x: max(0,x))
    st.dataframe(games, use_container_width=True)
else:
    st.warning("No games available to display today.")

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

############################################
# HISTORY VIEW
############################################

st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)
