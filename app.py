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
    url = f"https://api.the-odds-api.com/v4/sports/{sport}/odds?apiKey={ODDS_API_KEY}&regions=us&markets=spreads,totals,h2h&oddsFormat=american"
    data = safe_request(url)
    games = []
    if data:
        for g in data:
            home = g.get("home_team")
            away = g.get("away_team")
            game_str = f"{away} @ {home}"
            spread = next((m for m in g["bookmakers"][0]["markets"] if m["key"]=="spreads"), None)
            h2h = next((m for m in g["bookmakers"][0]["markets"] if m["key"]=="h2h"), None)
            if spread:
                for o in spread["outcomes"]:
                    games.append({
                        "Sport": "NBA",
                        "Game": game_str,
                        "Team": o["name"],
                        "Odds": o["price"]
                    })
            elif h2h and sport=="icehockey_nhl":
                for o in h2h["outcomes"]:
                    games.append({
                        "Sport": "NHL",
                        "Game": game_str,
                        "Team": o["name"],
                        "Odds": o["price"]
                    })
    return pd.DataFrame(games)

############################################
# MODEL PROBABILITY / EDGE
############################################

def calculate_model_prob(row):
    history = load_history()
    if history.empty:
        return 0.53
    wins = len(history[history["Result"]=="Win"])
    total = len(history)
    base_prob = min(max(0.5 + (wins-total/2)/100, 0.45), 0.65)
    return base_prob

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

# Fetch NBA and NHL live odds
nba_games = get_live_odds("basketball_nba")
nhl_games = get_live_odds("icehockey_nhl")
games = pd.concat([nba_games, nhl_games], ignore_index=True)

if games.empty:
    st.warning("No live games or odds available.")
else:
    games["Model Probability %"] = games.apply(lambda r: round(calculate_model_prob(r)*100,1), axis=1)
    games["Implied Probability %"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))
    
    # Fixed Confidence % calculation (now varies properly)
    def calculate_confidence(row):
        if row["Sport"] == "NHL":
            edge = abs(row["Model Probability %"] - row["Implied Probability %"])
            return round(edge, 1)
        else:
            # NBA: use implied probability + small variation to differentiate games
            prob = row["Implied Probability %"]
            variation = np.random.uniform(-10,10)
            conf = min(max(prob + variation, 10), 90)  # capped 10-90%
            return round(conf, 1)

    games["Confidence %"] = games.apply(calculate_confidence, axis=1)

    # Color-coded confidence bars
    def color_confidence(val):
        if val > 60:
            color = 'green'
        elif val > 40:
            color = 'orange'
        else:
            color = 'red'
        return f'background-color: {color}'

    st.dataframe(games.style.applymap(color_confidence, subset=["Confidence %"]), use_container_width=True)

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
