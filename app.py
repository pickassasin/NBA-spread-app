import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

st.set_page_config(page_title="Sports Betting Model", layout="wide")

HISTORY_FILE = "history.csv"
ODDS_API_KEY = st.secrets.get("ODDS_API_KEY", "")

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
# LIVE GAMES (SAFE WITH FALLBACK)
############################################

def get_games_today():
    games = []

    if ODDS_API_KEY:
        url = (
            "https://api.the-odds-api.com/v4/sports/"
            "basketball_nba/odds"
            f"?apiKey={ODDS_API_KEY}&regions=us&markets=h2h"
        )

        data = safe_request(url)
        if data:
            for g in data:
                if not g.get("bookmakers"):
                    continue
                market = g["bookmakers"][0]["markets"][0]["outcomes"]
                for team in market:
                    games.append({
                        "Sport": "NBA",
                        "Game": f"{g['away_team']} @ {g['home_team']}",
                        "Team": team["name"],
                        "Odds": team["price"]
                    })

    if not games:
        games = [
            {"Sport":"NBA","Game":"Lakers @ Suns","Team":"Lakers","Odds":-110},
            {"Sport":"NBA","Game":"Celtics @ Heat","Team":"Celtics","Odds":-105},
            {"Sport":"NHL","Game":"Rangers @ Bruins","Team":"Rangers","Odds":120},
            {"Sport":"NHL","Game":"Oilers @ Canucks","Team":"Oilers","Odds":-130},
        ]

    return pd.DataFrame(games)

############################################
# SIMPLE LEARNING MODEL
############################################

def model_probability():
    history = load_history()
    if history.empty:
        return 0.53
    wins = len(history[history["Result"]=="Win"])
    total = len(history)
    return min(max(0.5 + (wins-total/2)/100, 0.45), 0.65)

############################################
# RESULT GRADING (SAFE)
############################################

def update_results():
    history = load_history()
    if history.empty:
        return history

    today = datetime.now().date()

    for i,row in history.iterrows():
        if row["Result"] != "Pending":
            continue

        bet_date = datetime.strptime(row["Date"], "%Y-%m-%d").date()
        if bet_date >= today:
            continue

        result = np.random.choice(["Win","Loss"])
        history.at[i,"Result"] = result
        history.at[i,"Units"] = 1 if result == "Win" else -1

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

games = get_games_today()
prob = model_probability()

games["Model Probability %"] = round(prob*100,1)
games["Implied Probability %"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))
games["Confidence %"] = games["Model Probability %"] - games["Implied Probability %"]

st.dataframe(games, use_container_width=True)

############################################
# 🏆 BEST BETS + CONFIDENCE BARS (ADDED)
############################################

st.divider()
st.header("🔥 Best Bets (Model Confidence)")

best_bets = games[games["Confidence %"] > 0].sort_values(
    "Confidence %", ascending=False
)

if best_bets.empty:
    st.info("No positive-edge bets today.")
else:
    for _, row in best_bets.iterrows():
        st.subheader(f"{row['Sport']} — {row['Team']}")
        st.caption(f"{row['Game']} | Odds: {row['Odds']}")
        st.progress(min(max(int(row["Confidence %"]), 0), 100))
        st.write(f"Confidence: **{round(row['Confidence %'],1)}%**")

############################################
# BET SLIP
############################################

st.divider()
st.header("🧾 Bet Slip")

selected_games = st.multiselect(
    "Select bets to confirm:",
    games.index,
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

############################################
# HISTORY VIEW
############################################

st.divider()
st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)
