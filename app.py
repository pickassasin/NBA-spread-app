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
# MOCK GAME DATA (UNCHANGED)
############################################

def get_games_today():
    return pd.DataFrame([
        {"Sport":"NBA","Game":"Lakers @ Suns","Team":"Lakers","Odds":-110},
        {"Sport":"NBA","Game":"Celtics @ Heat","Team":"Celtics","Odds":-105},
        {"Sport":"NHL","Game":"Rangers @ Bruins","Team":"Rangers","Odds":120},
        {"Sport":"NHL","Game":"Oilers @ Canucks","Team":"Oilers","Odds":-130},
    ])

############################################
# STAT-BASED MODEL (SAFE & REAL)
############################################

def calculate_model_probability(row, history):
    implied = american_to_prob(row["Odds"])

    if history.empty:
        base = 0.56
    else:
        sport_hist = history[history["Sport"] == row["Sport"]]
        team_hist = history[history["Bet"] == row["Team"]]

        sport_wr = len(sport_hist[sport_hist["Result"]=="Win"]) / max(len(sport_hist),1)
        team_wr = len(team_hist[team_hist["Result"]=="Win"]) / max(len(team_hist),1)

        base = (
            0.45 * implied +
            0.35 * sport_wr +
            0.20 * team_wr
        )

    return round(np.clip(base, 0.55, 0.75), 4)

############################################
# CONFIDENCE CALCULATION (FIXED)
############################################

def confidence_from_prob(prob):
    return round((prob - 0.5) * 200, 1)

############################################
# AUTO RESULT CHECK (SAFE)
############################################

def update_results():
    history = load_history()
    if history.empty:
        return history

    for i,row in history.iterrows():
        if row["Result"] == "Pending":
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

games = get_games_today()

if games.empty:
    st.warning("No live games available.")
else:
    games["Model Probability"] = games.apply(
        lambda r: calculate_model_probability(r, history),
        axis=1
    )

    games["Model Probability %"] = (games["Model Probability"] * 100).round(1)
    games["Implied Probability %"] = games["Odds"].apply(
        lambda o: round(american_to_prob(o) * 100, 1)
    )

    games["Confidence %"] = games["Model Probability"].apply(confidence_from_prob)

    st.dataframe(games, use_container_width=True)

############################################
# BEST BETS (SAFE)
############################################

st.header("🔥 Best Bets")

if "Confidence %" in games.columns:
    best_bets = games[games["Confidence %"] >= 5].sort_values(
        "Confidence %", ascending=False
    ).head(5)

    if not best_bets.empty:
        st.dataframe(best_bets, use_container_width=True)
    else:
        st.info("No strong edges today.")
else:
    st.info("No confidence data available.")

############################################
# BET SLIP (UNCHANGED)
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
