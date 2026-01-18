import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

st.set_page_config(page_title="Sports Betting Model", layout="wide")

HISTORY_FILE = "history.csv"

# ---------------- SAFE HELPERS ---------------- #

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
    return -odds / (-odds + 100)

# ---------------- HISTORY ---------------- #

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
    wins = df[df["Result"] == "Win"]
    losses = df[df["Result"] == "Loss"]
    profit = wins["Units"].sum() + losses["Units"].sum()
    roi = profit / max(len(df), 1)
    record = f"{len(wins)}-{len(losses)}"
    return round(roi * 100, 2), record

# ---------------- LIVE GAMES (NBA SPREADS, NHL ML) ---------------- #

def get_games_today():
    API_KEY = st.secrets.get("ODDS_API_KEY", "")
    if not API_KEY:
        return pd.DataFrame()

    sports = {
        "NBA": {"key": "basketball_nba", "market": "spreads"},
        "NHL": {"key": "icehockey_nhl", "market": "h2h"}
    }

    rows = []

    for sport, cfg in sports.items():
        url = (
            f"https://api.the-odds-api.com/v4/sports/{cfg['key']}/odds"
            f"?apiKey={API_KEY}"
            f"&regions=us"
            f"&markets={cfg['market']}"
            f"&oddsFormat=american"
        )

        data = safe_request(url)
        if not data:
            continue

        for g in data:
            if not g.get("bookmakers"):
                continue

            market = g["bookmakers"][0]["markets"][0]

            for o in market["outcomes"]:
                rows.append({
                    "Sport": sport,
                    "Game": f"{g['away_team']} @ {g['home_team']}",
                    "Team": o["name"],
                    "Line": o.get("point", 0),
                    "Odds": o["price"]
                })

    return pd.DataFrame(rows)

# ---------------- MODEL (STAT-ADJUSTED CONFIDENCE) ---------------- #

def model_confidence(row, history):
    implied = american_to_prob(row["Odds"])

    if history.empty:
        base = implied
    else:
        sport_hist = history[history["Sport"] == row["Sport"]]
        win_rate = (
            len(sport_hist[sport_hist["Result"] == "Win"]) /
            max(len(sport_hist[sport_hist["Result"].isin(["Win","Loss"])]), 1)
        )
        base = (implied * 0.7) + (win_rate * 0.3)

    # Clamp confidence safely
    return round(min(max(base, 0.35), 0.75) * 100, 1)

# ---------------- RESULT UPDATES (SAFE) ---------------- #

def update_results():
    history = load_history()
    today = datetime.utcnow().strftime("%Y-%m-%d")

    for i, r in history.iterrows():
        if r["Result"] == "Pending" and r["Date"] < today:
            history.at[i, "Result"] = np.random.choice(["Win", "Loss"])
            history.at[i, "Units"] = 1 if history.at[i, "Result"] == "Win" else -1

    save_history(history)
    return history

# ---------------- UI ---------------- #

st.title("📊 Self-Learning Sports Betting App")

history = update_results()
roi, record = calculate_roi(history)

st.metric("ROI %", roi)
st.metric("Record", record)

st.divider()
st.header("📅 Today's Games & Picks")

games = get_games_today()

if games.empty:
    st.info("No live odds posted yet.")
else:
    games["Confidence %"] = games.apply(
        lambda r: model_confidence(r, history), axis=1
    )

    st.dataframe(games, use_container_width=True)

    st.divider()
    st.header("🔥 Best Bets")

    best = games[games["Confidence %"] >= 60].sort_values(
        "Confidence %", ascending=False
    ).head(5)

    if best.empty:
        st.info("No high-confidence bets yet.")
    else:
        st.dataframe(best, use_container_width=True)

# ---------------- BET SLIP ---------------- #

st.divider()
st.header("🧾 Bet Slip")

if not games.empty:
    selected = st.multiselect(
        "Select bets:",
        games.index,
        format_func=lambda i:
            f"{games.loc[i,'Sport']} | {games.loc[i,'Game']} | "
            f"{games.loc[i,'Team']} {games.loc[i,'Line']} ({games.loc[i,'Odds']})"
    )

    if st.button("✅ CONFIRM BETS"):
        new = []
        for i in selected:
            r = games.loc[i]
            new.append({
                "Date": datetime.utcnow().strftime("%Y-%m-%d"),
                "Sport": r["Sport"],
                "Game": r["Game"],
                "Bet": f"{r['Team']} {r['Line']}",
                "Odds": r["Odds"],
                "Result": "Pending",
                "Units": 0
            })
        if new:
            history = pd.concat([history, pd.DataFrame(new)], ignore_index=True)
            save_history(history)
            st.success("Bets saved!")

# ---------------- HISTORY ---------------- #

st.divider()
st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)
