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
# LIVE API GAMES (NBA + NHL)
############################################

def get_games_today():
    API_KEY = st.secrets.get("ODDS_API_KEY")
    if not API_KEY:
        return pd.DataFrame()

    sports_config = {
        "NBA": {"sport_key":"basketball_nba", "market":"spreads"},
        "NHL": {"sport_key":"icehockey_nhl", "market":"h2h"}
    }

    rows = []

    for sport, cfg in sports_config.items():
        url = (
            f"https://api.the-odds-api.com/v4/sports/{cfg['sport_key']}/odds"
            f"?apiKey={API_KEY}"
            f"&regions=us"
            f"&markets={cfg['market']}"
            f"&oddsFormat=american"
        )
        data = safe_request(url)
        if not data:
            continue

        for g in data:
            home = g["home_team"]
            away = g["away_team"]

            # Get the first available market (spread for NBA, h2h for NHL)
            market = None
            for bm in g.get("bookmakers", []):
                for m in bm.get("markets", []):
                    if m.get("key") == cfg["market"]:
                        market = m
                        break
                if market:
                    break

            if not market:
                continue

            # Pick the team with better odds as model favorite
            best_team = max(market.get("outcomes", []), key=lambda x: x["price"])
            rows.append({
                "Sport": sport,
                "Game": f"{away} @ {home}",
                "Team": best_team["name"],
                "Odds": round(best_team["price"],0),
                "Market": cfg["market"]
            })

    df = pd.DataFrame(rows)
    # Deduplicate so each game appears only once
    df = df.drop_duplicates(subset=["Game"]).reset_index(drop=True)
    return df

############################################
# SIMPLE LEARNING MODEL
############################################

def calculate_model_prob(row):
    history = load_history()
    if history.empty:
        return 0.55  # neutral start
    # calculate simple stat-based probability: prior win rate for this sport
    sport_hist = history[history["Sport"]==row["Sport"]]
    if sport_hist.empty:
        return 0.55
    wins = len(sport_hist[sport_hist["Result"]=="Win"])
    total = len(sport_hist)
    prob = 0.5 + (wins - (total - wins)) / (2*total)
    return max(min(prob,0.95),0.05)

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
    games["Implied Probability"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))

    # Confidence bars (green/yellow/red)
    def color_confidence(val):
        if val >= 60:
            return 'background-color: #00CC00'  # green
        elif val >= 50:
            return 'background-color: #FFD700'  # soft yellow
        else:
            return 'background-color: #FF3333'  # red

    st.dataframe(games.style.applymap(color_confidence, subset=["Model Probability %"]), use_container_width=True)
else:
    st.write("No live games available.")

############################################
# BET SLIP
############################################

st.header("🧾 Bet Slip")

if not games.empty:
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

st.divider()

############################################
# HISTORY VIEW
############################################

st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)

############################################
# BEST BETS
############################################

st.header("🔥 Best Bets")
if not games.empty:
    best_bets = games[games["Model Probability %"]>=60].sort_values("Model Probability %", ascending=False).head(5)
    st.dataframe(best_bets, use_container_width=True)
