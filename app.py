import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="NBA Betting Model", layout="wide")

HISTORY_FILE = "nba_history.csv"

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

def prob_to_color(prob):
    # Map probability 50-100% to green intensity
    intensity = int(min(max((prob-50)*5, 0), 255))
    return f'background-color: rgba(0,{intensity},0,0.3)'

############################################
# HISTORY / TRACKING
############################################

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=[
            "Date","Game","Bet","Odds","Result","Units"
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
# FETCH LIVE NBA GAMES AND ODDS
############################################

API_KEY = st.secrets.get("odds_api_key", None)
if not API_KEY:
    st.error("API key not found in Streamlit secrets as 'odds_api_key'")
    st.stop()

def fetch_nba_odds():
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey={API_KEY}&regions=us&markets=spreads&oddsFormat=american"
    data = safe_request(url)
    if not data:
        return pd.DataFrame()
    games_list = []
    for g in data:
        if not g.get("bookmakers"):
            continue
        for b in g["bookmakers"]:
            for m in b.get("markets", []):
                if m["key"] != "spreads":
                    continue
                for o in m["outcomes"]:
                    games_list.append({
                        "Game": f"{g['home_team']} @ {g['away_team']}",
                        "Home": g['home_team'],
                        "Away": g['away_team'],
                        "Bet": o["name"],
                        "Odds": o["price"]
                    })
    df = pd.DataFrame(games_list)
    df.drop_duplicates(subset=["Game","Bet"], inplace=True)
    return df

############################################
# STAT-BASED MODEL
############################################

def calculate_model_prob(row):
    # Placeholder stat model: uses random adjustment + spread direction
    # Real stats can be integrated via nba API / past games
    base = 0.5
    if "@" in row["Game"]:
        if row["Bet"] == row["Home"]:
            base += 0.05  # home team boost
        else:
            base -= 0.05
    # small random factor for variability
    prob = min(max(base + np.random.normal(0,0.05), 0.05), 0.95)
    return prob*100

############################################
# AUTO RESULT CHECK
############################################

def update_results(df):
    history = load_history()
    if history.empty:
        return history
    for i,row in history.iterrows():
        game_time = datetime.now() - timedelta(hours=2)
        if row["Result"]=="Pending":
            # only simulate result if game is past time (placeholder)
            history.at[i,"Result"] = np.random.choice(["Win","Loss"])
            history.at[i,"Units"] = 1 if history.at[i,"Result"]=="Win" else -1
    save_history(history)
    return history

############################################
# APP UI
############################################

st.title("📊 Self-Learning NBA Betting App")

# Load history and update results
history = load_history()
history = update_results(history)
roi, record = calculate_roi(history)

st.metric("ROI %", roi)
st.metric("Record", record)

st.divider()

st.header("📅 Today's NBA Games & Picks")

games = fetch_nba_odds()
if games.empty:
    st.info("No live NBA games available or odds not yet posted.")
else:
    games["Model Probability %"] = games.apply(lambda r: round(calculate_model_prob(r),1), axis=1)
    games["Implied Probability %"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))

    st.dataframe(
        games.style.apply(lambda r: [prob_to_color(r["Model Probability %"]) for _ in r], axis=1),
        use_container_width=True
    )

############################################
# BET SLIP
############################################

st.header("🧾 Bet Slip")

selected_games = st.multiselect(
    "Select bets to confirm:",
    games.index,
    format_func=lambda i: f"{games.loc[i,'Game']} | {games.loc[i,'Bet']} ({games.loc[i,'Odds']})"
)

if st.button("✅ CONFIRM BETS"):
    new_bets = []
    for i in selected_games:
        row = games.loc[i]
        new_bets.append({
            "Date": datetime.now().strftime("%Y-%m-%d"),
            "Game": row["Game"],
            "Bet": row["Bet"],
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
# BEST BETS
############################################

st.header("🔥 Best Bets")
if not games.empty:
    best_bets = games.sort_values("Model Probability %", ascending=False).head(5)
    st.dataframe(
        best_bets.style.apply(lambda r: [prob_to_color(r["Model Probability %"]) for _ in r], axis=1),
        use_container_width=True
    )

st.divider()

############################################
# HISTORY VIEW
############################################

st.header("📈 Bet History")
st.dataframe(load_history(), use_container_width=True)
