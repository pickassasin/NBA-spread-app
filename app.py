import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
import os

st.set_page_config(page_title="NBA Betting Model", layout="wide")

# -----------------------------
# Constants
# -----------------------------
HISTORY_FILE = "history.csv"

# -----------------------------
# SAFE API KEY
# -----------------------------
try:
    API_KEY = st.secrets["ODDS_API_KEY"]
except KeyError:
    st.error("⚠️ API key not found! Please add ODDS_API_KEY to your Streamlit secrets.")
    st.stop()

# -----------------------------
# HELPER FUNCTIONS
# -----------------------------

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

# -----------------------------
# HISTORY / TRACKING
# -----------------------------

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=["Date","Game","Bet","Odds","Result","Units"])
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

# -----------------------------
# FETCH NBA GAMES & ODDS
# -----------------------------
def get_nba_games():
    url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/odds?apiKey={API_KEY}&regions=us&markets=spreads&oddsFormat=american"
    data = safe_request(url)
    if not data:
        return pd.DataFrame()
    
    games_list = []
    for g in data:
        for outcome in g.get("bookmakers", [{}])[0].get("markets", [{}])[0].get("outcomes", []):
            games_list.append({
                "Game": f"{g.get('home_team')} @ {g.get('away_team')}",
                "Team": outcome.get("name"),
                "Odds": outcome.get("price"),
                "Home Team": g.get("home_team"),
                "Away Team": g.get("away_team")
            })
    return pd.DataFrame(games_list).drop_duplicates(subset=["Game","Team"])

# -----------------------------
# STAT-BASED MODEL
# -----------------------------
def calculate_model_prob(row, history):
    # Very simple example stat-based model:
    # starts at 50%, adjusts based on past wins/losses for team
    base = 0.50
    team_games = history[history["Bet"]==row["Team"]]
    if not team_games.empty:
        wins = len(team_games[team_games["Result"]=="Win"])
        total = len(team_games)
        base += (wins - (total-wins)) * 0.02  # adjust by 2% per net win
    return min(max(base,0.05),0.95)

# -----------------------------
# AUTO UPDATE RESULTS (SIMULATED)
# -----------------------------
def update_results(history):
    if history.empty:
        return history
    for i,row in history.iterrows():
        if row["Result"]=="Pending":
            # placeholder simulation
            history.at[i,"Result"] = np.random.choice(["Win","Loss"])
            history.at[i,"Units"] = 1 if history.at[i,"Result"]=="Win" else -1
    save_history(history)
    return history

# -----------------------------
# APP UI
# -----------------------------
st.title("📊 NBA Self-Learning Betting Model")

# Load & update history
history = load_history()
history = update_results(history)
roi, record = calculate_roi(history)

st.metric("ROI %", roi)
st.metric("Record", record)
st.divider()

# Today's games
st.header("📅 Today's NBA Games")
games = get_nba_games()

if games.empty:
    st.warning("No NBA games or odds available for today.")
else:
    # calculate probabilities
    games["Model Probability"] = games.apply(lambda r: round(calculate_model_prob(r, history)*100,1), axis=1)
    games["Implied Probability"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))
    
    # Confidence color bar
    def color_bar(val):
        # green to light green
        return f'background: linear-gradient(90deg, #4caf50 {val}%, transparent 0%)'
    
    st.dataframe(games.style.applymap(lambda v: color_bar(v) if isinstance(v,(int,float)) else "", subset=["Model Probability"]), use_container_width=True)

    # Best bets
    st.header("🔥 Best Bets")
    best_bets = games.sort_values("Model Probability", ascending=False).head(5)
    st.dataframe(best_bets[["Game","Team","Odds","Model Probability"]], use_container_width=True)

# -----------------------------
# BET SLIP
# -----------------------------
st.header("🧾 Bet Slip")
if not games.empty:
    selected_games = st.multiselect(
        "Select bets to confirm:",
        games.index,
        format_func=lambda i: f"{games.loc[i,'Game']} | {games.loc[i,'Team']} ({games.loc[i,'Odds']})"
    )
    
    if st.button("✅ CONFIRM BETS"):
        new_bets = []
        for i in selected_games:
            row = games.loc[i]
            new_bets.append({
                "Date": datetime.now().strftime("%Y-%m-%d"),
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

# -----------------------------
# HISTORY VIEW
# -----------------------------
st.header("📈 Bet History")
st.dataframe(history, use_container_width=True)
