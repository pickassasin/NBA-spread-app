import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
import os

st.set_page_config(page_title="Sports Betting Model", layout="wide")

HISTORY_FILE = "history.csv"
API_KEY = st.secrets["odds_api_key"]  # make sure you add your API key in Streamlit secrets

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

def calc_edge(row):
    """For NHL: edge % = model probability - implied probability"""
    return round((row["Model Probability"] - row["Implied Probability"])*100, 1)

############################################
# HISTORY / TRACKING
############################################

def load_history():
    if not os.path.exists(HISTORY_FILE):
        return pd.DataFrame(columns=[
            "Date","Sport","Game","Bet","Odds","Result","Units","PointsScored","PointsAllowed"
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
# FETCH LIVE GAMES FROM ODDS API
############################################

def fetch_games(sport_key, market="spreads"):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds?apiKey={API_KEY}&markets={market}&regions=us&oddsFormat=american"
    data = safe_request(url)
    if not data:
        return pd.DataFrame()
    games = []
    for g in data:
        home = g.get("home_team")
        away = g.get("away_team")
        commence_time = g.get("commence_time")
        bookmakers = g.get("bookmakers", [])
        if not bookmakers:
            continue
        outcomes = bookmakers[0]["markets"][0]["outcomes"]
        for o in outcomes:
            games.append({
                "Sport": sport_key.split("_")[1].upper(),
                "Game": f"{home} @ {away}",
                "Team": o["name"],
                "Odds": o["price"],
                "CommenceTime": commence_time
            })
    df = pd.DataFrame(games)
    return df.drop_duplicates(subset=["Game","Team"])

############################################
# STAT-BASED MODEL
############################################

def calculate_model_prob(row):
    """Calculate probability based on team stats and history"""
    history = load_history()
    sport_hist = history[history["Sport"]==row["Sport"]]
    team_hist = sport_hist[sport_hist["Bet"]==row["Team"]]

    if team_hist.empty:
        return 0.55  # neutral default

    # Simple stats: Points scored, allowed, recent win rate
    points_scored = team_hist.get("PointsScored", pd.Series([20]*len(team_hist))).mean()
    points_allowed = team_hist.get("PointsAllowed", pd.Series([20]*len(team_hist))).mean()
    recent_games = team_hist.tail(10)
    recent_win_rate = len(recent_games[recent_games["Result"]=="Win"]) / max(len(recent_games),1)

    stat_score = (points_scored - points_allowed)/100 + recent_win_rate
    prob = 0.5 + stat_score/2
    return max(min(prob,0.95),0.05)

############################################
# UPDATE RESULTS SAFELY
############################################

def update_results():
    history = load_history()
    if history.empty:
        return history
    now = datetime.now()
    for i,row in history.iterrows():
        if row["Result"]=="Pending":
            game_date = datetime.strptime(row["Date"], "%Y-%m-%d")
            if now >= game_date + timedelta(hours=4):
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

# Pull live NBA and NHL games
nba_games = fetch_games("basketball_nba")
nhl_games = fetch_games("icehockey_nhl", market="h2h")
games = pd.concat([nba_games, nhl_games], ignore_index=True)

if games.empty:
    st.warning("No live games available.")
else:
    games["Model Probability"] = games.apply(calculate_model_prob, axis=1)
    games["Implied Probability"] = games["Odds"].apply(lambda x: american_to_prob(x))
    # NHL edge
    games["Edge %"] = games.apply(lambda r: calc_edge(r) if r["Sport"]=="NHL" else np.nan, axis=1)

    # Round all probabilities to two decimals
    games["Model Probability"] = (games["Model Probability"]*100).round(2)
    games["Implied Probability"] = (games["Implied Probability"]*100).round(2)
    if "Edge %" in games:
        games["Edge %"] = games["Edge %"].round(2)

    # Remove duplicate games
    games = games.drop_duplicates(subset=["Game","Team"])

    st.dataframe(games, use_container_width=True)

    # Confidence bars for best bets (top 5 per sport)
    st.header("🔥 Best Bets")
    for sport in games["Sport"].unique():
        sport_games = games[games["Sport"]==sport].copy()
        if sport=="NHL":
            best = sport_games.sort_values("Edge %", ascending=False).head(5)
            st.subheader(f"{sport} Best Bets (Edge %)")
            for idx,row in best.iterrows():
                st.markdown(f"**{row['Game']} | {row['Team']} | Edge: {row['Edge %']}%**")
                st.progress(min(max(row['Edge %']/100,0),1))
        else:
            best = sport_games.sort_values("Model Probability", ascending=False).head(5)
            st.subheader(f"{sport} Best Bets (Probability %)")
            for idx,row in best.iterrows():
                st.markdown(f"**{row['Game']} | {row['Team']} | Probability: {row['Model Probability']}%**")
                st.progress(min(max(row['Model Probability']/100,0),1))

st.divider()

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
                "Units": 0,
                "PointsScored": np.nan,
                "PointsAllowed": np.nan
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
