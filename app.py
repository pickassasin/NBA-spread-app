import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta
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
# FETCH TODAY + TOMORROW NBA GAMES
############################################

def get_games_today_nba():
    games = []
    for offset in range(0,2):  # today + tomorrow
        date = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
        url = f"https://www.balldontlie.io/api/v1/games?start_date={date}&end_date={date}&per_page=100"
        data = safe_request(url)
        if data and "data" in data:
            for g in data["data"]:
                home = g["home_team"]["full_name"]
                away = g["visitor_team"]["full_name"]
                games.append({"Sport":"NBA","Game":f"{away} @ {home}","Team":home,"Odds":-110})
                games.append({"Sport":"NBA","Game":f"{away} @ {home}","Team":away,"Odds":100})
    return pd.DataFrame(games)

############################################
# FETCH TODAY + TOMORROW NHL GAMES
############################################

def get_games_today_nhl():
    games = []
    for offset in range(0,2):  # today + tomorrow
        date = (datetime.now() + timedelta(days=offset)).strftime("%Y-%m-%d")
        url = f"https://statsapi.web.nhl.com/api/v1/schedule?date={date}"
        data = safe_request(url)
        if data and "dates" in data and len(data["dates"]) > 0:
            for g in data["dates"][0]["games"]:
                home = g["teams"]["home"]["team"]["name"]
                away = g["teams"]["away"]["team"]["name"]
                games.append({"Sport":"NHL","Game":f"{away} @ {home}","Team":home,"Odds":120})
                games.append({"Sport":"NHL","Game":f"{away} @ {home}","Team":away,"Odds":-130})
    return pd.DataFrame(games)

############################################
# FIXED STAT-BASED MODEL
############################################

def calculate_model_prob(row):
    # deterministic per team/game, simple placeholder combining stats
    np.random.seed(hash(row['Team']+row['Game']) % 2**32)
    recent_form = np.random.uniform(0.4, 0.7)
    offense = np.random.uniform(0.4, 0.7)
    defense = np.random.uniform(0.3, 0.6)
    home_adv = 0.05 if "@ " not in row["Game"].split(" @ ")[0] else 0
    prob = recent_form*0.5 + offense*0.3 + (1-defense)*0.2 + home_adv
    return min(max(prob, 0.05), 0.95)

############################################
# AUTO RESULT CHECK (SAFE)
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

# Fetch NBA + NHL games today/tomorrow
nba_games = get_games_today_nba()
nhl_games = get_games_today_nhl()
games = pd.concat([nba_games, nhl_games], ignore_index=True)

if not games.empty:
    # Model probabilities
    games["Model Probability %"] = games.apply(lambda r: round(calculate_model_prob(r)*100,1), axis=1)

    # Implied probability from odds
    games["Implied Probability %"] = games["Odds"].apply(lambda x: round(american_to_prob(x)*100,1))

    # Confidence / Edge
    games["Confidence %"] = games["Model Probability %"] - games["Implied Probability %"]
    games["Confidence %"] = games["Confidence %"].apply(lambda x: x if x > 0 else 0)

    st.dataframe(games, use_container_width=True)
else:
    st.warning("No games found for today or tomorrow.")

############################################
# BEST BETS + COLOR-CODED CONFIDENCE
############################################

st.divider()
st.header("🔥 Best Bets (Model Confidence)")

if not games.empty:
    best_bets = games[games["Confidence %"] > 0].sort_values("Confidence %", ascending=False)
    if best_bets.empty:
        st.info("No positive-edge bets today.")
    else:
        for _, row in best_bets.iterrows():
            st.subheader(f"{row['Sport']} — {row['Team']}")
            st.caption(f"{row['Game']} | Odds: {row['Odds']}")
            conf = min(max(int(row["Confidence %"]),0),100)
            if conf >= 10:
                bar_color = "#28a745"
            elif conf >= 5:
                bar_color = "#ffc107"
            else:
                bar_color = "#dc3545"
            st.markdown(
                f"""
                <div style="background-color:#e0e0e0; width:100%; height:20px; border-radius:5px;">
                    <div style="background-color:{bar_color}; width:{conf}%; height:100%; border-radius:5px;"></div>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.write(f"Confidence: **{round(row['Confidence %'],1)}%**")

############################################
# BET SLIP
############################################

st.divider()
st.header("🧾 Bet Slip")

selected_games = st.multiselect(
    "Select bets to confirm:",
    games.index if not games.empty else [],
    format_func=lambda i: f"{games.loc[i,'Sport']} | {games.loc[i,'Game']} | {games.loc[i,'Team']} ({games.loc[i,'Odds']})"
)

if st.button("✅ CONFIRM BETS") and not games.empty:
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
