import streamlit as st
import pandas as pd
import numpy as np
import requests
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Sports Betting App", layout="wide")

API_KEY = st.secrets["ODDS_API_KEY"]
HISTORY_FILE = "history.csv"

SPORTS = {
    "NBA": {
        "odds_key": "basketball_nba",
        "market": "spreads",
        "mode": "prob"
    },
    "NHL": {
        "odds_key": "icehockey_nhl",
        "market": "h2h",
        "mode": "edge"
    }
}

# ---------------- UTILITIES ----------------
def american_to_prob(odds):
    odds = float(odds)
    if odds > 0:
        return 100 / (odds + 100)
    else:
        return -odds / (-odds + 100)

def spread_strength(spread):
    return min(abs(float(spread)) / 10, 1)

def load_history():
    try:
        return pd.read_csv(HISTORY_FILE)
    except:
        return pd.DataFrame(columns=[
            "Date","Sport","Game","Pick","Odds","Result","Units"
        ])

def save_history(df):
    df.to_csv(HISTORY_FILE, index=False)

# ---------------- TEAM STATS ----------------
def calculate_team_elo(history, team):
    """Simple Elo: start 1500, +20 for win, -20 for loss"""
    h = history[(history["Pick"]==team) | (history["Game"].str.contains(team))]
    elo = 1500
    for _, row in h.iterrows():
        if row["Result"]=="Win" and row["Pick"]==team:
            elo += 20
        elif row["Result"]=="Loss" and row["Pick"]==team:
            elo -= 20
    return elo

def get_recent_form(history, team, sport):
    h = history[(history["Sport"]==sport) & ((history["Pick"]==team) | (history["Game"].str.contains(team)))]
    last5 = h.tail(5)
    if len(last5)==0:
        return 0.5
    return (last5["Result"]=="Win").mean()

def get_rest_days(history, team):
    h = history[(history["Pick"]==team) | (history["Game"].str.contains(team))]
    if len(h)==0:
        return 2
    last_date = pd.to_datetime(h["Date"].iloc[-1])
    return (datetime.now() - last_date).days

# ---------------- MODEL ----------------
def train_model(history, sport):
    data = history[history["Sport"] == sport]
    if len(data) < 25:
        return None

    X = data[["Odds"]]  # keep simple; can expand later with Elo/form
    y = (data["Result"]=="Win").astype(int)

    model = LogisticRegression()
    model.fit(X, y)
    return model

# ---------------- FETCH ODDS ----------------
def fetch_odds(sport_key, market):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": market,
        "oddsFormat": "american"
    }

    r = requests.get(url, params=params)
    r.raise_for_status()
    return r.json()

# ---------------- PROCESS GAMES ----------------
def build_games(sport, config, model, history):
    games = []
    raw = fetch_odds(config["odds_key"], config["market"])

    for g in raw:
        home = g["home_team"]
        away = g["away_team"]

        book = g["bookmakers"][0]
        market = book["markets"][0]

        outcomes = market["outcomes"]
        best = max(outcomes, key=lambda x: x["price"])

        odds = best["price"]
        pick = best["name"]

        market_prob = american_to_prob(odds)

        # ---------------- CALCULATE REAL STATS ----------------
        home_adv = 1 if pick==home else 0
        team_elo = calculate_team_elo(history, pick)
        opp_elo = calculate_team_elo(history, away if pick==home else home)
        recent_form = get_recent_form(history, pick, sport)
        rest_days = get_rest_days(history, pick)
        spread_power = spread_strength(odds)

        # ---------------- MODEL LOGIC ----------------
        if config["mode"]=="prob":  # NBA probability
            if model:
                X_input = [[market_prob, spread_power, home_adv, team_elo, opp_elo, recent_form, rest_days]]
                model_adjustment = model.predict_proba(X_input)[0][1]
                final_prob = (market_prob*0.6) + (model_adjustment*0.4)
            else:
                final_prob = market_prob
            prob = final_prob
            edge = None
        elif config["mode"]=="edge":  # NHL edge
            if model:
                X_input = [[market_prob, home_adv, team_elo, opp_elo, recent_form]]
                model_prob = model.predict_proba(X_input)[0][1]
                edge = (model_prob - market_prob) * 100
            else:
                edge = (market_prob - 0.5)*100
            prob = None

        games.append({
            "Sport": sport,
            "Game": f"{away} @ {home}",
            "Pick": pick,
            "Odds": odds,
            "Probability": prob*100 if prob else None,
            "Edge": edge
        })

    return pd.DataFrame(games)

# ---------------- UI ----------------
st.title("📊 AI Sports Betting App")

history = load_history()

tabs = st.tabs(["NBA", "NHL", "Best Bets", "Performance"])

all_games = []

for sport, config in SPORTS.items():
    with tabs[list(SPORTS.keys()).index(sport)]:
        st.header(sport)

        model = train_model(history, sport)

        try:
            df = build_games(sport, config, model, history)
            all_games.append(df)

            if df.empty:
                st.info("No games available yet.")
            else:
                st.dataframe(df, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to load odds: {e}")

# ---------------- BEST BETS ----------------
with tabs[2]:
    st.header("🔥 Best Bets")

    if all_games:
        bets = pd.concat(all_games)
    else:
        bets = pd.DataFrame()

    if bets.empty:
        st.info("No best bets right now.")
    else:
        best = bets.copy()

        for _, row in best.iterrows():
            st.markdown(f"### {row['Game']}")
            st.markdown(f"**Pick:** {row['Pick']} | **Odds:** {row['Odds']}")

            if row["Sport"]=="NBA":
                st.markdown(f"**Probability:** {row['Probability']:.1f}%")
                st.progress(min(row["Probability"]/100,1))

            if row["Sport"]=="NHL":
                st.markdown(f"**Edge:** {row['Edge']:.1f}%")
                st.progress(min(abs(row["Edge"])/15,1))

            st.divider()

# ---------------- PERFORMANCE ----------------
with tabs[3]:
    st.header("📈 Performance")

    if history.empty:
        st.info("No tracked bets yet.")
    else:
        for sport in history["Sport"].unique():
            h = history[history["Sport"]==sport]
            wins = (h["Result"]=="Win").sum()
            losses = (h["Result"]=="Loss").sum()
            roi = h["Units"].sum()

            st.subheader(sport)
            st.write(f"Record: {wins}-{losses}")
            st.write(f"ROI: {roi:.2f} units")
