import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime

st.set_page_config(page_title="Sports Betting Model", layout="wide")

API_KEY = st.secrets["ODDS_API_KEY"]
HISTORY_FILE = "bet_history.csv"

SPORTS = {
    "NBA": {
        "key": "basketball_nba",
        "market": "spreads",
        "type": "probability"
    },
    "NFL": {
        "key": "americanfootball_nfl",
        "market": "spreads",
        "type": "probability"
    },
    "NHL": {
        "key": "icehockey_nhl",
        "market": "h2h",
        "type": "edge"
    }
}

# ---------------- INIT ----------------
if not os.path.exists(HISTORY_FILE):
    pd.DataFrame(columns=[
        "Date","Sport","Home","Away","BetOn",
        "Odds","Probability","Result","Units"
    ]).to_csv(HISTORY_FILE,index=False)

history = pd.read_csv(HISTORY_FILE)

# ---------------- ELO ----------------
def build_elo(hist, base=1500, k=20):
    elo = {}
    for _, r in hist.iterrows():
        if r["Result"] not in ["WIN","LOSS"]:
            continue
        h,a = r["Home"], r["Away"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)
        expected = 1/(1+10**((elo[a]-elo[h])/400))
        score = 1 if r["BetOn"]==h else 0
        elo[h] += k*(score-expected)
        elo[a] -= k*(score-expected)
    return elo

elo = build_elo(history)

# ---------------- FETCH ODDS ----------------
def fetch_games(sport):
    url = "https://api.the-odds-api.com/v4/sports/{}/odds".format(sport["key"])
    params = {
        "apiKey": API_KEY,
        "markets": sport["market"],
        "oddsFormat": "american",
        "regions": "us"
    }
    r = requests.get(url, params=params, timeout=10)
    r.raise_for_status()
    return r.json()

# ---------------- UI ----------------
tabs = st.tabs(["NBA","NFL","NHL","📊 Performance","🔥 Best Bets"])
today = datetime.utcnow().date().isoformat()
all_bets = []

for i,(sport_name, sport) in enumerate(SPORTS.items()):
    with tabs[i]:
        try:
            games = fetch_games(sport)
        except:
            st.error("Failed to load odds.")
            continue

        if not games:
            st.info("No games with odds available yet.")
            continue

        rows = []
        for g in games:
            if not g["bookmakers"]:
                continue
            market = g["bookmakers"][0]["markets"][0]
            outcomes = market["outcomes"]
            home = g["home_team"]
            away = g["away_team"]

            elo.setdefault(home,1500)
            elo.setdefault(away,1500)
            elo_diff = elo[home]-elo[away]
            prob = np.clip(1/(1+np.exp(-elo_diff/200)),0,1)

            if sport["type"]=="edge":
                odds_home = next(o["price"] for o in outcomes if o["name"]==home)
                implied = abs(odds_home)/(abs(odds_home)+100) if odds_home<0 else 100/(odds_home+100)
                edge = (prob-implied)*100
                bet_on = home if edge>0 else away
                rows.append([home,away,bet_on,odds_home,prob,edge])
            else:
                bet_on = home if prob>=0.5 else away
                rows.append([home,away,bet_on,None,prob,None])

        df = pd.DataFrame(rows,columns=[
            "Home","Away","BetOn","Odds","Probability","Edge %"
        ])

        st.dataframe(df,use_container_width=True)

        save = df.copy()
        save["Date"] = today
        save["Sport"] = sport_name
        save["Result"] = "PENDING"
        save["Units"] = 0

        history = pd.concat([
            history,
            save[history.columns]
        ],ignore_index=True)

        history.to_csv(HISTORY_FILE,index=False)
        all_bets.append(df.assign(Sport=sport_name))

# ---------------- PERFORMANCE ----------------
with tabs[3]:
    for s in SPORTS:
        df = history[history["Sport"]==s]
        wins = (df["Result"]=="WIN").sum()
        losses = (df["Result"]=="LOSS").sum()
        units = df["Units"].sum()
        roi = (units/max(len(df),1))*100
        st.subheader(s)
        st.write(f"Record: {wins}-{losses}")
        st.write(f"ROI: {roi:.2f}%")
        st.write(f"Units: {units:.2f}")

# ---------------- BEST BETS ----------------
with tabs[4]:
    if all_bets:
        best = pd.concat(all_bets)
        if "Edge %" in best.columns:
            st.dataframe(best.sort_values(by=["Edge %","Probability"],ascending=False))
        else:
            st.dataframe(best.sort_values(by="Probability",ascending=False))
