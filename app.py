import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
import joblib
from datetime import datetime
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Self-Learning Sports Model", layout="wide")

# ---------------- CONFIG ---------------- #
API_KEY = st.secrets["ODDS_API_KEY"]
HISTORY_FILE = "bet_history.csv"

SPORTS = {
    "NBA": {"key": "basketball_nba", "model": "nba_model.pkl"},
    "NFL": {"key": "americanfootball_nfl", "model": "nfl_model.pkl"},
    "NHL": {"key": "icehockey_nhl", "model": "nhl_model.pkl"}
}

# ---------------- UTILITIES ---------------- #
def odds_to_prob(odds):
    return 100 / (odds + 100) if odds > 0 else abs(odds) / (abs(odds) + 100)

def init_history():
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=[
            "Date","Sport","Home","Away","BetOn","Odds","Result","Units"
        ]).to_csv(HISTORY_FILE, index=False)

# ---------------- FETCH ODDS ---------------- #
def fetch_odds(sport_key):
    markets = ["spreads","h2h","totals"]
    try:
        r = requests.get(
            f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds",
            params={"apiKey": API_KEY, "regions":"us","markets":",".join(markets),"oddsFormat":"american"},
            timeout=10
        )
        if r.status_code != 200:
            st.warning(f"API returned status {r.status_code} for {sport_key}")
            return pd.DataFrame()
        data = r.json()
        st.write(f"API returned {len(data)} games for {sport_key}")  # debug/log
        rows = []
        for g in data:
            try:
                for book in g["bookmakers"]:
                    for market in book["markets"]:
                        outcomes = market.get("outcomes",[])
                        home = g["home_team"]
                        away = g["away_team"]
                        home_odds = next((o["price"] for o in outcomes if o["name"]==home), None)
                        if home_odds is None:
                            continue
                        rows.append({
                            "Home": home,
                            "Away": away,
                            "Odds": home_odds,
                            "MarketUsed": market["key"],
                            "Bookmaker": book["title"]
                        })
                        break
                    if rows:
                        break
            except:
                continue
        return pd.DataFrame(rows)
    except Exception as e:
        st.error(f"Error fetching odds: {e}")
        return pd.DataFrame()

# ---------------- ELO ---------------- #
def build_elo(history, base=1500, k=20):
    elo = {}
    for _, r in history.iterrows():
        if r["Result"] not in ["WIN","LOSS"]:
            continue
        h, a = r["Home"], r["Away"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)
        expected = 1 / (1 + 10 ** ((elo[a]-elo[h])/400))
        score = 1 if r["BetOn"]==h else 0
        elo[h] += k*(score-expected)
        elo[a] -= k*(score-expected)
    return elo

# ---------------- MODEL ---------------- #
def train_model(sport, history, model_path):
    df = history[history["Sport"]==sport]
    df = df[df["Result"].isin(["WIN","LOSS"])]
    if len(df)<30:
        return None
    elo = build_elo(df)
    df["EloDiff"] = df["Home"].map(elo).fillna(1500) - df["Away"].map(elo).fillna(1500)
    df["MarketProb"] = df["Odds"].apply(odds_to_prob)
    df["Target"] = (df["Result"]=="WIN").astype(int)
    X = df[["EloDiff","MarketProb"]]
    y = df["Target"]
    model = LogisticRegression()
    model.fit(X,y)
    joblib.dump(model, model_path)
    return model

def load_or_train(sport, history, model_path):
    if os.path.exists(model_path):
        try:
            return joblib.load(model_path)
        except:
            pass
    return train_model(sport, history, model_path)

# ---------------- GRADE PENDING ---------------- #
def grade_bets(history):
    pending = history[history["Result"]=="PENDING"]
    if pending.empty:
        return history
    for idx, row in pending.iterrows():
        try:
            scores = requests.get(f"https://api.the-odds-api.com/v4/sports/{SPORTS[row['Sport']]['key']}/scores",
                                  params={"apiKey":API_KEY}, timeout=10).json()
            for g in scores:
                if g["home_team"]==row["Home"] and g.get("completed",False):
                    home_score = g["scores"][0]["score"]
                    away_score = g["scores"][1]["score"]
                    winner = row["Home"] if home_score>away_score else row["Away"]
                    history.at[idx,"Result"]="WIN" if row["BetOn"]==winner else "LOSS"
                    history.at[idx,"Units"]=1 if row["BetOn"]==winner else -1
        except:
            continue
    history.to_csv(HISTORY_FILE,index=False)
    return history

# ---------------- APP ---------------- #
init_history()
history = pd.read_csv(HISTORY_FILE)
history = grade_bets(history)
tabs = st.tabs(["NBA","NFL","NHL","🔥 Best Bets","📊 Performance"])
all_bets=[]

for i,sport in enumerate(["NBA","NFL","NHL"]):
    with tabs[i]:
        odds = fetch_odds(SPORTS[sport]["key"])
        model = load_or_train(sport, history, SPORTS[sport]["model"])
        elo = build_elo(history[history["Sport"]==sport])

        # ---------- Fallback if no odds ----------
        if odds.empty:
            st.info("No live odds yet. Using Elo-only predictions.")
            # Try to get matchups from API h2h if available
            try:
                r = requests.get(
                    f"https://api.the-odds-api.com/v4/sports/{SPORTS[sport]['key']}/odds",
                    params={"apiKey": API_KEY, "regions":"us","markets":"h2h,totals","oddsFormat":"american"},
                    timeout=10
                )
                data = r.json()
                teams=[]
                for g in data:
                    teams.append((g["home_team"], g["away_team"]))
            except:
                teams=[]
            if teams:
                table = pd.DataFrame({
                    "Home": [t[0] for t in teams],
                    "Away": [t[1] for t in teams],
                    "EloDiff": [elo.get(t[0],1500)-elo.get(t[1],1500) for t in teams]
                })
                table["Probability %"] = 50 + 0.05*table["EloDiff"]  # Elo-based probability
                table["BetOn"] = np.where(table["Probability %"]>=50, table["Home"], table["Away"])
                st.dataframe(table,use_container_width=True)
            else:
                st.warning("No games found yet.")
            continue

        # ---------- Normal odds processing ----------
        odds["EloDiff"] = odds["Home"].map(elo).fillna(1500) - odds["Away"].map(elo).fillna(1500)
        odds["MarketProb"] = odds["Odds"].apply(odds_to_prob)
        if model:
            odds["ModelProb"] = model.predict_proba(odds[["EloDiff","MarketProb"]])[:,1]
        else:
            odds["ModelProb"] = odds["MarketProb"]
        if sport=="NHL":
            odds["Edge %"] = ((odds["ModelProb"]-odds["MarketProb"])*100).round(2)
            odds["BetOn"] = np.where(odds["Edge %"]>0, odds["Home"], odds["Away"])
        else:
            odds["Probability %"] = (odds["ModelProb"]*100).round(1)
            odds["BetOn"] = np.where(odds["ModelProb"]>=0.5, odds["Home"], odds["Away"])
        st.dataframe(odds,use_container_width=True)

        save=odds.copy()
        save["Date"]=datetime.utcnow().date().isoformat()
        save["Sport"]=sport
        save["Result"]="PENDING"
        save["Units"]=0
        history=pd.concat([history,save[["Date","Sport","Home","Away","BetOn","Odds","Result","Units"]]])
        history.to_csv(HISTORY_FILE,index=False)
        all_bets.append(odds.assign(Sport=sport))

with tabs[3]:
    if all_bets:
        st.dataframe(pd.concat(all_bets),use_container_width=True)

with tabs[4]:
    for sport in SPORTS:
        df = history[history["Sport"]==sport]
        wins = (df["Result"]=="WIN").sum()
        losses = (df["Result"]=="LOSS").sum()
        units = df["Units"].sum()
        roi = (units/max(len(df),1))*100
        st.subheader(sport)
        st.metric("Record",f"{wins}-{losses}")
        st.metric("ROI %",f"{roi:.1f}%")
        st.metric("Units",f"{units:.1f}")
