import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.linear_model import LogisticRegression

st.set_page_config(page_title="Daily Sports Predictor", layout="wide")

# ---------------- CONFIG ---------------- #
HISTORY_FILE = "bet_history.csv"
SPORTS = ["NBA","NFL","NHL"]

# ---------------- UTILITIES ---------------- #
def init_history():
    if not os.path.exists(HISTORY_FILE):
        pd.DataFrame(columns=[
            "Date","Sport","Home","Away","BetOn","Probability","Result","Units"
        ]).to_csv(HISTORY_FILE,index=False)

def build_elo(history, base=1500, k=20):
    elo = {}
    for _, r in history.iterrows():
        if r["Result"] not in ["WIN","LOSS"]:
            continue
        h, a = r["Home"], r["Away"]
        elo.setdefault(h, base)
        elo.setdefault(a, base)
        expected = 1/(1+10**((elo[a]-elo[h])/400))
        score = 1 if r["BetOn"]==h else 0
        elo[h] += k*(score-expected)
        elo[a] -= k*(score-expected)
    return elo

# ---------------- DAILY SCHEDULES ---------------- #
# Replace with API calls for live games if needed
DAILY_GAMES = {
    "NBA":[
        ("Memphis Grizzlies","Orlando Magic"),
        ("Detroit Pistons","Phoenix Suns"),
        ("San Antonio Spurs","Milwaukee Bucks"),
        ("Miami Heat","Boston Celtics"),
        ("Golden State Warriors","New York Knicks")
    ],
    "NHL":[
        ("San Jose Sharks","Washington Capitals"),
        ("Minnesota Wild","Winnipeg Jets")
    ],
    "NFL":[
        # No games today (Jan 15, 2026)
    ]
}

# ---------------- MODEL ---------------- #
def train_model(sport, history):
    df = history[history["Sport"]==sport]
    df = df[df["Result"].isin(["WIN","LOSS"])]
    elo = build_elo(history)
    if len(df)<10:
        return None, elo  # always return a tuple
    df["EloDiff"] = df["Home"].map(elo).fillna(1500)-df["Away"].map(elo).fillna(1500)
    df["Target"] = (df["Result"]=="WIN").astype(int)
    X = df[["EloDiff"]]
    y = df["Target"]
    model = LogisticRegression()
    model.fit(X,y)
    return model, elo

# ---------------- APP ---------------- #
init_history()
history = pd.read_csv(HISTORY_FILE)
tabs = st.tabs(["NBA","NFL","NHL","🔥 Best Bets","📊 Performance"])
all_bets=[]

today = datetime.utcnow().date().isoformat()

for i,sport in enumerate(SPORTS):
    with tabs[i]:
        games = DAILY_GAMES.get(sport,[])
        if not games:
            st.info("No games today.")
            continue

        # Train model and get Elo
        model, elo = train_model(sport, history)

        table = pd.DataFrame({
            "Home":[g[0] for g in games],
            "Away":[g[1] for g in games]
        })
        table["EloDiff"] = table["Home"].map(elo).fillna(1500)-table["Away"].map(elo).fillna(1500)

        # Predict probabilities
        if model:
            table["Probability"] = model.predict_proba(table[["EloDiff"]])[:,1]
        else:
            table["Probability"] = 0.5 + 0.05*table["EloDiff"]

        # Determine picks
        if sport=="NHL":
            table["Edge %"] = (table["Probability"]-0.5)*200
            table["BetOn"] = np.where(table["Edge %"]>0, table["Home"], table["Away"])
        else:
            table["BetOn"] = np.where(table["Probability"]>=0.5, table["Home"], table["Away"])

        st.dataframe(table,use_container_width=True)

        # Save to history safely
        save = table.copy()
        save["Date"] = today
        save["Sport"] = sport
        save["Result"] = "PENDING"
        save["Units"] = 0
        if "Probability" not in save.columns:
            save["Probability"] = 0.5
        history = pd.concat([history, save[["Date","Sport","Home","Away","BetOn","Probability","Result","Units"]]],ignore_index=True)
        history.to_csv(HISTORY_FILE,index=False)
        all_bets.append(table.assign(Sport=sport))

# ---------------- BEST BETS ---------------- #
with tabs[3]:
    if all_bets:
        best = pd.concat(all_bets,ignore_index=True)
        st.dataframe(best.sort_values(by="Probability",ascending=False),use_container_width=True)

# ---------------- PERFORMANCE ---------------- #
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
