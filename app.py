import requests, pandas as pd, numpy as np, os
from datetime import datetime, timedelta
import streamlit as st
from sklearn.ensemble import RandomForestClassifier

# ===== CONFIG =====
API_KEY = st.secrets["ODDS_API_KEY"]
SPORT = "basketball_nba"
CSV_FILE = "nba_bet_results.csv"
STAKE = 100
ODDS = -110
PAYOUT = STAKE * (100 / abs(ODDS))

st.set_page_config(page_title="NBA Spread Predictor", layout="wide")
st.title("🏀 NBA Spread Predictor")

# ===== LOAD / CREATE LOG =====
if os.path.exists(CSV_FILE):
    bets_log = pd.read_csv(CSV_FILE)
else:
    bets_log = pd.DataFrame(columns=[
        "Date","Home Team","Away Team","Spread",
        "Home Cover %","Confidence","Bet Placed","Result","Profit"
    ])

def confidence(p):
    if p >= 0.60: return "HIGH"
    if p >= 0.55: return "MEDIUM"
    return "LOW"

@st.cache_data(ttl=300)
def fetch_odds():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {"apiKey": API_KEY, "regions": "us", "markets": "spreads"}
    r = requests.get(url, params=params).json()
    games = []
    for g in r:
        if not g.get("bookmakers"): continue
        market = g["bookmakers"][0]["markets"][0]["outcomes"]
        games.append({
            "Away Team": g["away_team"],
            "Home Team": g["home_team"],
            "Spread": market[0]["point"]
        })
    return pd.DataFrame(games)

live_games = fetch_odds()

results = []
for i in range(3):
    d = (datetime.today() - timedelta(days=i)).strftime("%Y-%m-%d")
    url = f"https://www.balldontlie.io/api/v1/games?start_date={d}&end_date={d}&per_page=100"
    for g in requests.get(url).json()["data"]:
        results.append({
            "Home Team": g["home_team"]["full_name"],
            "Away Team": g["visitor_team"]["full_name"],
            "Home Score": g["home_team_score"],
            "Away Score": g["visitor_team_score"]
        })
results = pd.DataFrame(results)

def calc_result(row):
    m = results[
        (results["Home Team"] == row["Home Team"]) &
        (results["Away Team"] == row["Away Team"])
    ]
    if m.empty: return ""
    return "Win" if (m.iloc[0]["Home Score"] - m.iloc[0]["Away Score"]) > row["Spread"] else "Lose"

live_games["Result"] = live_games.apply(calc_result, axis=1)
live_games["Home Cover %"] = np.clip(0.5 - live_games["Spread"] * 0.025, 0.40, 0.65)

train = bets_log[bets_log["Result"].isin(["Win","Lose"])]
if len(train) >= 5:
    X = train[["Spread","Home Cover %"]]
    y = train["Result"].map({"Win":1,"Lose":0})
    model = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42)
    model.fit(X,y)
    live_games["Home Cover %"] = model.predict_proba(
        live_games[["Spread","Home Cover %"]]
    )[:,1]

live_games["Confidence"] = live_games["Home Cover %"].apply(confidence)

today = datetime.today().strftime("%Y-%m-%d")
for _, r in live_games.iterrows():
    if r["Confidence"] == "HIGH":
        if not ((bets_log["Date"] == today) & (bets_log["Home Team"] == r["Home Team"])).any():
            bets_log = pd.concat([bets_log, pd.DataFrame([{
                "Date": today,
                "Home Team": r["Home Team"],
                "Away Team": r["Away Team"],
                "Spread": r["Spread"],
                "Home Cover %": r["Home Cover %"],
                "Confidence": "HIGH",
                "Bet Placed": "Yes",
                "Result": r["Result"],
                "Profit": 0
            }])], ignore_index=True)

bets_log["Profit"] = 0
bets_log.loc[bets_log["Result"] == "Win", "Profit"] = PAYOUT
bets_log.loc[bets_log["Result"] == "Lose", "Profit"] = -STAKE
bets_log.to_csv(CSV_FILE, index=False)

st.subheader("🔥 Best Bets")
st.dataframe(
    live_games[live_games["Confidence"]=="HIGH"]
    .sort_values("Home Cover %", ascending=False),
    use_container_width=True
)

st.subheader("📊 All Games")
st.dataframe(live_games, use_container_width=True)

profit = bets_log["Profit"].sum()
bets = len(bets_log)
roi = profit / (bets * STAKE) if bets else 0

st.subheader("💰 Performance")
st.metric("Total Profit", f"${profit:.2f}")
st.metric("ROI", f"{roi*100:.2f}%")
