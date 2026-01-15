import streamlit as st
import requests
import pandas as pd
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

# ======================
# CONFIG
# ======================
st.set_page_config(page_title="Sports Betting Edge App", layout="wide")
API_KEY = st.secrets["ODDS_API_KEY"]

NBA_LOG = "nba_bets.csv"
NFL_LOG = "nfl_bets.csv"
NHL_LOG = "nhl_bets.csv"

HIGH_EDGE_THRESHOLD = 6  # %

# ======================
# HELPERS
# ======================
def odds_to_prob(odds):
    try:
        return 100/(odds+100) if odds>0 else abs(odds)/(abs(odds)+100)
    except:
        return 0.5

def prob_to_odds(p):
    try:
        return int(-100*p/(1-p)) if p>0.5 else int(100*(1-p)/p)
    except:
        return 0

def edge_confidence(edge):
    if edge>=6:
        return "HIGH"
    if edge>=3:
        return "MEDIUM"
    return "LOW"

def safe_request(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code !=200:
            return []
        return r.json()
    except:
        return []

def pick_team(row):
    return row["Home Team"] if row["Edge %"]>0 else row["Away Team"]

def init_log(path):
    if not os.path.exists(path):
        pd.DataFrame(columns=["Date","Sport","Home Team","Away Team","Bet On","Book Odds","Fair Odds","Edge %","Result"]).to_csv(path,index=False)

def log_bets(df, sport, path):
    init_log(path)
    existing = pd.read_csv(path)
    new = df.copy()
    new["Date"] = datetime.utcnow().date()
    new["Sport"] = sport
    new["Result"] = "PENDING"
    merged = pd.concat([existing,new],ignore_index=True)
    merged.drop_duplicates(subset=["Date","Sport","Home Team","Away Team","Bet On"], inplace=True)
    merged.to_csv(path,index=False)

def calculate_stats(path):
    if not os.path.exists(path):
        return 0,0
    df = pd.read_csv(path)
    df = df[df["Result"].isin(["WIN","LOSS"])]
    if df.empty:
        return 0,0
    wins = (df["Result"]=="WIN").sum()
    total = len(df)
    def profit(row):
        if row["Result"]=="WIN":
            odds=row["Book Odds"]
            return (odds/100) if odds>0 else (100/abs(odds))
        return -1
    df["Profit"] = df.apply(profit,axis=1)
    roi = (df["Profit"].sum()/total)*100
    return wins,roi

# ======================
# FETCH ODDS
# ======================
def fetch_odds(sport_key, market):
    url = f"https://api.the-odds-api.com/v4/sports/{sport_key}/odds"
    params = {"apiKey":API_KEY,"regions":"us","markets":market,"oddsFormat":"american"}
    data = safe_request(url, params)
    games=[]
    for g in data:
        try:
            if not g.get("bookmakers"):
                continue
            bm = g["bookmakers"][0]
            if not bm.get("markets"):
                continue
            if market=="spreads":
                outcome = bm["markets"][0]["outcomes"][0]
            else:  # NHL moneyline
                home = g["home_team"]
                outcomes = bm["markets"][0]["outcomes"]
                outcome = next((o for o in outcomes if o["name"]==home), None)
                if outcome is None:
                    continue
            games.append({"Home Team":g["home_team"],"Away Team":g["away_team"],"Book Odds":outcome["price"]})
        except:
            continue
    return pd.DataFrame(games)

def run_model(df, log_path):
    if df.empty:
        df["Model_Prob"]=0.52
        return df
    if not os.path.exists(log_path):
        df["Model_Prob"]=0.52
        return df
    hist = pd.read_csv(log_path)
    hist = hist[hist["Result"].isin(["WIN","LOSS"])]
    if hist.empty:
        df["Model_Prob"]=0.52
        return df
    try:
        X = hist[["Book Odds"]]
        y = (hist["Result"]=="WIN").astype(int)
        model = RandomForestClassifier()
        model.fit(X,y)
        df["Model_Prob"] = model.predict_proba(df[["Book Odds"]])[:,1]
    except:
        df["Model_Prob"]=0.52
    return df

# ======================
# AUTO-FETCH RESULTS
# ======================
def update_results(log_path, sport):
    if not os.path.exists(log_path):
        return
    df = pd.read_csv(log_path)
    today = datetime.utcnow().date()
    df_pending = df[df["Result"]=="PENDING"]
    if df_pending.empty:
        return
    for idx,row in df_pending.iterrows():
        try:
            if sport=="NBA":
                url=f"https://www.balldontlie.io/api/v1/games?start_date={row['Date']}&end_date={row['Date']}&per_page=100"
                games = safe_request(url)
                game = next((g for g in games if g["home_team"]["full_name"]==row["Home Team"] and g["visitor_team"]["full_name"]==row["Away Team"]),None)
                if game:
                    df.at[idx,"Result"] = "WIN" if game["home_team_score"]>game["visitor_team_score"] and row["Bet On"]==row["Home Team"] else "LOSS" if row["Bet On"]==row["Home Team"] else "WIN" if game["home_team_score"]<game["visitor_team_score"] and row["Bet On"]==row["Away Team"] else "LOSS"
            elif sport=="NFL":
                # Placeholder: set PENDING; can integrate free NFL API if available
                df.at[idx,"Result"]="PENDING"
            else:  # NHL
                url=f"https://statsapi.web.nhl.com/api/v1/schedule?teamId=&date={row['Date']}"
                data = safe_request(url)
                # basic placeholder logic; real score fetch would require team matching
                df.at[idx,"Result"]="PENDING"
        except:
            continue
    df.to_csv(log_path,index=False)

# ======================
# MAIN APP
# ======================
screen = st.sidebar.radio("Select Sport", ["NBA","NFL","NHL"])
st.title(f"{screen} Betting Edge Finder")

# Fetch odds
if screen=="NBA":
    df = fetch_odds("basketball_nba","spreads")
    df = run_model(df,NBA_LOG)
    LOG_PATH = NBA_LOG
elif screen=="NFL":
    df = fetch_odds("americanfootball_nfl","spreads")
    df = run_model(df,NFL_LOG)
    LOG_PATH = NFL_LOG
else:
    df = fetch_odds("icehockey_nhl","h2h")
    df = run_model(df,NHL_LOG)
    LOG_PATH = NHL_LOG

# Update results automatically
update_results(LOG_PATH,screen)

if df.empty:
    st.info("No games available right now.")
    st.stop()

# Calculate edge & bets
df["Book Prob"] = df["Book Odds"].apply(odds_to_prob)
df["Edge %"] = (df["Model_Prob"]-df["Book Prob"])*100 if screen=="NHL" else df["Model_Prob"]*100
df["Fair Odds"] = df["Model_Prob"].apply(prob_to_odds)
df["Confidence"] = df["Edge %"].apply(edge_confidence)
df["Bet On"] = df.apply(pick_team,axis=1)
df["Abs Edge %"] = df["Edge %"].abs()

# HIGH EDGE ALERT
high_edge = df[df["Abs Edge %"]>=HIGH_EDGE_THRESHOLD]
if not high_edge.empty:
    st.toast("🔥 HIGH EDGE BETS AVAILABLE",icon="🔥")
    st.warning(f"{len(high_edge)} high-edge opportunities detected.")

# LOG BETS
log_bets(df[["Home Team","Away Team","Bet On","Book Odds","Fair Odds","Edge %"]],screen,LOG_PATH)

# DISPLAY DATAFRAME
st.dataframe(df[["Bet On","Home Team","Away Team","Book Odds","Fair Odds","Abs Edge %","Confidence"]].sort_values("Abs Edge %",ascending=False),use_container_width=True)

# SHOW RECORD & ROI
wins, roi = calculate_stats(LOG_PATH)
total_bets = len(pd.read_csv(LOG_PATH)[pd.read_csv(LOG_PATH)["Result"].isin(["WIN","LOSS"])])
st.subheader(f"{screen} Historical Stats")
st.write(f"Record: {wins} wins / {total_bets-wins} losses | ROI: {roi:.2f}%")
