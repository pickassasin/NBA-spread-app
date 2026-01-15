import streamlit as st
import requests, pandas as pd, numpy as np, os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(layout="wide", page_title="Auto Sports Betting App")

API_KEY = st.secrets["ODDS_API_KEY"]

LOGS = {
    "NBA": "nba_bets.csv",
    "NFL": "nfl_bets.csv",
    "NHL": "nhl_bets.csv"
}

EDGE_ALERT = 6

# ---------------- UTIL ---------------- #

def safe_json(url, params=None):
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

def odds_to_prob(o):
    return 100/(o+100) if o>0 else abs(o)/(abs(o)+100)

def prob_to_odds(p):
    return int(-100*p/(1-p)) if p>0.5 else int(100*(1-p)/p)

def confidence(x):
    if x >= 6: return "HIGH"
    if x >= 3: return "MED"
    return "LOW"

def init_log(path):
    if not os.path.exists(path):
        pd.DataFrame(columns=[
            "Date","Sport","Home","Away","Bet On",
            "Book Odds","Model Prob","Result"
        ]).to_csv(path, index=False)

def log_bets(df, sport):
    path = LOGS[sport]
    init_log(path)
    old = pd.read_csv(path)
    df["Date"] = datetime.utcnow().date()
    df["Sport"] = sport
    df["Result"] = "PENDING"
    new = pd.concat([old, df]).drop_duplicates(
        subset=["Date","Sport","Home","Away","Bet On"]
    )
    new.to_csv(path, index=False)

# ---------------- RESULTS ---------------- #

def update_results():
    for sport, path in LOGS.items():
        if not os.path.exists(path): continue
        df = pd.read_csv(path)
        pending = df[df["Result"]=="PENDING"]

        for i,r in pending.iterrows():
            d = r["Date"]

            try:
                if sport=="NBA":
                    url=f"https://www.balldontlie.io/api/v1/games?start_date={d}&end_date={d}"
                    data = safe_json(url)
                    if not data: continue
                    for g in data["data"]:
                        if g["home_team"]["full_name"]==r["Home"]:
                            h,a=g["home_team_score"],g["visitor_team_score"]
                            df.at[i,"Result"]="WIN" if (
                                (h>a and r["Bet On"]==r["Home"]) or
                                (a>h and r["Bet On"]==r["Away"])
                            ) else "LOSS"

                elif sport=="NHL":
                    url=f"https://statsapi.web.nhl.com/api/v1/schedule?date={d}"
                    data = safe_json(url)
                    if not data: continue
                    for day in data.get("dates",[]):
                        for g in day["games"]:
                            h=g["teams"]["home"]["team"]["name"]
                            a=g["teams"]["away"]["team"]["name"]
                            if h==r["Home"]:
                                hs,as_=g["teams"]["home"]["score"],g["teams"]["away"]["score"]
                                df.at[i,"Result"]="WIN" if (
                                    (hs>as_ and r["Bet On"]==h) or
                                    (as_>hs and r["Bet On"]==a)
                                ) else "LOSS"

                elif sport=="NFL":
                    url="https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
                    data = safe_json(url)
                    if not data: continue
                    for g in data.get("events",[]):
                        teams=g["competitions"][0]["competitors"]
                        h=teams[0]["team"]["displayName"]
                        a=teams[1]["team"]["displayName"]
                        if h==r["Home"]:
                            hs=int(teams[0]["score"])
                            as_=int(teams[1]["score"])
                            df.at[i,"Result"]="WIN" if (
                                (hs>as_ and r["Bet On"]==h) or
                                (as_>hs and r["Bet On"]==a)
                            ) else "LOSS"
            except:
                continue

        df.to_csv(path, index=False)

# ---------------- MODEL ---------------- #

def run_model(df, log_path):
    if df.empty:
        df["Model Prob"]=0.52
        return df

    hist = pd.read_csv(log_path) if os.path.exists(log_path) else pd.DataFrame()
    hist = hist[hist["Result"].isin(["WIN","LOSS"])]

    try:
        le = LabelEncoder()
        df["H"]=le.fit_transform(df["Home"])
        df["A"]=le.fit_transform(df["Away"])
        X=df[["H","A","Book Odds"]]

        if not hist.empty:
            hist["H"]=le.fit_transform(hist["Home"])
            hist["A"]=le.fit_transform(hist["Away"])
            y=(hist["Result"]=="WIN").astype(int)
            model=RandomForestClassifier()
            model.fit(hist[["H","A","Book Odds"]],y)
            df["Model Prob"]=model.predict_proba(X)[:,1]
        else:
            df["Model Prob"]=0.45+0.1*np.random.rand(len(df))

        df["Model Prob"]=df["Model Prob"].clip(0.25,0.75)
    except:
        df["Model Prob"]=0.52

    return df

# ---------------- ODDS ---------------- #

def fetch_odds(sport, market):
    key={
        "NBA":"basketball_nba",
        "NFL":"americanfootball_nfl",
        "NHL":"icehockey_nhl"
    }[sport]

    url=f"https://api.the-odds-api.com/v4/sports/{key}/odds"
    data=safe_json(url,{
        "apiKey":API_KEY,"regions":"us","markets":market,"oddsFormat":"american"
    })

    games=[]
    if not data: return pd.DataFrame()

    for g in data:
        try:
            bm=g["bookmakers"][0]
            out=bm["markets"][0]["outcomes"]
            home=g["home_team"]
            price=next(o["price"] for o in out if o["name"]==home)
            games.append({
                "Home":g["home_team"],
                "Away":g["away_team"],
                "Book Odds":price
            })
        except:
            continue

    return pd.DataFrame(games)

# ---------------- MAIN ---------------- #

update_results()

tabs = st.tabs(["NBA","NFL","NHL","🔥 Best Bets"])

ALL_BETS=[]

for i,sport in enumerate(["NBA","NFL","NHL"]):
    with tabs[i]:
        market="spreads" if sport!="NHL" else "h2h"
        df=fetch_odds(sport,market)
        df=run_model(df,LOGS[sport])

        if df.empty:
            st.info("No games available.")
            continue

        df["Bet On"]=np.where(df["Model Prob"]>=0.5,df["Home"],df["Away"])

        if sport=="NHL":
            df["Edge %"]=(df["Model Prob"]-df["Book Odds"].apply(odds_to_prob))*100
            df["Strength"]=df["Edge %"].abs()
        else:
            df["Probability %"]=df["Model Prob"]*100
            df["Strength"]=abs(df["Probability %"]-50)

        df["Confidence"]=df["Strength"].apply(confidence)

        log_bets(df[["Home","Away","Bet On","Book Odds","Model Prob"]],sport)
        ALL_BETS.append(df.assign(Sport=sport))

        st.dataframe(
            df.sort_values("Strength",ascending=False),
            use_container_width=True
        )

with tabs[3]:
    best=pd.concat(ALL_BETS)
    best=best[best["Strength"]>=EDGE_ALERT]
    st.dataframe(best.sort_values("Strength",ascending=False),use_container_width=True)
