# app.py
import streamlit as st
import requests
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np
import os
import pickle

# -----------------------------
# CONFIG
# -----------------------------
API_KEY = "1ba9374915afd7c955dfec5e98e8dbd9"
SPORT = "basketball_nba"
REGION = "us"
MARKET = "spreads"
DB_FILE = "nba_results.db"
MODEL_FILE = "spread_model.pkl"

# -----------------------------
# DATABASE FUNCTIONS
# -----------------------------
def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS results (
            game_id TEXT PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_spread REAL,
            away_spread REAL,
            cover INTEGER,
            commence_time TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_results(results_df):
    conn = sqlite3.connect(DB_FILE)
    results_df.to_sql('results', conn, if_exists='replace', index=False)
    conn.close()

def load_recent_results(days=3):
    conn = sqlite3.connect(DB_FILE)
    query = f"SELECT * FROM results WHERE commence_time >= datetime('now','-{days} days')"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# -----------------------------
# API FUNCTIONS
# -----------------------------
def get_todays_games():
    url = f"https://api.the-odds-api.com/v4/sports/{SPORT}/odds"
    params = {
        "apiKey": API_KEY,
        "regions": REGION,
        "markets": MARKET,
        "dateFormat": "iso"
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    games = []
    for game in data:
        game_dict = {
            "game_id": game['id'],
            "home_team": game['home_team'],
            "away_team": game['away_team'],
            "commence_time": game['commence_time'],
        }
        # Get spreads from first bookmaker
        if game['bookmakers']:
            markets = game['bookmakers'][0]['markets']
            if markets:
                outcomes = markets[0]['outcomes']
                for o in outcomes:
                    if o['name'] == game['home_team']:
                        game_dict['home_spread'] = o['point']
                    elif o['name'] == game['away_team']:
                        game_dict['away_spread'] = o['point']
        games.append(game_dict)
    return pd.DataFrame(games)

# -----------------------------
# MODEL FUNCTIONS
# -----------------------------
def train_model(results_df):
    if results_df.empty:
        st.warning("No past results yet. Using random predictions.")
        return None
    
    # Features: simple stats (can expand later)
    results_df['home_adv'] = results_df['home_score'] - results_df['away_score']
    results_df['cover'] = np.where(results_df['home_adv'] + results_df['home_spread'] > 0, 1, 0)
    
    X = results_df[['home_spread', 'away_spread']]  # simple features
    y = results_df['cover']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)
    
    # Save model and scaler
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump((model, scaler), f)
    
    return model, scaler

def predict_cover(model_scaler, games_df):
    if model_scaler is None:
        return np.random.rand(len(games_df))  # fallback random probabilities
    
    model, scaler = model_scaler
    X_pred = games_df[['home_spread', 'away_spread']].fillna(0)
    X_scaled = scaler.transform(X_pred)
    preds = model.predict_proba(X_scaled)[:,1]
    return preds

# -----------------------------
# STREAMLIT APP
# -----------------------------
st.title("NBA Spread Predictor")

# Initialize DB
init_db()

# Pull today's games
st.subheader("Today's NBA Games")
games = get_todays_games()
st.dataframe(games)

# Load recent results & train model
recent_results = load_recent_results(days=3)
model_scaler = train_model(recent_results)

# Predict % chance to cover
games['pred_cover_prob'] = predict_cover(model_scaler, games) * 100
games['pred_cover_prob'] = games['pred_cover_prob'].round(1)

st.subheader("Predicted Chance to Cover Spread")
st.dataframe(games[['home_team', 'away_team', 'home_spread', 'away_spread', 'pred_cover_prob']])

# -----------------------------
# OPTIONAL: Update DB with actual results (if API provides results)
# -----------------------------
if st.button("Fetch Yesterday's Results & Update DB"):
    st.info("Fetching results...")
    # You'd call the odds API / another source for final scores here
    # Example: just simulating results
    results_df = games.copy()
    results_df['home_score'] = np.random.randint(90, 130, len(results_df))
    results_df['away_score'] = np.random.randint(90, 130, len(results_df))
    results_df['cover'] = np.where(results_df['home_score'] + results_df['home_spread'] > results_df['away_score'], 1, 0)
    
    save_results(results_df)
    st.success("Results saved! Model will use these for next predictions.")
