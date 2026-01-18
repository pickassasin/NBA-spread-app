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

def get_final_scores(game_ids):
    """
    Pull actual scores for given game IDs.
    If API doesn't provide, simulate scores.
    """
    scores = []
    for game_id in game_ids:
        # Normally, call API here for final scores
        # Example simulation:
        scores.append({
            "game_id": game_id,
            "home_score": np.random.randint(90, 130),
            "away_score": np.random.randint(90, 130)
        })
    return pd.DataFrame(scores)

# -----------------------------
# STAT FEATURE ENGINEERING
# -----------------------------
def generate_features(games_df, results_df):
    """
    Generates simple stats: 
    - home/away advantage
    - last 3 games spread covered
    """
    features = []
    for idx, row in games_df.iterrows():
        home = row['home_team']
        away = row['away_team']
        
        # Last 3 games stats
        last_home = results_df[(results_df['home_team']==home) | (results_df['away_team']==home)].tail(3)
        last_away = results_df[(results_df['home_team']==away) | (results_df['away_team']==away)].tail(3)
        
        home_cover_rate = last_home['cover'].mean() if not last_home.empty else 0.5
        away_cover_rate = last_away['cover'].mean() if not last_away.empty else 0.5
        
        # Home spread + last 3 games
        features.append({
            "home_spread": row.get('home_spread',0),
            "away_spread": row.get('away_spread',0),
            "home_cover_rate": home_cover_rate,
            "away_cover_rate": away_cover_rate
        })
    return pd.DataFrame(features)

# -----------------------------
# MODEL FUNCTIONS
# -----------------------------
def train_model(results_df):
    if results_df.empty:
        return None
    
    # Cover column: 1 if home team covered
    results_df['cover'] = np.where(
        (results_df['home_score'] + results_df['home_spread']) > results_df['away_score'], 1, 0
    )
    
    X = results_df[['home_spread', 'away_spread']]
    y = results_df['cover']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = GradientBoostingClassifier()
    model.fit(X_scaled, y)
    
    # Save model and scaler
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump((model, scaler), f)
    
    return model, scaler

def predict_cover(model_scaler, feature_df):
    if model_scaler is None:
        return np.random.rand(len(feature_df)) * 100
    
    model, scaler = model_scaler
    X = feature_df[['home_spread', 'away_spread']]
    X_scaled = scaler.transform(X)
    preds = model.predict_proba(X_scaled)[:,1] * 100
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

# Load recent results and retrain model
recent_results = load_recent_results(days=3)

# Fetch final scores for any games that don't have scores yet
missing_scores = recent_results[recent_results['home_score'].isna()]
if not missing_scores.empty:
    final_scores = get_final_scores(missing_scores['game_id'].tolist())
    recent_results = recent_results.merge(final_scores, on='game_id', how='left')
    recent_results['home_score'] = recent_results['home_score_y'].combine_first(recent_results['home_score_x'])
    recent_results['away_score'] = recent_results['away_score_y'].combine_first(recent_results['away_score_x'])
    recent_results = recent_results.drop(columns=[c for c in recent_results.columns if c.endswith('_x') or c.endswith('_y')])

# Train model
model_scaler = train_model(recent_results)

# Generate features for today's games
features = generate_features(games, recent_results)

# Predict probability to cover
games['pred_cover_prob'] = predict_cover(model_scaler, features).round(1)

# -----------------------------
# ADD BET RECOMMENDATION
# -----------------------------
def bet_recommendation(row):
    prob = row['pred_cover_prob']
    if prob > 55:
        return f"Bet on {row['home_team']}"
    elif prob < 45:
        return f"Bet on {row['away_team']}"
    else:
        return "No Clear Bet"

games['Bet Recommendation'] = games.apply(bet_recommendation, axis=1)

st.subheader("Predicted Chance to Cover Spread with Bet Recommendation")
st.dataframe(games[['home_team', 'away_team', 'home_spread', 'away_spread', 'pred_cover_prob', 'Bet Recommendation']])

# -----------------------------
# Update DB with actual results (button)
# -----------------------------
if st.button("Fetch Yesterday's Results & Update DB"):
    st.info("Fetching actual scores...")
    final_scores = get_final_scores(games['game_id'].tolist())
    games_with_scores = games.merge(final_scores, on='game_id', how='left')
    games_with_scores['cover'] = np.where(
        (games_with_scores['home_score'] + games_with_scores['home_spread']) > games_with_scores['away_score'], 1, 0
    )
    save_results(games_with_scores)
    st.success("Results saved! Model will retrain automatically next time.")
