# predict.py
from math import sqrt
import os
import pandas as pd
import joblib
from scipy.stats import norm
import requests
import time
import json

from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# ===============================
# Config
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(BASE_DIR, "model", "nba_points_model.pkl")
CACHE_DIR = os.path.join(BASE_DIR, "data", "player_cache")
OUTPUT_PATH = os.path.join(BASE_DIR, "public", "picks.json")
META_PATH = os.path.join(BASE_DIR, "model", "model_meta.json")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)


FEATURES = [
    "pts_last5", "pts_last10", "pts_std_5", "season_pts_avg",
    "min_last5", "min_last10", "minutes_trend",
    "fga_last5", "fga_last10", "fga_trend",
    "home_flag", "rest_days", "back_to_back"
]

MODEL = joblib.load(MODEL_PATH)
ALL_PLAYERS = players.get_players()

# ===============================
# Helpers
# ===============================
def lookup_nba_player_id(name):
    name = name.lower()
    for p in ALL_PLAYERS:
        if p["full_name"].lower() == name:
            return p["id"]
    return None


def load_or_fetch_player_games(player_id, name):
    path = f"{CACHE_DIR}/{name}.csv"

    if os.path.exists(path):
        df = pd.read_csv(path)
        df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
        return df

    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season="2025-26",
        season_type_all_star="Regular Season"
    ).get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df.to_csv(path, index=False)
    return df


def engineer_features(df):
    df = df.sort_values("GAME_DATE")

    df["pts_last5"] = df["PTS"].rolling(5).mean()
    df["pts_last10"] = df["PTS"].rolling(10).mean()
    df["pts_std_5"] = df["PTS"].rolling(5).std()
    df["season_pts_avg"] = df["PTS"].expanding().mean()

    df["min_last5"] = df["MIN"].rolling(5).mean()
    df["min_last10"] = df["MIN"].rolling(10).mean()
    df["minutes_trend"] = df["min_last5"] - df["min_last10"]

    df["fga_last5"] = df["FGA"].rolling(5).mean()
    df["fga_last10"] = df["FGA"].rolling(10).mean()
    df["fga_trend"] = df["fga_last5"] - df["fga_last10"]

    df["home_flag"] = df["MATCHUP"].str.contains("vs").astype(int)
    df["rest_days"] = df["GAME_DATE"].diff().dt.days
    df["back_to_back"] = (df["rest_days"] == 1).astype(int)

    return df.dropna()


# ========================
# GET PRIZEPICK LINES
# ========================
def get_prizepicks():
    url = "https://partner-api.prizepicks.com/projections"
    
    params = {
        "league_id": 7,   # NBA
        "per_page": 250
    }

    r = requests.get(url, params=params)
    if r.status_code != 200:
        print("❌ PrizePicks API Error", r.text)
        return []

    json_data = r.json()
    projections = json_data["data"]
    included = json_data["included"]

    # Lookups
    player_lookup = {}
    game_lookup = {}

    for item in included:
        if item["type"] == "new_player":
            player_lookup[item["id"]] = item["attributes"]["name"]

        if item["type"] == "game":
            game_lookup[item["id"]] = {
                "start_time": item["attributes"].get("start_time"),
                "home_team": item["attributes"].get("home_team_abbreviation"),
                "away_team": item["attributes"].get("away_team_abbreviation"),
            }

    props = []

    for proj in projections:
        attr = proj["attributes"]

        stat = (attr.get("stat_type") or "").lower()
        if stat != "points":
            continue

        if attr.get("odds_type") != "standard":
            continue

        if attr.get("adjusted_odds") is not None:
            continue

        player_id = attr.get("new_player_id")
        if not player_id:
            rel = proj.get("relationships", {})
            player_id = rel.get("new_player", {}).get("data", {}).get("id")

        if not player_id:
            continue

        player_name = player_lookup.get(str(player_id))
        if not player_name:
            continue

        # Game relationship
        rel = proj.get("relationships", {})
        game_id = rel.get("game", {}).get("data", {}).get("id")
        game = game_lookup.get(game_id, {})

        player_team = attr.get("team")
        home = game.get("home_team")
        away = game.get("away_team")

        opponent = None
        if player_team and home and away:
            opponent = away if player_team == home else home

        props.append({
            "player": player_name,
            "line": attr["line_score"],
            "team": player_team,
            "opponent": opponent,
            "game_time": game.get("start_time"),
        })

    print(f"✅ Loaded {len(props)} PrizePicks props")
    return props


# ===============================
# Predict
# ===============================
def make_predictions(model, props):
    picks = []

    for prop in props:
        name = prop["player"]
        line = prop["line"]

        player_id = lookup_nba_player_id(name)
        if not player_id:
            continue

        df = load_or_fetch_player_games(player_id, name)
        if df is None or df.empty:
            continue
        time.sleep(1)

        df = engineer_features(df)
        if df.empty:
            continue

        latest = df.iloc[-1]
        X = pd.DataFrame([latest[FEATURES]], columns=FEATURES)

        pred = model.predict(X)[0]
        diff = pred - line

        # Calculate Confidence Level
        with open(META_PATH) as f:
            meta = json.load(f)

        model_mae = meta["mae"]

        std = max(
            0.7 * latest["pts_std_5"] +
            0.3 * df["PTS"].std(),
            3.0
        )
        effective_std = sqrt(std**2 + model_mae**2)

        z = diff / max(effective_std, 1.0)

        prob_over = norm.cdf(z)

        pick = "OVER" if prob_over >= 0.5 else "UNDER"
        confidence = int(round(prob_over * 100))

        if pick == "UNDER":
            confidence = 100 - confidence
        
        # Make sure confidence is not more than 80
        confidence = min(confidence, 80)

        # Append to picks
        picks.append({
            "player": name,
            "line": line,
            "predicted": round(pred, 1),
            "pick": pick,
            "confidence": confidence,
            "game_time": prop.get("game_time"),
        })

    return picks


if __name__ == "__main__":
    print("🚀 Loading model and fetching PrizePicks...")

    props = get_prizepicks()
    picks = make_predictions(MODEL, props)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(picks, f, indent=2)

    
    print("✅ picks.json generated")
