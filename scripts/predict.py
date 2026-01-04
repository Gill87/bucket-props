# predict.py
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
        "per_page": 250   # returns full slate
    }

    r = requests.get(url, params=params)

    if r.status_code != 200:
        print("❌ PrizePicks API Error", r.text)
        return []

    json_data = r.json()

    projections = json_data["data"]
    included = json_data["included"]

    # Build player lookup
    player_lookup = {}
    for item in included:
        if item["type"] == "new_player":
            player_lookup[item["id"]] = item["attributes"]["name"]

    props = []

    # Get the props
    for proj in projections:
        attr = proj["attributes"]

        stat = (attr.get("stat_type") or attr.get("display_stat") or "").lower()

        if stat not in ("points",):
            continue
        
        player_id = attr.get("new_player_id")

        # MAIN BOARD FILTER (this is the key)
        if attr.get("odds_type") != "standard":
            continue

        if attr.get("adjusted_odds") is not None:
            continue

        if not player_id:
            rel = proj.get("relationships", {})
            new_player = rel.get("new_player", {}).get("data", {})
            player_id = new_player.get("id")

        if not player_id:
            print("⚠️ Missing player ID for projection, skipping")
            continue

        
        player_name = player_lookup.get(str(player_id), "Unknown Player")
        display = (
            attr.get("display_stat")
            or attr.get("stat_type")
            or attr.get("label")
            or "unknown"
        )
        props.append({
            "player": player_name,
            "line": attr["line_score"],
            "stat": attr["stat_type"],       # "points", "rebounds", etc.
            "display": display, # "PTS", "REB"
            "team": attr.get("team", None),
            "id": proj["id"]
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
        time.sleep(0.7)

        df = engineer_features(df)
        if df.empty:
            continue

        latest = df.iloc[-1]
        X = pd.DataFrame([latest[FEATURES]], columns=FEATURES)

        pred = model.predict(X)[0]
        diff = pred - line

        z = diff / max(latest["pts_std_5"], 1.0)
        prob_over = norm.cdf(z)

        pick = "OVER" if prob_over >= 0.5 else "UNDER"
        confidence = int(round(prob_over * 100))

        if pick == "UNDER":
            confidence = 100 - confidence
        
        # Make sure confidence is not 100
        if(confidence == 100):
            confidence = 99

        picks.append({
            "player": name,
            "line": line,
            "predicted": round(pred, 1),
            "pick": pick,
            "confidence": confidence,
        })

    return picks


if __name__ == "__main__":
    print("🚀 Loading model and fetching PrizePicks...")

    props = get_prizepicks()
    picks = make_predictions(MODEL, props)

    with open(OUTPUT_PATH, "w") as f:
        json.dump(picks, f, indent=2)

    
    print("✅ picks.json generated")
