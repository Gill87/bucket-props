# train_model.py
import os
import pandas as pd
import joblib
from datetime import datetime, timezone
import json

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from nba_api.stats.endpoints import playergamelog, commonallplayers

# ===============================
# Config
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data", "training")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

TRAINING_SEASONS = ["2025-26", "2024-25", "2023-24"]

# ===============================
# Data Collection
# ===============================
def get_active_players(limit=50):
    players = commonallplayers.CommonAllPlayers(is_only_current_season=0).get_data_frames()[0]
    players = players[players["ROSTERSTATUS"] == 1]
    players = players.sample(frac=1, random_state=42)     # random selection of players
    return players["PERSON_ID"].tolist()[:limit]

def build_training_dataset():
    all_data = []
    player_ids = get_active_players()

    for pid in player_ids:
        for season in TRAINING_SEASONS:
            try:
                df = playergamelog.PlayerGameLog(
                    player_id=pid,
                    season=season,
                    season_type_all_star="Regular Season"
                ).get_data_frames()[0]

                if df.empty:
                    continue

                df["PLAYER_ID"] = pid
                df["SEASON"] = season
                all_data.append(df)

            except Exception:
                continue

    return pd.concat(all_data, ignore_index=True)


# ===============================
# Feature Engineering
# ===============================
def engineer_features(df):
    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

    def build(player_df):
        player_df = player_df.copy()

        player_df["pts_last5"] = player_df["PTS"].rolling(5).mean()
        player_df["pts_last10"] = player_df["PTS"].rolling(10).mean()
        player_df["pts_std_5"] = player_df["PTS"].rolling(5).std()
        player_df["season_pts_avg"] = player_df.groupby("SEASON")["PTS"].transform("mean")

        player_df["min_last5"] = player_df["MIN"].rolling(5).mean()
        player_df["min_last10"] = player_df["MIN"].rolling(10).mean()
        player_df["minutes_trend"] = player_df["min_last5"] - player_df["min_last10"]

        player_df["fga_last5"] = player_df["FGA"].rolling(5).mean()
        player_df["fga_last10"] = player_df["FGA"].rolling(10).mean()
        player_df["fga_trend"] = player_df["fga_last5"] - player_df["fga_last10"]

        player_df["home_flag"] = player_df["MATCHUP"].str.contains("vs").astype(int)
        player_df["rest_days"] = player_df["GAME_DATE"].diff().dt.days
        player_df["back_to_back"] = (player_df["rest_days"] == 1).astype(int)

        return player_df.dropna()

    return df.groupby("PLAYER_ID").apply(build, include_groups=False).reset_index(drop=True)


# ===============================
# Train
# ===============================
def train():
    print("📊 Building training dataset...")
    df = build_training_dataset()
    df = engineer_features(df)

    df.to_csv(f"{DATA_DIR}/training_data.csv", index=False)

    FEATURES = [
        "pts_last5", "pts_last10", "pts_std_5", "season_pts_avg",
        "min_last5", "min_last10", "minutes_trend",
        "fga_last5", "fga_last10", "fga_trend",
        "home_flag", "rest_days", "back_to_back"
    ]

    X = df[FEATURES]
    y = df["PTS"]

    split = int(len(df) * 0.8)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.08,
        max_depth=3,
        random_state=42
    )

    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    print("✅ MAE:", mae)

    joblib.dump(model, f"{MODEL_DIR}/nba_points_model.pkl")
    print("💾 Model saved")

    # Save metadata
    meta = {
        "mae": float(mae),
        "model_type": "GradientBoostingRegressor",
        "n_estimators": 200,
        "learning_rate": 0.08,
        "max_depth": 3,
        "features": FEATURES,
        "train_seasons": TRAINING_SEASONS,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "trained_at": datetime.now(timezone.utc).isoformat()
    }

    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print("💾 Model + metadata saved")


if __name__ == "__main__":
    train()