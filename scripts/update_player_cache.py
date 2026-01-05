# update_player_cache.py
import os
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog

# ===============================
# Config
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(BASE_DIR, "data", "player_cache")

os.makedirs(CACHE_DIR, exist_ok=True)


SEASON = "2025-26"
ALL_PLAYERS = players.get_players()
PLAYER_LOOKUP = {p["full_name"]: p["id"] for p in ALL_PLAYERS}


# ===============================
# Helpers
# ===============================
def fetch_player_games(player_id):
    df = playergamelog.PlayerGameLog(
        player_id=player_id,
        season=SEASON,
        season_type_all_star="Regular Season",
        timeout=30,
    ).get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def update_player_cache(player_name):
    print(f"➡️ Fetching {player_name}", flush=True)
    player_id = PLAYER_LOOKUP.get(player_name)
    if not player_id:
        print(f"⚠️ Could not find NBA ID for {player_name}, skipping")
        return

    path = f"{CACHE_DIR}/{player_name}.csv"

    try:
        new_df = fetch_player_games(player_id)
        time.sleep(0.6)
    except Exception as e:
        print(f"❌ Failed fetching {player_name}: {e}")
        return

    old_df = pd.read_csv(path)
    old_df["GAME_DATE"] = pd.to_datetime(old_df["GAME_DATE"])

    combined = pd.concat([old_df, new_df])

    # Robust duplicate handling
    if "GAME_ID" in combined.columns:
        combined = combined.drop_duplicates(subset=["GAME_ID"])
    else:
        combined = combined.drop_duplicates(subset=["GAME_DATE", "MATCHUP"])

    combined = combined.sort_values("GAME_DATE")


    if len(combined) > len(old_df):
        combined.to_csv(path, index=False)
        print(f"🔄 Updated {player_name} (+{len(combined) - len(old_df)} games)")
    else:
        print(f"✅ {player_name} already up to date")


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    print("🚀 Updating existing player caches only...")

    csv_files = [
        f for f in os.listdir(CACHE_DIR)
        if f.endswith(".csv")
    ]

    if not csv_files:
        print("⚠️ No existing player caches found")
        exit(0)

    for file in csv_files:
        player_name = file.replace(".csv", "")
        update_player_cache(player_name)

    print("✅ Player cache refresh complete")
