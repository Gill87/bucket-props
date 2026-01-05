# update_player_cache.py
import os
import time
import pandas as pd
from nba_api.stats.static import players
from nba_api.stats.endpoints import playergamelog
from nba_api.library.http import NBAStatsHTTP
import requests
import random

# NBA API HARDENING (CRITICAL)
# ===============================
NBAStatsHTTP.headers = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
}

# Use ONE persistent session (huge)
session = requests.Session()
NBAStatsHTTP._session = session

# ===============================
# Config
# ===============================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

CACHE_DIR = os.path.join(BASE_DIR, "data", "player_cache")

os.makedirs(CACHE_DIR, exist_ok=True)


SEASON = "2025-26"
TIMEOUT = 35                 # cloud-safe timeout
MIN_SLEEP = 1.8              # seconds
MAX_SLEEP = 3.2              # seconds
MAX_RETRIES = 2              # per player
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
        timeout=TIMEOUT,
    ).get_data_frames()[0]

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
    return df


def update_player_cache(player_name):
    print(f"➡️ Fetching {player_name}", flush=True)

    player_id = PLAYER_LOOKUP.get(player_name)
    if not player_id:
        print(f"⚠️ No NBA ID for {player_name}, skipping", flush=True)
        return

    path = os.path.join(CACHE_DIR, f"{player_name}.csv")
    if not os.path.exists(path):
        print(f"⚠️ Cache missing for {player_name}, skipping", flush=True)
        return

    old_df = pd.read_csv(path)
    old_df["GAME_DATE"] = pd.to_datetime(old_df["GAME_DATE"])
    latest_cached = old_df["GAME_DATE"].max()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            new_df = fetch_player_games(player_id)
            new_df["GAME_DATE"] = pd.to_datetime(new_df["GAME_DATE"])

            # Early exit if no new games
            if new_df["GAME_DATE"].max() <= latest_cached:
                print(f"✅ {player_name} already up to date", flush=True)
                return

            combined = pd.concat([old_df, new_df])

            if "GAME_ID" in combined.columns:
                combined = combined.drop_duplicates(subset=["GAME_ID"])
            else:
                combined = combined.drop_duplicates(subset=["GAME_DATE", "MATCHUP"])

            combined = combined.sort_values("GAME_DATE")
            combined.to_csv(path, index=False)

            print(
                f"🔄 Updated {player_name} (+{len(combined) - len(old_df)} games)",
                flush=True,
            )
            return

        except Exception as e:
            print(
                f"❌ Attempt {attempt}/{MAX_RETRIES} failed for {player_name}: {e}",
                flush=True,
            )
            time.sleep(5)

    print(f"🚫 Giving up on {player_name} this run", flush=True)


# ===============================
# Main
# ===============================
if __name__ == "__main__":
    print("🚀 Updating existing player caches only...", flush=True)

    csv_files = sorted(
        f for f in os.listdir(CACHE_DIR) if f.endswith(".csv")
    )

    if not csv_files:
        print("⚠️ No existing player caches found", flush=True)
        exit(0)

    for file in csv_files:
        player_name = file.replace(".csv", "")
        update_player_cache(player_name)

        # Randomized human-like delay
        sleep_time = random.uniform(MIN_SLEEP, MAX_SLEEP)
        time.sleep(sleep_time)

    print("✅ Player cache refresh complete", flush=True)
