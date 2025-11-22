import requests
import pandas as pd
import logging
import time
from rapidfuzz import process, fuzz
from prop_analyzer import config as cfg

API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba"

# Map API Market Names to Our System's Prop Categories
MARKET_MAP = {
    "player_points": "PTS",
    "player_rebounds": "REB",
    "player_assists": "AST",
    "player_threes": "FG3M",
    "player_points_rebounds_assists": "PRA",
    "player_points_rebounds": "PR",
    "player_points_assists": "PA",
    "player_rebounds_assists": "RA",
    "player_steals": "STL",
    "player_blocks": "BLK",
    "player_turnovers": "TOV"
}

def get_active_games(api_key):
    """Fetches list of active NBA game IDs."""
    url = f"{API_URL}/events"
    params = {"apiKey": api_key}
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Error fetching games: {e}")
        return []

def fetch_player_props(game_id, api_key):
    """Fetches player props for a specific game."""
    url = f"{API_URL}/events/{game_id}/odds"
    params = {
        "apiKey": api_key,
        "regions": "us",
        "markets": ",".join(cfg.ODDS_MARKETS),
        "oddsFormat": "american",
        "bookmakers": cfg.ODDS_SPORTSBOOKS
    }
    
    try:
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        logging.error(f"Error fetching props for game {game_id}: {e}")
        return None

def parse_odds_response(response_json):
    """Parses the nested JSON into a flat DataFrame."""
    rows = []
    
    if not response_json or 'bookmakers' not in response_json:
        return pd.DataFrame()

    game_date = response_json.get('commence_time', '').split('T')[0]
    
    for book in response_json.get('bookmakers', []):
        book_name = book['title']
        for market in book.get('markets', []):
            prop_cat = MARKET_MAP.get(market['key'])
            if not prop_cat: continue
            
            for outcome in market.get('outcomes', []):
                player_name = outcome['description']
                line = outcome.get('point')
                label = outcome['name'] # 'Over' or 'Under'
                price = outcome['price']
                
                if line is None: continue
                
                rows.append({
                    'Player Name': player_name,
                    'Prop Category': prop_cat,
                    'Prop Line': float(line),
                    'Bet Type': label, # Over/Under
                    'Odds': int(price),
                    'Sportsbook': book_name,
                    'GAME_DATE': game_date
                })

    return pd.DataFrame(rows)

def get_daily_odds():
    """Main function to fetch all available odds for the day."""
    if not cfg.ODDS_API_KEY or cfg.ODDS_API_KEY == "YOUR_API_KEY_HERE":
        logging.warning("No Odds API Key found in config. Skipping odds fetch.")
        return pd.DataFrame()

    logging.info("--- Fetching Live Odds from The Odds API ---")
    games = get_active_games(cfg.ODDS_API_KEY)
    
    if not games:
        logging.info("No active games found.")
        return pd.DataFrame()

    all_odds = []
    logging.info(f"Found {len(games)} games. Fetching props (this consumes quota)...")

    for game in games:
        game_id = game['id']
        logging.info(f"  Fetching: {game['home_team']} vs {game['away_team']}")
        data = fetch_player_props(game_id, cfg.ODDS_API_KEY)
        if data:
            df = parse_odds_response(data)
            if not df.empty:
                all_odds.append(df)
        time.sleep(0.5) # Respect rate limits

    if not all_odds:
        return pd.DataFrame()

    final_df = pd.concat(all_odds, ignore_index=True)
    
    # Group by Player/Prop/Line to get the best odds (or average)
    # For now, we filter to just the "Over" bets since that's usually the primary key
    # But better to keep both and let the analyzer decide
    return final_df