import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from nba_api.live.nba.endpoints import scoreboard, boxscore
except ImportError:
    print("CRITICAL: Please install nba_api (pip install nba_api)")
    sys.exit(1)

from prop_analyzer import config as cfg
from prop_analyzer.data import loader
from prop_analyzer.features import generator
from prop_analyzer.models import registry, inference
from prop_analyzer.utils import text, common

# --- CONFIGURATION ---
# Keywords that trigger the creation of a prop request
PROP_KEYWORDS = [
    "Points", "Rebounds", "Assists", 
    "Pts + Rebs + Asts", "Rebounds + Assists", "Points + Rebounds", "Points + Assists", 
    "Steals + Blocks", "3-Pointers Made", "Fantasy Score", 
    "Turnovers", "Steals", "Blocks", "Double Doubles",
    "Pts + Rebs", "Pts + Asts", "Reb + Ast"
]

# Map text to internal codes
PROP_MAP = {
    "Pts + Rebs + Asts": "PRA", "Rebounds + Assists": "RA",
    "Points + Rebounds": "PR", "Points + Assists": "PA",
    "Steals + Blocks": "STK", "3-Pointers Made": "FG3M",
    "Points": "PTS", "Rebounds": "REB", "Assists": "AST",
    "Turnovers": "TOV", "Steals": "STL", "Blocks": "BLK"
}

# Lines to explicitly ignore
IGNORE_LIST = [
    "Champions", "Drafts", "Live", "Results", "Rankings", "News feed",
    "In-game", "In-game NBA", "Add picks", "Standard", "Flex",
    "Pick a minimum of 2 players", "Champions entry amount", "Add picks"
]

# --- PARSER ---
def parse_live_input_to_df():
    """
    Parses the specific Copy-Paste format from the betting site.
    Structure: Player -> (Line -> Current -> Prop)*
    """
    input_path = cfg.INPUT_DIR / "live_input.txt"
    if not input_path.exists():
        logging.error("No input/live_input.txt found.")
        return pd.DataFrame()

    with open(input_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    clean_rows = []
    current_player = None
    buffer = [] # Stores recent numbers: [Line, Current]

    # Regex for Game Info (e.g. "CLE 26 - IND 25 - 1st Quarter")
    game_info_pattern = re.compile(r'[A-Z]{3}.*\d+.*-.*[A-Z]{3}')

    for line in lines:
        # 1. Filter Noise
        if line in IGNORE_LIST or line.startswith('$'):
            continue
        
        # 2. Skip Game Info lines
        if game_info_pattern.search(line) or (' - ' in line and 'Quarter' in line):
            continue

        # 3. Check if line is a Prop Keyword (The Trigger)
        # Logic: If we see "Points", the previous two items in buffer should be [Line, Current]
        if line in PROP_KEYWORDS:
            if len(buffer) >= 2 and current_player:
                # The Line is usually the item before the Current value
                line_val_str = buffer[-2]
                
                try:
                    line_val = float(line_val_str)
                    prop_code = PROP_MAP.get(line, None) # Normalize to PTS/REB/etc
                    
                    if prop_code: # Only add if it's a supported prop
                        clean_rows.append({
                            'Player': current_player,
                            'Prop': prop_code,
                            'Line': line_val
                        })
                except ValueError:
                    pass # Failed to parse line number
            
            # Reset buffer because we consumed the numbers for this prop
            buffer = []
            continue

        # 4. Check if line is Numeric or a Status Code (Expired/Refreshing)
        # These go into the buffer to be used by the next Keyword
        is_numeric = re.match(r'^\d+(\.\d+)?$', line)
        is_status = line in ["Expired", "Projection refreshing...", "0"]
        
        if is_numeric or is_status:
            buffer.append(line)
            continue

        # 5. Assume Player Name
        # If it's not noise, not game info, not keyword, not number... it's a Player.
        current_player = line
        buffer = [] # Clear buffer for new player

    return pd.DataFrame(clean_rows)

# --- LIVE DATA FUNCTIONS ---
def get_live_games():
    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        game_map = {}
        for g in games:
            # 2=InProgress, 3=Final
            if g['gameStatus'] in [2, 3]: 
                gid = g['gameId']
                h_team = g['homeTeam']['teamTricode']
                a_team = g['awayTeam']['teamTricode']
                game_map[h_team] = {'GameID': gid, 'Opp': a_team, 'IsHome': True}
                game_map[a_team] = {'GameID': gid, 'Opp': h_team, 'IsHome': False}
        return game_map
    except: return {}

def get_live_player_stats(game_id, player_id):
    try:
        box = boxscore.BoxScore(game_id=game_id)
        data = box.game.get_dict()
        all_players = data['homeTeam']['players'] + data['awayTeam']['players']
        
        for p in all_players:
            if p['personId'] == player_id:
                stats = p['statistics']
                min_str = stats.get('minutes', 'PT00M00.00S')
                minutes = 0.0
                if ':' in min_str:
                    m, s = min_str.split(':')
                    minutes = int(m) + (int(s)/60)
                elif 'M' in min_str:
                    match = re.search(r'PT(\d+)M(\d+)', min_str)
                    if match: minutes = int(match.group(1)) + (int(match.group(2))/60)

                return {
                    'MIN': minutes, 'PTS': stats['points'], 'REB': stats['reboundsTotal'],
                    'AST': stats['assists'], 'FG3M': stats['threePointersMade'],
                    'STL': stats['steals'], 'BLK': stats['blocks'], 'TOV': stats['turnovers']
                }
        return None
    except: return None

def calculate_projection(current_val, minutes_played, avg_minutes):
    if minutes_played < 1.0: return current_val 
    if minutes_played >= avg_minutes: return current_val 
    return (current_val / minutes_played) * avg_minutes

# --- MAIN ENGINE ---
def run_live_analysis():
    common.setup_logging(name="live")
    logging.info(">>> Starting Live Prop Analyzer")

    # 1. Parse Input
    requests_df = parse_live_input_to_df()
    if requests_df.empty:
        print("No valid props found in input/live_input.txt")
        return

    # 2. Load Database
    logging.info("Loading database...")
    player_stats, team_stats, _ = loader.load_static_data()
    model_cache = registry.load_model_cache()
    try: dvp_df = pd.read_csv(cfg.DATA_DIR / "master_dvp_stats.csv")
    except: dvp_df = None

    # 3. Pre-load History (Optimization)
    logging.info(f"Processing {len(requests_df)} props...")
    relevant_ids = []
    player_match_cache = {}

    for name in requests_df['Player'].unique():
        p_data = text.fuzzy_match_player(name, player_stats)
        if p_data is not None:
            relevant_ids.append(p_data['PLAYER_ID'])
            player_match_cache[name] = p_data

    if not relevant_ids:
        print("No matching players found in database. Check naming or run ETL.")
        return

    box_scores = loader.load_box_scores(relevant_ids)
    
    # 4. Get Live Games
    live_games_map = get_live_games()
    if not live_games_map:
        print("No active NBA games found via API.")
        # return # Uncomment in production

    results = []

    # 5. Analysis Loop
    for _, row in requests_df.iterrows():
        name = row['Player']
        prop_cat = row['Prop']
        line = float(row['Line'])
        
        p_data = player_match_cache.get(name)
        if p_data is None: continue
            
        pid = p_data['PLAYER_ID']
        team = p_data['TEAM_ABBREVIATION']
        
        # Check Live Status
        game_info = live_games_map.get(team)
        if not game_info: continue
            
        live_stats = get_live_player_stats(game_info['GameID'], pid)
        if not live_stats: continue
            
        # Current Value
        curr_val = 0.0
        if prop_cat in live_stats: curr_val = live_stats[prop_cat]
        elif prop_cat == 'PRA': curr_val = live_stats['PTS'] + live_stats['REB'] + live_stats['AST']
        elif prop_cat == 'PR': curr_val = live_stats['PTS'] + live_stats['REB']
        elif prop_cat == 'PA': curr_val = live_stats['PTS'] + live_stats['AST']
        elif prop_cat == 'RA': curr_val = live_stats['REB'] + live_stats['AST']
        elif prop_cat == 'STK': curr_val = live_stats['STL'] + live_stats['BLK']
            
        # --- HYBRID MODEL ---
        features, _, _ = generator.build_feature_vector(
            p_data, pid, prop_cat, line, team, game_info['Opp'], 
            game_info['IsHome'], datetime.now().strftime('%Y-%m-%d'),
            box_scores, team_stats, pd.DataFrame(), player_stats, dvp_df
        )
        
        pre_game_pred = inference.predict_prop(model_cache, prop_cat, features)
        
        # Baselines
        baseline_val = line
        q20_base, q80_base = line * 0.8, line * 1.2 
        
        if pre_game_pred:
            baseline_val = (pre_game_pred['q20'] + pre_game_pred['q80']) / 2
            q20_base = pre_game_pred['q20']
            q80_base = pre_game_pred['q80']
            
        # Live Projection (Pace)
        minutes = live_stats['MIN']
        avg_min = p_data.get('MIN', 30.0)
        if pd.isna(avg_min): avg_min = 30.0
        
        pace_proj = calculate_projection(curr_val, minutes, avg_min)
        
        # Dynamic Weighting
        game_progress = min(minutes / avg_min, 1.0)
        
        # Blended Outputs
        final_proj = (baseline_val * (1 - game_progress)) + (pace_proj * game_progress)
        final_q20 = (q20_base * (1 - game_progress)) + (pace_proj * 0.9 * game_progress)
        final_q80 = (q80_base * (1 - game_progress)) + (pace_proj * 1.1 * game_progress)
        
        # Determine Rank/Tier
        simulated_prob = 0.5
        if final_q80 > final_q20:
            simulated_prob = (final_proj - final_q20) / (final_q80 - final_q20)
            simulated_prob = max(0.0, min(1.0, simulated_prob))
            
        metrics = inference.determine_tier(line, final_q20, final_q80, simulated_prob)
        
        results.append({
            'Player': name,
            'Prop': prop_cat,
            'Line': line,
            'Curr': int(curr_val),
            'Min': int(minutes),
            'Proj': round(final_proj, 1),
            'Diff': round(final_proj - line, 1),
            'Pick': metrics['Best Pick'],
            'Tier': metrics['Tier'],
            'Edge': metrics['Edge'] 
        })

    # 6. Sort & Print
    if not results:
        print("No matching players found active.")
        return

    df = pd.DataFrame(results)
    
    # Sort: Tier (S->A->B->C) then Edge (Desc)
    tier_order = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
    df['sort_idx'] = df['Tier'].map(tier_order).fillna(99)
    df.sort_values(by=['sort_idx', 'Edge'], ascending=[True, False], inplace=True)

    print("\n" + "="*100)
    print(f"LIVE PROP RANKINGS | {len(df)} Bets Analyzed")
    print("="*100)
    print(f"{'Tier':<6} | {'Player':<20} | {'Prop':<5} | {'Line':>5} | {'Curr':>4} | {'Min':>3} | {'Proj':>5} | {'Diff':>5} | {'Pick':<5}")
    print("-" * 100)

    for _, row in df.head(30).iterrows():
        print(
            f"{row['Tier']:<6} | "
            f"{str(row['Player'])[:20]:<20} | "
            f"{row['Prop']:<5} | "
            f"{row['Line']:>5.1f} | "
            f"{row['Curr']:>4} | "
            f"{row['Min']:>3} | "
            f"{row['Proj']:>5.1f} | "
            f"{row['Diff']:>5.1f} | "
            f"{row['Pick']:<5}"
        )
    print("="*100)

if __name__ == "__main__":
    run_live_analysis()