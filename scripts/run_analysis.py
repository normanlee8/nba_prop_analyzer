import sys
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import loader
from prop_analyzer.features import generator
from prop_analyzer.models import registry, inference
from prop_analyzer.utils import common, text

# NBA API for Game Status
try:
    from nba_api.live.nba.endpoints import scoreboard, boxscore
except ImportError:
    print("CRITICAL: Please install nba_api (pip install nba_api)")
    sys.exit(1)

# --- CONFIGURATION ---

# Manually map nicknames/short names to Official DB Names
PROP_NAME_FIX_MAP = {
    'Cam Johnson': 'Cameron Johnson', 
    'Deuce McBride': 'Miles McBride',
    'Herb Jones': 'Herbert Jones', 
    'Ron Holland': 'Ronald Holland II',
    'Lu Dort': 'Luguentz Dort', 
    'Nicolas Claxton': 'Nic Claxton',
    'Kelly Oubre': 'Kelly Oubre Jr.', 
    'Jabari Smith': 'Jabari Smith Jr.',
    'Trey Murphy': 'Trey Murphy III',
    'Tim Hardaway': 'Tim Hardaway Jr.',
    'GG Jackson': 'GG Jackson II'
}

# --- GAME CONTEXT HELPERS ---

def get_daily_game_context():
    """
    Fetches the day's schedule and returns a map:
    Team_Abbrev -> {GameID, Status, Opponent, IsHome, Period, Clock}
    Status Codes: 1=Scheduled, 2=Live, 3=Final
    """
    logging.info("Fetching daily schedule from NBA API...")
    try:
        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        
        context_map = {}
        
        for g in games:
            gid = g['gameId']
            status = g['gameStatus'] # 1: Pre, 2: Live, 3: Final
            period = g.get('period', 0)
            clock = g.get('gameClock', '')
            
            h_team = g['homeTeam']['teamTricode']
            a_team = g['awayTeam']['teamTricode']
            
            # Map Home Team
            context_map[h_team] = {
                'GameID': gid, 'Status': status, 'Opp': a_team, 
                'IsHome': True, 'Period': period, 'Clock': clock
            }
            
            # Map Away Team
            context_map[a_team] = {
                'GameID': gid, 'Status': status, 'Opp': h_team, 
                'IsHome': False, 'Period': period, 'Clock': clock
            }
            
        return context_map
    except Exception as e:
        logging.error(f"Failed to fetch scoreboard: {e}")
        return {}

def get_live_player_box(game_id, player_id):
    """Fetches current stats for a specific player in a live game."""
    try:
        box = boxscore.BoxScore(game_id=game_id)
        data = box.game.get_dict()
        # Check both rosters
        all_players = data['homeTeam']['players'] + data['awayTeam']['players']
        
        for p in all_players:
            if p['personId'] == player_id:
                stats = p['statistics']
                # Parse Minutes "PT24M30.00S" or "24:30"
                min_str = stats.get('minutes', 'PT00M00.00S')
                minutes = 0.0
                if ':' in min_str:
                    m, s = min_str.split(':')
                    minutes = int(m) + (int(s)/60)
                elif 'M' in min_str:
                    match = re.search(r'PT(\d+)M', min_str)
                    if match: minutes = int(match.group(1))
                    match_s = re.search(r'M(\d+)\.', min_str)
                    if match_s: minutes += int(match_s.group(1))/60

                return {
                    'MIN': minutes, 
                    'PTS': stats['points'], 
                    'REB': stats['reboundsTotal'],
                    'AST': stats['assists'], 
                    'FG3M': stats['threePointersMade'],
                    'STL': stats['steals'], 
                    'BLK': stats['blocks'], 
                    'TOV': stats['turnovers']
                }
        return None
    except:
        return None

def calculate_live_projection(current_val, minutes_played, avg_minutes):
    """Projects final stat based on current pace and average rotation."""
    if minutes_played < 1.0: return current_val # Too early to project
    
    # If they've already played their average minutes, assume they are done or close to it
    if minutes_played >= avg_minutes: 
        return current_val 
        
    # Simple linear projection: (Current / Mins) * Avg_Mins
    return (current_val / minutes_played) * avg_minutes

# --- CORE ANALYZER ---

def main():
    common.setup_logging(name="unified_analyzer")
    logging.info(">>> Starting Unified NBA Prop Analyzer (Odds-Free)")

    # 1. Load Static Data
    player_stats, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    model_cache = registry.load_model_cache()
    
    try:
        dvp_df = pd.read_csv(cfg.DATA_DIR / "master_dvp_stats.csv")
    except:
        dvp_df = None

    if player_stats is None or model_cache is None:
        logging.critical("Missing required data. Run 'run_build_db.py' first.")
        return

    # 2. Load Input Props
    if not cfg.PROPS_FILE.exists():
        logging.critical(f"Props file not found: {cfg.PROPS_FILE}")
        return
        
    try:
        # Supporting standard CSV input
        props_df = pd.read_csv(cfg.PROPS_FILE)
        # Ensure minimal columns exist
        req_cols = ['Player Name', 'Prop Category', 'Prop Line']
        if not all(col in props_df.columns for col in req_cols):
            logging.error(f"Input CSV must contain columns: {req_cols}")
            return

        # --- APPLY NAME FIXES HERE ---
        logging.info("Applying name corrections...")
        props_df['Player Name'] = props_df['Player Name'].replace(PROP_NAME_FIX_MAP)
        
    except Exception as e:
        logging.critical(f"Error reading props file: {e}")
        return

    # 3. Build Game Context
    context_map = get_daily_game_context()
    if not context_map:
        logging.warning("Context map empty. Assuming all games are PRE-GAME (Status 1).")

    # 4. Optimize Box Score Loading
    relevant_ids = []
    player_lookup = {}
    
    # Pre-scan players to bulk load data
    for name in props_df['Player Name'].unique():
        p_data = text.fuzzy_match_player(name, player_stats)
        if p_data is not None: 
            relevant_ids.append(p_data['PLAYER_ID'])
            player_lookup[name] = p_data
        else:
            # Log this clearly so user knows exactly who failed even after fixes
            logging.warning(f"Player not found in DB (Check Fix Map): {name}")
            
    box_scores = loader.load_box_scores(relevant_ids)

    results = []
    
    logging.info(f"Analyzing {len(props_df)} props...")

    # 5. Analysis Loop
    for _, row in props_df.iterrows():
        player_name = row['Player Name']
        prop_cat_raw = row['Prop Category']
        line = float(row['Prop Line'])
        
        # Normalize Prop Category
        prop_cat = cfg.MASTER_PROP_MAP.get(prop_cat_raw, prop_cat_raw)
        if prop_cat not in cfg.SUPPORTED_PROPS: 
            continue

        p_data = player_lookup.get(player_name)
        if p_data is None: 
            continue
            
        pid = p_data['PLAYER_ID']
        team = p_data['TEAM_ABBREVIATION']
        
        # --- DETERMINE STATUS ---
        # Default to Pre-Game if team not in context
        game_ctx = context_map.get(team, {'Status': 1, 'GameID': None, 'Opp': 'UNK', 'IsHome': True, 'Period': 0, 'Clock': ''})
        status = game_ctx['Status'] # 1: Pre, 2: Live, 3: Final
        
        if status == 3:
            continue # Skip finalized games

        # --- FETCH LIVE STATS (If Status == 2) ---
        live_stats = None
        curr_val = 0.0
        minutes_played = 0.0
        
        if status == 2 and game_ctx['GameID']:
            live_stats = get_live_player_box(game_ctx['GameID'], pid)
            
            if live_stats:
                minutes_played = live_stats['MIN']
                # Calculate Composite Stats
                if prop_cat in live_stats: curr_val = live_stats[prop_cat]
                elif prop_cat == 'PRA': curr_val = live_stats['PTS'] + live_stats['REB'] + live_stats['AST']
                elif prop_cat == 'PR': curr_val = live_stats['PTS'] + live_stats['REB']
                elif prop_cat == 'PA': curr_val = live_stats['PTS'] + live_stats['AST']
                elif prop_cat == 'RA': curr_val = live_stats['REB'] + live_stats['AST']
                elif prop_cat == 'STK': curr_val = live_stats['STL'] + live_stats['BLK']

        # --- FEATURE GENERATION ---
        # Generate features as if pre-game to get model baseline
        features, _, _ = generator.build_feature_vector(
            p_data, pid, prop_cat, line,
            team, game_ctx['Opp'], 
            game_ctx['IsHome'], datetime.now().strftime('%Y-%m-%d'),
            box_scores, team_stats, vs_opp_df, full_roster_df=player_stats,
            dvp_df=dvp_df 
        )

        # --- PREDICTION ---
        model_out = inference.predict_prop(model_cache, prop_cat, features)
        if not model_out: continue

        # Baseline Projection from Model
        baseline_proj = (model_out['q20'] + model_out['q80']) / 2
        q20 = model_out['q20']
        q80 = model_out['q80']
        
        # --- LIVE ADJUSTMENT ---
        if status == 2 and live_stats:
            avg_min = p_data.get('MIN', 30.0)
            if pd.isna(avg_min): avg_min = 30.0
            
            pace_proj = calculate_live_projection(curr_val, minutes_played, avg_min)
            
            # Weighting based on game progress
            game_progress = min(minutes_played / avg_min, 1.0)
            
            # Blended Projections
            baseline_proj = (baseline_proj * (1 - game_progress)) + (pace_proj * game_progress)
            
            # Tighten confidence intervals as game progresses
            q20 = (q20 * (1 - game_progress)) + (pace_proj * 0.9 * game_progress)
            q80 = (q80 * (1 - game_progress)) + (pace_proj * 1.1 * game_progress)

        # --- RECALCULATE WIN PROBABILITY ---
        # Triangle Distribution Logic based on new Q20/Q80
        if line < q20:
            win_prob = 0.80 + (0.20 * (1 - (line/q20)))
        elif line > q80:
            win_prob = 0.20 * (q80/line)
        else:
            range_width = q80 - q20
            if range_width == 0: range_width = 0.1
            position = (line - q20) / range_width
            win_prob = 0.80 - (0.60 * position)
            
        win_prob = max(0.0, min(1.0, win_prob))

        metrics = inference.determine_tier(line, q20, q80, win_prob)
        
        res = {
            'Status': 'LIVE' if status == 2 else 'PRE',
            'Game': f"{team} vs {game_ctx['Opp']}",
            'Qtr': f"Q{game_ctx['Period']}" if status == 2 else "-",
            'Player': p_data['PLAYER_NAME'],
            'Prop': prop_cat,
            'Line': line,
            'Curr': int(curr_val) if status == 2 else 0,
            'Proj': round(baseline_proj, 1),
            'Edge': round(baseline_proj - line, 1),
            'Win%': metrics['Win_Prob'],
            'Pick': metrics['Best Pick'],
            'Tier': metrics['Tier']
        }
        results.append(res)

    # 6. Report Generation
    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    
    # Sort: Status (Live first), then Tier, then Win%
    tier_order = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
    df['sort_idx'] = df['Tier'].map(tier_order).fillna(99)
    
    df.sort_values(by=['Status', 'sort_idx', 'Win%'], ascending=[True, True, False], inplace=True)

    print("\n" + "="*120)
    print(f"UNIFIED NBA PROP REPORT | {len(df)} Props Analyzed")
    print("="*120)
    print(f"{'Sts':<4} | {'Game':<9} | {'Qtr':<3} | {'Player':<20} | {'Prop':<6} | {'Line':>5} | {'Curr':>4} | {'Proj':>5} | {'Edge':>5} | {'Win%':>6} | {'Pick':<5} | {'Tier':<5}")
    print("-" * 120)

    for _, row in df.head(50).iterrows():
        win_prob_str = f"{row['Win%']:.1%}"
        print(
            f"{row['Status']:<4} | "
            f"{row['Game']:<9} | "
            f"{row['Qtr']:<3} | "
            f"{str(row['Player'])[:20]:<20} | "
            f"{row['Prop']:<6} | "
            f"{row['Line']:>5.1f} | "
            f"{row['Curr']:>4} | "
            f"{row['Proj']:>5.1f} | "
            f"{row['Edge']:>5.1f} | "
            f"{win_prob_str:>6} | "
            f"{row['Pick']:<5} | "
            f"{row['Tier']:<5}"
        )
    print("="*120)
    
    # Save
    df.drop(columns=['sort_idx'], inplace=True)
    df.to_csv(cfg.PROCESSED_OUTPUT, index=False)
    logging.info(f"Saved results to {cfg.PROCESSED_OUTPUT}")

if __name__ == "__main__":
    main()