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

# --- LIVE DATA MANAGER ---

class LiveGameManager:
    """
    Centralizes all interactions with the NBA Live API.
    """
    def __init__(self):
        self.board = None
        self.games = {}
        self.context_map = {}

    def fetch_daily_context(self):
        logging.info("Fetching daily schedule from NBA API...")
        try:
            self.board = scoreboard.ScoreBoard()
            self.games = self.board.games.get_dict()
            self.context_map = {}
            
            for g in self.games:
                gid = g['gameId']
                status = g['gameStatus'] # 1: Pre, 2: Live, 3: Final
                period = g.get('period', 0)
                clock = g.get('gameClock', '')
                
                h_team = g['homeTeam']['teamTricode']
                a_team = g['awayTeam']['teamTricode']
                
                # Map Home Team
                self.context_map[h_team] = {
                    'GameID': gid, 'Status': status, 'Opp': a_team, 
                    'IsHome': True, 'Period': period, 'Clock': clock
                }
                
                # Map Away Team
                self.context_map[a_team] = {
                    'GameID': gid, 'Status': status, 'Opp': h_team, 
                    'IsHome': False, 'Period': period, 'Clock': clock
                }
            return self.context_map
        except Exception as e:
            logging.error(f"Failed to fetch scoreboard: {e}")
            return {}

    def get_live_player_stats(self, game_id, player_id):
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

def calculate_blended_live_projection(baseline_proj, current_val, minutes_played, avg_minutes):
    """
    Uses Bayesian Decay to blend Pre-Game Projection with Live Pace.
    Prevents wild swings early in the game.
    """
    if minutes_played <= 0: return baseline_proj
    
    # Cap minutes to avoid division by zero or negative remaining
    avg_minutes = max(avg_minutes, 15.0) 
    minutes_played = min(minutes_played, avg_minutes)
    
    # Calculate Game Progress (0.0 to 1.0)
    progress = minutes_played / avg_minutes
    
    # Live Pace (Extrapolated)
    live_pace = (current_val / minutes_played) * avg_minutes
    
    # Weighting: Trust baseline early, trust live pace late
    # We use a sigmoid-like weight or simple linear weight
    # Let's use linear for simplicity but start trusting live after 5 mins
    if minutes_played < 5.0:
        weight_live = 0.1
    else:
        weight_live = progress
        
    blended = (baseline_proj * (1 - weight_live)) + (live_pace * weight_live)
    return blended

# --- MAIN ANALYZER ---

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
        props_df = pd.read_csv(cfg.PROPS_FILE)
        req_cols = ['Player Name', 'Prop Category', 'Prop Line']
        if not all(col in props_df.columns for col in req_cols):
            logging.error(f"Input CSV must contain columns: {req_cols}")
            return

        logging.info("Applying name corrections...")
        props_df['Player Name'] = props_df['Player Name'].replace(PROP_NAME_FIX_MAP)
        
    except Exception as e:
        logging.critical(f"Error reading props file: {e}")
        return

    # 3. Init Live Manager & Context
    live_manager = LiveGameManager()
    context_map = live_manager.fetch_daily_context()
    
    if not context_map:
        logging.warning("Context map empty. Assuming all games are PRE-GAME.")

    # 4. Optimize Box Score Loading
    relevant_ids = []
    player_lookup = {}
    
    for name in props_df['Player Name'].unique():
        p_data = text.fuzzy_match_player(name, player_stats)
        if p_data is not None: 
            relevant_ids.append(p_data['PLAYER_ID'])
            player_lookup[name] = p_data
        else:
            logging.warning(f"Player not found in DB: {name}")
            
    box_scores = loader.load_box_scores(relevant_ids)

    results = []
    
    logging.info(f"Analyzing {len(props_df)} props...")

    # 5. Analysis Loop
    for _, row in props_df.iterrows():
        player_name = row['Player Name']
        prop_cat_raw = row['Prop Category']
        line = float(row['Prop Line'])
        
        prop_cat = cfg.MASTER_PROP_MAP.get(prop_cat_raw, prop_cat_raw)
        if prop_cat not in cfg.SUPPORTED_PROPS: 
            continue

        p_data = player_lookup.get(player_name)
        if p_data is None: 
            continue
            
        pid = p_data['PLAYER_ID']
        team = p_data['TEAM_ABBREVIATION']
        
        # --- DETERMINE STATUS ---
        game_ctx = context_map.get(team, {'Status': 1, 'GameID': None, 'Opp': 'UNK', 'IsHome': True, 'Period': 0, 'Clock': ''})
        status = game_ctx['Status'] # 1: Pre, 2: Live, 3: Final
        
        if status == 3: continue # Skip finalized

        # --- FETCH LIVE STATS ---
        live_stats = None
        curr_val = 0.0
        minutes_played = 0.0
        
        if status == 2 and game_ctx['GameID']:
            live_stats = live_manager.get_live_player_stats(game_ctx['GameID'], pid)
            if live_stats:
                minutes_played = live_stats['MIN']
                # Composite logic
                if prop_cat in live_stats: curr_val = live_stats[prop_cat]
                elif prop_cat == 'PRA': curr_val = live_stats['PTS'] + live_stats['REB'] + live_stats['AST']
                elif prop_cat == 'PR': curr_val = live_stats['PTS'] + live_stats['REB']
                elif prop_cat == 'PA': curr_val = live_stats['PTS'] + live_stats['AST']
                elif prop_cat == 'RA': curr_val = live_stats['REB'] + live_stats['AST']
                elif prop_cat == 'STK': curr_val = live_stats['STL'] + live_stats['BLK']

        # --- FEATURE GENERATION ---
        features, _, _ = generator.build_feature_vector(
            p_data, pid, prop_cat, line,
            team, game_ctx['Opp'], 
            game_ctx['IsHome'], datetime.now().strftime('%Y-%m-%d'),
            box_scores, team_stats, vs_opp_df, full_roster_df=player_stats,
            dvp_df=dvp_df 
        )

        # --- PREDICTION ---
        # Inference now returns 'prob_over' from the CLASSIFIER
        model_out = inference.predict_prop(model_cache, prop_cat, features)
        if not model_out: continue

        baseline_proj = (model_out['q20'] + model_out['q80']) / 2
        
        # --- LIVE ADJUSTMENT ---
        final_proj = baseline_proj
        if status == 2 and live_stats:
            avg_min = p_data.get('MIN', 30.0)
            if pd.isna(avg_min): avg_min = 30.0
            
            # Use blended projection instead of linear extrapolation
            final_proj = calculate_blended_live_projection(baseline_proj, curr_val, minutes_played, avg_min)

        # --- TIER DETERMINATION ---
        # We pass the Raw Model Probability into determine_tier
        # The logic inside determine_tier handles the "Divergence" check
        metrics = inference.determine_tier(line, model_out['q20'], model_out['q80'], model_out['prob_over'])
        
        # Visual flag for divergent models (Classifier says Over, Regression says Under)
        tier_display = metrics['Tier']
        if metrics.get('Is_Divergent', False):
            tier_display += "*"

        res = {
            'Status': 'LIVE' if status == 2 else 'PRE',
            'Game': f"{team} vs {game_ctx['Opp']}",
            'Qtr': f"Q{game_ctx['Period']}" if status == 2 else "-",
            'Player': p_data['PLAYER_NAME'],
            'Prop': prop_cat,
            'Line': line,
            'Curr': int(curr_val) if status == 2 else 0,
            'Proj': round(final_proj, 1),
            'Edge': round(final_proj - line, 1),
            'Win%': metrics['Win_Prob'],
            'Pick': metrics['Best Pick'],
            'Tier': tier_display
        }
        results.append(res)

    # 6. Report Generation
    if not results:
        print("No results generated.")
        return

    df = pd.DataFrame(results)
    
    # Clean sorting
    def sort_tier(t):
        t = t.replace('*', '') # Ignore divergence flag for sorting
        if t == 'S Tier': return 0
        if t == 'A Tier': return 1
        if t == 'B Tier': return 2
        return 3
        
    df['sort_idx'] = df['Tier'].apply(sort_tier)
    df.sort_values(by=['Status', 'sort_idx', 'Win%'], ascending=[True, True, False], inplace=True)

    print("\n" + "="*120)
    print(f"UNIFIED NBA PROP REPORT | {len(df)} Props Analyzed")
    print("="*120)
    print(f"{'Sts':<4} | {'Game':<9} | {'Qtr':<3} | {'Player':<20} | {'Prop':<6} | {'Line':>5} | {'Curr':>4} | {'Proj':>5} | {'Edge':>5} | {'Win%':>6} | {'Pick':<5} | {'Tier':<6}")
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
            f"{row['Tier']:<6}"
        )
    print("="*120)
    print("* = Divergent Model Signal (Lower Confidence)")
    
    # Save
    df.drop(columns=['sort_idx'], inplace=True)
    df.to_csv(cfg.PROCESSED_OUTPUT, index=False)
    logging.info(f"Saved results to {cfg.PROCESSED_OUTPUT}")

if __name__ == "__main__":
    main()