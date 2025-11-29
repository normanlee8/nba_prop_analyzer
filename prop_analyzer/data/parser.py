import csv
import re
import datetime
import logging
import pandas as pd
from pathlib import Path
from prop_analyzer import config as cfg
from nba_api.stats.endpoints import scoreboardv2

# Expanded map to catch common abbreviations
DAYS_MAP = {
    'Mon': 0, 'Tue': 1, 'Wed': 2, 'Thu': 3,
    'Fri': 4, 'Sat': 5, 'Sun': 6,
    'MON': 0, 'TUE': 1, 'WED': 2, 'THU': 3,
    'FRI': 4, 'SAT': 5, 'SUN': 6
}

class SmartDateDetector:
    """
    1. Checks historical box scores (Past 3 Days) to see if it's a recent game (Backtesting).
    2. If not found, checks the NBA Schedule for Today and Tomorrow (Live/Upcoming).
    """
    def __init__(self, lookback_days=3):
        self.history_map = {}
        self.lookback_days = lookback_days
        self.schedule_cache = {}  # Cache schedule to avoid spamming API
        self._load_history()

    def _load_history(self):
        """Loads the last N days of matchups."""
        files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_BOX_SCORES_PATTERN))
        if not files: return

        try:
            dfs = []
            for f in files:
                try:
                    # Only load what we need
                    d = pd.read_csv(f, usecols=['TEAM_ABBREVIATION', 'OPPONENT_ABBREV', 'GAME_DATE'])
                    dfs.append(d)
                except: continue
            
            if not dfs: return

            df = pd.concat(dfs, ignore_index=True)
            df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
            
            # Dynamic cutoff based on lookback_days
            cutoff = pd.Timestamp.now() - pd.Timedelta(days=self.lookback_days)
            recent = df[df['GAME_DATE'] >= cutoff].sort_values('GAME_DATE', ascending=True)
            
            for _, row in recent.iterrows():
                t1 = row['TEAM_ABBREVIATION']
                t2 = row['OPPONENT_ABBREV']
                date_str = row['GAME_DATE'].strftime('%Y-%m-%d')
                self.history_map[(t1, t2)] = date_str
                self.history_map[(t2, t1)] = date_str
                
        except Exception as e:
            logging.warning(f"Failed to load history: {e}")

    def _check_nba_schedule(self, team, opponent, check_date):
        """Checks if a specific matchup is scheduled for a specific date string (YYYY-MM-DD)."""
        
        # 1. Check Cache first
        if check_date in self.schedule_cache:
            games = self.schedule_cache[check_date]
        else:
            # 2. Fetch from API if not cached
            try:
                # Reduced timeout to fail faster if NBA stats is laggy
                board = scoreboardv2.ScoreboardV2(game_date=check_date, timeout=3)
                games_df = board.game_header.get_data_frame()
                
                # Cache the result (even if empty) to prevent re-fetching
                self.schedule_cache[check_date] = games_df
                games = games_df
                
            except Exception as e:
                logging.warning(f"Could not fetch schedule for {check_date}: {e}")
                # CRITICAL FIX: Cache the failure (empty df) so we don't retry 50 times
                self.schedule_cache[check_date] = pd.DataFrame()
                return False

        # 3. Check if our teams match any game in the cached dataframe
        if games is None or games.empty:
            return False
            
        # Normalize inputs
        t1, t2 = team.upper(), opponent.upper()
        
        # Iterate rows to find match
        for _, row in games.iterrows():
            # CRITICAL FIX: Use .get() to avoid KeyError if API format changes
            home_code_raw = row.get('HOME_TEAM_EST_TO', '')
            away_code_raw = row.get('VISITOR_TEAM_EST_TO', '')
            
            if not isinstance(home_code_raw, str) or not isinstance(away_code_raw, str):
                continue

            home_code = home_code_raw.replace('NOP', 'NO').upper()
            away_code = away_code_raw.replace('NOP', 'NO').upper()
            
            # Simple check (Order doesn't matter for "is playing")
            if (t1 == home_code and t2 == away_code) or (t1 == away_code and t2 == home_code):
                return True
                
        return False

    def find_date(self, team, opponent):
        """
        Logic:
        1. Check History (Last 3 Days).
        2. Check Schedule (Today).
        3. Check Schedule (Tomorrow).
        4. Fallback -> Today.
        """
        # 1. Check History (Backtest)
        if (team, opponent) in self.history_map:
            return self.history_map[(team, opponent)]
        
        today = datetime.datetime.now()
        tomorrow = today + datetime.timedelta(days=1)
        
        str_today = today.strftime("%Y-%m-%d")
        str_tomorrow = tomorrow.strftime("%Y-%m-%d")

        # 2. Check Today's Schedule
        if self._check_nba_schedule(team, opponent, str_today):
            return str_today
            
        # 3. Check Tomorrow's Schedule
        if self._check_nba_schedule(team, opponent, str_tomorrow):
            return str_tomorrow

        # 4. Fallback (Default to Today if unknown)
        return str_today

def clean_prop_line(text):
    """Robustly extracts a numeric value from a line string."""
    s = text.strip().upper()
    s = s.replace(',', '.') 
    s = re.sub(r'^[OU]\s+', '', s)
    s = s.replace('OVER', '').replace('UNDER', '').strip()
    try:
        val = float(s)
        return str(val)
    except ValueError:
        return None

def parse_matchup(matchup_line):
    """Extracts Team abbreviations from a matchup line."""
    # Normalize inputs to @ for detection, but output as vs.
    line = matchup_line.replace(' vs ', ' @ ').replace(' vs. ', ' @ ').replace('-', ' @ ')
    match = re.search(r'\b([A-Z]{3})\s*@\s*([A-Z]{3})\b', line)
    
    if match:
        team1 = match.group(1)
        team2 = match.group(2)
        # Output as "vs." instead of "@"
        full_matchup_string = f"{team1} vs. {team2}"
        return team1, team2, full_matchup_string
    return None, None, None

def parse_text_to_csv(input_path=None, output_path=None):
    if input_path is None: input_path = cfg.INPUT_PROPS_TXT
    if output_path is None: output_path = cfg.PROPS_FILE
    
    if not input_path.exists():
        logging.error(f"Input file not found: {input_path}")
        return

    # Initialize Smart Detector
    date_detector = SmartDateDetector(lookback_days=3)
    
    current_player = None
    current_team = None
    current_opponent = None
    current_matchup = None
    current_game_date = None 
    
    data_to_write = [] 

    try:
        with open(input_path, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()

        for line in lines:
            line = line.strip()
            if not line: continue

            # 1. Check for Time/Matchup Line (Context Switch)
            # Regex looks for "LAL @ BOS" or times
            # We prioritize finding the teams
            t1, t2, full_matchup = parse_matchup(line)
            
            # If we found a valid matchup line (e.g. "7:00 PM LAL @ BOS")
            if full_matchup:
                current_matchup = full_matchup
                current_team = t1
                current_opponent = t2
                
                # --- SMART DATE LOOKUP ---
                # Check if this exact matchup happened recently
                current_game_date = date_detector.find_date(t1, t2)
                continue 

            # 2. Check for Prop Line Value
            cleaned_val = clean_prop_line(line)
            if cleaned_val:
                prop_line_value = cleaned_val
                continue

            # 3. Check for Category
            if 'prop_line_value' in locals() and prop_line_value is not None:
                prop_category_str = line
                prop_category_std = cfg.MASTER_PROP_MAP.get(prop_category_str, None)
                
                # Auto-fix common variations
                if not prop_category_std:
                    for k, v in cfg.MASTER_PROP_MAP.items():
                        if k.lower() == prop_category_str.lower():
                            prop_category_std = v
                            break
                
                if prop_category_std and current_player and current_matchup:
                    data_to_write.append([
                        current_player, current_team, current_opponent,
                        current_matchup, prop_category_std, prop_line_value,
                        current_game_date 
                    ])
                
                prop_line_value = None 
                continue

            # 4. Fallback: Player Name
            current_player = line
            prop_line_value = None 
            continue

        if not data_to_write:
            logging.warning("No valid props parsed. Check input format.")
            return

        header = ['Player Name', 'Team', 'Opponent', 'Matchup', 'Prop Category', 'Prop Line', 'GAME_DATE']

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', newline='', encoding='utf-8') as f_out:
            writer = csv.writer(f_out)
            writer.writerow(header)
            writer.writerows(data_to_write)
            
        logging.info(f"Successfully converted props to {output_path} ({len(data_to_write)} rows)")
        
        # Save timestamped record
        now_ts = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        rec_path = cfg.INPUT_DIR / "records" / f"{now_ts}.csv"
        rec_path.parent.mkdir(parents=True, exist_ok=True)
        with open(rec_path, 'w', newline='', encoding='utf-8') as f_rec:
            writer = csv.writer(f_rec)
            writer.writerow(header)
            writer.writerows(data_to_write)

    except Exception as e:
        logging.error(f"Error parsing props: {e}", exc_info=True)