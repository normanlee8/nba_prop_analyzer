import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from rapidfuzz import process, fuzz
from unidecode import unidecode
import warnings

# Import config
from prop_analyzer import config as cfg

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=pd.errors.DtypeWarning)
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

TEAM_NAME_MAP = {
    "Atlanta": "ATL", "Atlanta Hawks": "ATL",
    "Boston": "BOS", "Boston Celtics": "BOS",
    "Brooklyn": "BKN", "Brooklyn Nets": "BKN",
    "Charlotte": "CHA", "Charlotte Hornets": "CHA",
    "Chicago": "CHI", "Chicago Bulls": "CHI",
    "Cleveland": "CLE", "Cleveland Cavaliers": "CLE",
    "Dallas": "DAL", "Dallas Mavericks": "DAL",
    "Denver": "DEN", "Denver Nuggets": "DEN",
    "Detroit": "DET", "Detroit Pistons": "DET",
    "Golden State": "GSW", "Golden State Warriors": "GSW",
    "Houston": "HOU", "Houston Rockets": "HOU",
    "Indiana": "IND", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles Lakers": "LAL",
    "Memphis": "MEM", "Memphis Grizzlies": "MEM",
    "Miami": "MIA", "Miami Heat": "MIA",
    "Milwaukee": "MIL", "Milwaukee Bucks": "MIL",
    "Minnesota": "MIN", "Minnesota Timberwolves": "MIN",
    "New Orleans": "NOP", "New Orleans Pelicans": "NOP",
    "New York": "NYK", "New York Knicks": "NYK",
    "Okla City": "OKC", "Oklahoma City Thunder": "OKC",
    "Orlando": "ORL", "Orlando Magic": "ORL",
    "Philadelphia": "PHI", "Philadelphia 76ers": "PHI",
    "Phoenix": "PHX", "Phoenix Suns": "PHX",
    "Portland": "POR", "Portland Trail Blazers": "POR",
    "Sacramento": "SAC", "Sacramento Kings": "SAC",
    "San Antonio": "SAS", "San Antonio Spurs": "SAS",
    "Toronto": "TOR", "Toronto Raptors": "TOR",
    "Utah": "UTA", "Utah Jazz": "UTA",
    "Washington": "WAS", "Washington Wizards": "WAS",
}

PLAYER_STAT_PREFIX_MAP = {
    'HOME': 'Home',
    'AWAY': 'Away:Road',
    'L5': 'Last 5 Games'
}

BBREF_COLUMN_MAP = {
    'G': 'SEASON_G',
    'PTS': 'SEASON_PTS',
    'TRB': 'SEASON_TRB', 
    'AST': 'SEASON_AST',
    'STL': 'SEASON_STL',
    'BLK': 'SEASON_BLK',
    '3P': 'SEASON_FG3M'  
}

def load_clean_csv(filepath, required_cols):
    if not filepath.exists():
        logging.warning(f"File not found: {filepath.name}. Skipping.")
        return None
            
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.empty: return None
            
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logging.warning(f"File {filepath.name} is missing cols: {missing}.")
        
        return df
    except Exception as e:
        logging.error(f"Error loading {filepath.name}: {e}")
        return None

def get_metric_from_filename(filename, prefix="NBA Team "):
    if not filename.startswith(prefix) or not filename.endswith(".csv"):
        return None
    return filename[len(prefix):-len(".csv")]

def sniff_file_type(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            line = f.readline()
            if 'Team' in line and '20' in line: return 'teamrankings'
            if 'TEAM_ID' in line and 'TEAM_NAME' in line: return 'nba_api'
        return None 
    except: return None

def create_player_id_map(data_dir):
    logging.info("Creating Player ID Map...")
    player_map_files = {
        "Home": data_dir / "NBA Player Stats Home.csv",
        "Away": data_dir / "NBA Player Stats Away:Road.csv",
        "L5": data_dir / "NBA Player Stats Last 5 Games.csv"
    }
    
    required_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION']
    all_player_dfs = []
    
    for key, filepath in player_map_files.items():
        df = load_clean_csv(filepath, required_cols=required_cols)
        if df is not None:
            all_player_dfs.append(df[required_cols])
    
    if not all_player_dfs:
        logging.critical("CRITICAL: No valid player stat files found.")
        return None
        
    player_map_df = pd.concat(all_player_dfs)
    player_map_df.drop_duplicates(subset=['PLAYER_ID'], inplace=True)
    
    player_map_df['Player_Clean'] = player_map_df['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).lower().strip())
    return player_map_df

def process_master_player_stats(data_dir, player_id_map, output_dir):
    logging.info("--- Starting: process_master_player_stats ---")
    try:
        api_player_stats = []
        for file_prefix, friendly_name in PLAYER_STAT_PREFIX_MAP.items():
            filepath = data_dir / f"NBA Player Stats {friendly_name}.csv"
            df = load_clean_csv(filepath, required_cols=['PLAYER_ID', 'PLAYER_NAME'])
            if df is not None:
                df = df.add_prefix(f"{file_prefix}_")
                df.rename(columns={f"{file_prefix}_PLAYER_ID": "PLAYER_ID", f"{file_prefix}_PLAYER_NAME": "PLAYER_NAME"}, inplace=True)
                api_player_stats.append(df)

        if not api_player_stats: return
            
        master_player_df = api_player_stats[0]
        for df in api_player_stats[1:]:
            master_player_df = pd.merge(master_player_df, df, on=["PLAYER_ID", "PLAYER_NAME"], how="outer")
            
        # Load Quarter Stats
        for q in range(1, 5):
            filepath = data_dir / f"NBA Player Q{q}.csv"
            df_q = load_clean_csv(filepath, required_cols=['PLAYER_ID', 'PTS', 'MIN'])
            if df_q is not None:
                cols_to_norm = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS']
                cols_to_norm = [c for c in cols_to_norm if c in df_q.columns]
                
                if 'GP' in df_q.columns:
                    df_q = df_q[df_q['GP'] > 0].copy()
                    for col in cols_to_norm:
                        df_q[col] = (df_q[col] / df_q['GP']).round(2)

                df_q = df_q.add_prefix(f"Q{q}_")
                df_q.rename(columns={f"Q{q}_PLAYER_ID": "PLAYER_ID"}, inplace=True)
                master_player_df = pd.merge(master_player_df, df_q, on="PLAYER_ID", how="left")

        # Bball-Ref Stats
        bball_ref_df = load_clean_csv(data_dir / "NBA Player Per Game Averages.csv", required_cols=['Player', 'PTS'])
        if bball_ref_df is not None:
            bball_ref_df['Player_Clean'] = bball_ref_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
            bball_ref_df = bball_ref_df.rename(columns=BBREF_COLUMN_MAP)
            
            id_map_clean = player_id_map[['PLAYER_ID', 'Player_Clean']].drop_duplicates(subset=['Player_Clean'])
            name_to_id = id_map_clean.set_index('Player_Clean')['PLAYER_ID'].to_dict()
            
            def find_match(name):
                if not name: return None
                match = process.extractOne(name, name_to_id.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=90)
                return name_to_id.get(match[0]) if match else None

            bball_ref_df['PLAYER_ID'] = bball_ref_df['Player_Clean'].apply(find_match)
            bball_ref_df = bball_ref_df[bball_ref_df['PLAYER_ID'].notna()]
            bball_ref_df.drop_duplicates(subset=['PLAYER_ID'], inplace=True)
            
            season_cols = ['PLAYER_ID', 'Pos', 'SEASON_G', 'SEASON_PTS', 'SEASON_TRB', 'SEASON_AST', 'SEASON_STL', 'SEASON_BLK', 'SEASON_FG3M']
            cols_exist = [col for col in season_cols if col in bball_ref_df.columns]
            master_player_df = pd.merge(master_player_df, bball_ref_df[cols_exist], on="PLAYER_ID", how="left")
            
            # Advanced Stats
            adv_df = load_clean_csv(data_dir / "NBA Player Advanced Stats.csv", required_cols=['Player', 'USG%'])
            if adv_df is not None:
                adv_df['Player_Clean'] = adv_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                adv_df['PLAYER_ID'] = adv_df['Player_Clean'].apply(find_match)
                adv_df = adv_df[adv_df['PLAYER_ID'].notna()].drop_duplicates(subset=['PLAYER_ID'])
                
                adv_cols = [c for c in ['PLAYER_ID', 'TS%', 'USG%', 'PER'] if c in adv_df.columns]
                master_player_df = pd.merge(master_player_df, adv_df[adv_cols], on="PLAYER_ID", how="left", suffixes=('', '_adv'))

        master_player_df = pd.merge(player_id_map[['PLAYER_ID', 'Player_Clean', 'TEAM_ID', 'TEAM_ABBREVIATION']], master_player_df, on="PLAYER_ID", how="right")
        master_player_df.rename(columns={'Player_Clean': 'clean_name'}, inplace=True)
        master_player_df.to_csv(output_dir / "master_player_stats.csv", index=False)
        logging.info("Saved master_player_stats.csv")
    except Exception as e:
        logging.error(f"Error in process_master_player_stats: {e}")

def process_master_team_stats(data_dir, player_id_map, output_dir):
    logging.info("--- Starting: process_master_team_stats ---")
    all_team_dfs = []
    team_id_to_abbr = player_id_map[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates().set_index('TEAM_ID')['TEAM_ABBREVIATION'].to_dict()

    for filepath in data_dir.glob("NBA Team *.csv"):
        file_type = sniff_file_type(filepath)
        
        if file_type == 'teamrankings':
            df = load_clean_csv(filepath, required_cols=['Team'])
            if df is None: continue
            metric_name = get_metric_from_filename(filepath.name)
            if not metric_name: continue
            
            # --- IMPROVED COLUMN SELECTION ---
            # Find all columns that look like years (e.g., "2025", "2026")
            year_cols = [col for col in df.columns if re.match(r'202\d', str(col))]
            
            if year_cols:
                # Pick the largest year (Current Season)
                val_col = max(year_cols, key=lambda x: int(x))
            else:
                # Fallback to default index if no year found (e.g. maybe "Value")
                val_col = df.columns[2] if len(df.columns) > 2 else None

            if not val_col: 
                continue

            df['TEAM_ABBREVIATION'] = df['Team'].map(TEAM_NAME_MAP)
            df = df[df['TEAM_ABBREVIATION'].notna()]
            df[metric_name] = pd.to_numeric(df[val_col].astype(str).str.replace(r'[%,]', '', regex=True), errors='coerce')
            all_team_dfs.append(df[['TEAM_ABBREVIATION', metric_name]])

        elif file_type == 'nba_api':
            df = load_clean_csv(filepath, required_cols=['TEAM_ID'])
            if df is None: continue
            metric_name = get_metric_from_filename(filepath.name)
            prefix = re.sub(r'[^A-Z_]', '', metric_name.upper()[:4])
            df['TEAM_ABBREVIATION'] = df['TEAM_ID'].map(team_id_to_abbr)
            df = df[df['TEAM_ABBREVIATION'].notna()]
            cols = [col for col in df.columns if col not in ['TEAM_ABBREVIATION', 'TEAM_ID', 'TEAM_NAME']]
            df.rename(columns={col: f"{prefix}_{col}" for col in cols}, inplace=True)
            all_team_dfs.append(df)

    if not all_team_dfs: return

    try:
        master_team_df = pd.DataFrame(player_id_map['TEAM_ABBREVIATION'].unique(), columns=['TEAM_ABBREVIATION']).dropna()
        for df in all_team_dfs:
            master_team_df = pd.merge(master_team_df, df, on='TEAM_ABBREVIATION', how='outer')
        master_team_df.to_csv(output_dir / "master_team_stats.csv", index=False)
        logging.info("Saved master_team_stats.csv")
    except Exception as e:
        logging.error(f"Error merging team stats: {e}")

def calculate_historical_vacancy(bs_df, player_df):
    logging.info("--- Calculating Historical Usage Vacancy (Split G/F) ---")
    usg_col = 'USG_PROXY' if 'USG_PROXY' in player_df.columns else 'USG%'
    if usg_col not in player_df.columns: return bs_df

    player_usg_map = player_df[['PLAYER_ID', usg_col]].drop_duplicates(subset=['PLAYER_ID']).set_index('PLAYER_ID')[usg_col].to_dict()
    
    # Map IDs to Position Group (G vs F)
    pos_map = player_df[['PLAYER_ID', 'Pos']].set_index('PLAYER_ID')['Pos'].to_dict()
    def is_guard(pid):
        pos = str(pos_map.get(pid, ''))
        return 1 if ('G' in pos) else 0
    
    bs_df = bs_df.sort_values('GAME_DATE')
    vacancy_records = []
    
    for team in bs_df['TEAM_ABBREVIATION'].dropna().unique():
        team_games = bs_df[bs_df['TEAM_ABBREVIATION'] == team].copy()
        if team_games.empty: continue

        min_matrix = team_games.pivot_table(index='Game_ID', columns='PLAYER_ID', values='MIN', aggfunc='max').fillna(0)
        rolling_mins = min_matrix.rolling(window=10, min_periods=1).mean().shift(1).fillna(0)
        is_missing = (rolling_mins > 12.0) & (min_matrix == 0)
        
        if not is_missing.any().any(): continue

        team_ids = min_matrix.columns
        team_usgs = np.array([player_usg_map.get(pid, 0.0) for pid in team_ids])
        
        # Create Position Masks
        is_guard_mask = np.array([is_guard(pid) for pid in team_ids])
        is_front_mask = 1 - is_guard_mask
        
        missing_usg = (is_missing.astype(float) * team_usgs).sum(axis=1)
        missing_min = (is_missing.astype(float) * rolling_mins).sum(axis=1)
        
        # NEW: Split Vacancy
        missing_usg_g = (is_missing.astype(float) * (team_usgs * is_guard_mask)).sum(axis=1)
        missing_usg_f = (is_missing.astype(float) * (team_usgs * is_front_mask)).sum(axis=1)
        
        vacancy_records.append(pd.DataFrame({
            'Game_ID': min_matrix.index, 'TEAM_ABBREVIATION': team,
            'TEAM_MISSING_USG': missing_usg, 'TEAM_MISSING_MIN': missing_min,
            'MISSING_USG_G': missing_usg_g, 'MISSING_USG_F': missing_usg_f
        }))

    if vacancy_records:
        full_vacancy = pd.concat(vacancy_records, ignore_index=True)
        bs_df = pd.merge(bs_df, full_vacancy, on=['Game_ID', 'TEAM_ABBREVIATION'], how='left')
        cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
        bs_df[cols] = bs_df[cols].fillna(0.0).round(2)
    else:
        bs_df['TEAM_MISSING_USG'] = 0.0
        bs_df['TEAM_MISSING_MIN'] = 0.0
        bs_df['MISSING_USG_G'] = 0.0
        bs_df['MISSING_USG_F'] = 0.0
    return bs_df

def process_master_box_scores(data_dir, player_id_map, output_dir):
    logging.info("--- Starting: process_master_box_scores ---")
    try:
        bs_df = load_clean_csv(data_dir / "NBA Player Box Scores.csv", required_cols=['Player_ID', 'Game_ID', 'PTS'])
        if bs_df is None: return

        bs_df.rename(columns={'Player_ID': 'PLAYER_ID'}, inplace=True)
        bs_df.dropna(subset=['PLAYER_ID'], inplace=True)
        bs_df['PLAYER_ID'] = bs_df['PLAYER_ID'].astype(int)
        if 'GAME_DATE' in bs_df.columns: bs_df['GAME_DATE'] = pd.to_datetime(bs_df['GAME_DATE'], errors='coerce')

        id_map = player_id_map[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'Player_Clean']].drop_duplicates(subset=['PLAYER_ID'])
        bs_df = pd.merge(bs_df, id_map, on='PLAYER_ID', how='left')
        
        # NEW: Merge Position from Master Player Stats
        player_stats_path = output_dir / "master_player_stats.csv"
        if player_stats_path.exists():
            p_stats = pd.read_csv(player_stats_path, usecols=['PLAYER_ID', 'Pos'])
            p_stats.drop_duplicates(subset=['PLAYER_ID'], inplace=True)
            bs_df = pd.merge(bs_df, p_stats, on='PLAYER_ID', how='left')

        for col in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 'FGM', 'FGA', 'FTA', 'MIN']:
            if col in bs_df.columns: bs_df[col] = pd.to_numeric(bs_df[col], errors='coerce').fillna(0)
        
        bs_df['PRA'] = bs_df['PTS'] + bs_df['REB'] + bs_df['AST']
        bs_df['PR'] = bs_df['PTS'] + bs_df['REB']
        bs_df['PA'] = bs_df['PTS'] + bs_df['AST']
        bs_df['RA'] = bs_df['REB'] + bs_df['AST']
        bs_df['STK'] = bs_df['STL'] + bs_df['BLK']
        bs_df['FANTASY_PTS'] = bs_df['PTS'] + (bs_df['REB']*1.2) + (bs_df['AST']*1.5) + (bs_df['STL']*3) + (bs_df['BLK']*3) - bs_df['TOV']
        
        ts_denom = 2 * (bs_df['FGA'] + 0.44 * bs_df['FTA'])
        bs_df['TS_PCT'] = np.where(ts_denom > 0, bs_df['PTS'] / ts_denom, 0.0)
        usg_num = (bs_df['FGA'] + 0.44 * bs_df['FTA'] + bs_df['TOV'])
        bs_df['USG_PROXY'] = np.where(bs_df['MIN'] > 0, usg_num / bs_df['MIN'], 0.0)

        # --- FIX 2: PER 36 NORMALIZATION ---
        per_36_cols = ['PTS', 'REB', 'AST', 'PRA', 'FG3M', 'STL', 'BLK', 'TOV']
        for col in per_36_cols:
            if col in bs_df.columns:
                bs_df[f'{col}_PER36'] = np.where(bs_df['MIN'] > 0, (bs_df[col] / bs_df['MIN']) * 36, 0.0).round(2)

        if (output_dir / "master_player_stats.csv").exists():
            bs_df = calculate_historical_vacancy(bs_df, pd.read_csv(output_dir / "master_player_stats.csv"))
        
        # NEW: Add Opponent Abbrev here so we don't have to do it later
        def get_opponent(matchup):
            if not isinstance(matchup, str): return "UNKNOWN"
            return matchup.split(" vs. ")[-1] if " vs. " in matchup else matchup.split(" @ ")[-1] if " @ " in matchup else "UNKNOWN"
        bs_df['OPPONENT_ABBREV'] = bs_df['MATCHUP'].apply(get_opponent)
        
        bs_df.to_csv(output_dir / "master_box_scores.csv", index=False)
        logging.info(f"Saved master_box_scores.csv ({len(bs_df)} rows)")
    except Exception as e:
        logging.error(f"Error in process_master_box_scores: {e}")

def process_vs_opponent_stats(data_dir, output_dir):
    logging.info("--- Starting: process_vs_opponent_stats ---")
    try:
        if not (output_dir / "master_box_scores.csv").exists(): return
        df = pd.read_csv(output_dir / "master_box_scores.csv", low_memory=False)
        
        # Check if OPPONENT_ABBREV exists (it should from process_master_box_scores now)
        if 'OPPONENT_ABBREV' not in df.columns:
            def get_opponent(matchup):
                if not isinstance(matchup, str): return "UNKNOWN"
                return matchup.split(" vs. ")[-1] if " vs. " in matchup else matchup.split(" @ ")[-1] if " @ " in matchup else "UNKNOWN"
            df['OPPONENT_ABBREV'] = df['MATCHUP'].apply(get_opponent)
        
        agg_cols = {k: 'mean' for k in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 'PRA', 'PR', 'PA', 'RA', 'FANTASY_PTS', 'MIN'] if k in df.columns}
        if 'Game_ID' in df.columns: agg_cols['Game_ID'] = 'count'
        
        vs_opp_df = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'OPPONENT_ABBREV']).agg(agg_cols).reset_index()
        if 'Game_ID' in vs_opp_df.columns: vs_opp_df.rename(columns={'Game_ID': 'GAMES_PLAYED'}, inplace=True)
        
        vs_opp_df.round(2).to_csv(output_dir / "master_vs_opponent.csv", index=False)
        logging.info("Saved master_vs_opponent.csv")
    except Exception as e:
        logging.error(f"Error in process_vs_opponent_stats: {e}")

def process_dvp_stats(output_dir):
    """Calculates Defense vs Position stats based on master_box_scores."""
    logging.info("--- Starting: process_dvp_stats ---")
    try:
        bs_path = output_dir / "master_box_scores.csv"
        if not bs_path.exists(): return

        df = pd.read_csv(bs_path, low_memory=False)
        
        # Ensure we have Position and Matchup info
        if 'Pos' not in df.columns or 'OPPONENT_ABBREV' not in df.columns:
            logging.warning("Missing Pos or OPPONENT_ABBREV in box scores. Skipping DvP.")
            return

        # --- IMPROVED POSITION NORMALIZATION ---
        def normalize_pos(pos):
            if not isinstance(pos, str): return 'UNKNOWN'
            p = pos.split('-')[0].upper().strip()
            # Map Generics to Standard
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p
        
        df['Primary_Pos'] = df['Pos'].apply(normalize_pos)
        
        # Filter strictly for standard 5 positions to keep DvP clean
        valid_positions = ['PG', 'SG', 'SF', 'PF', 'C']
        df = df[df['Primary_Pos'].isin(valid_positions)]

        # Calculate stats allowed by Opponent + Position
        stat_cols = ['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA', 'STL', 'BLK', 'TOV']
        # Use Per 36 metrics if available for better accuracy
        agg_dict = {}
        for col in stat_cols:
            if f'{col}_PER36' in df.columns:
                agg_dict[f'{col}_PER36'] = 'mean' # Use normalized stats for DvP if available
            elif col in df.columns:
                agg_dict[col] = 'mean'
        
        if not agg_dict: return

        dvp_df = df.groupby(['OPPONENT_ABBREV', 'Primary_Pos']).agg(agg_dict).reset_index()
        
        # Rename columns to DVP_{STAT}
        rename_map = {}
        for col in agg_dict.keys():
            base_stat = col.replace('_PER36', '')
            rename_map[col] = f"DVP_{base_stat}"
            
        dvp_df.rename(columns=rename_map, inplace=True)

        dvp_df.round(2).to_csv(output_dir / "master_dvp_stats.csv", index=False)
        logging.info("Saved master_dvp_stats.csv")
    except Exception as e:
        logging.error(f"Error in process_dvp_stats: {e}")

def main():
    start_time = pd.Timestamp.now()
    logging.info(f"========= STARTING NBA DATA ETL SCRIPT =========")
    
    # Use Config Paths
    DATA_DIR = cfg.DATA_DIR
    OUTPUT_DIR = cfg.DATA_DIR
    
    player_id_map = create_player_id_map(DATA_DIR)
    if player_id_map is None:
        logging.critical("Fatal Error: Could not create Player ID Map.")
        return

    process_master_player_stats(DATA_DIR, player_id_map, OUTPUT_DIR)
    process_master_team_stats(DATA_DIR, player_id_map, OUTPUT_DIR)
    process_master_box_scores(DATA_DIR, player_id_map, OUTPUT_DIR)
    process_vs_opponent_stats(DATA_DIR, OUTPUT_DIR)
    process_dvp_stats(OUTPUT_DIR) # <--- NEW Phase 2 Call

    end_time = pd.Timestamp.now()
    logging.info(f"========= ETL COMPLETE. TIME: {end_time - start_time} =========")

if __name__ == "__main__":
    main()