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

def get_season_folders(data_dir):
    """
    Finds all season subfolders (e.g., '2024-25', '2025-26') in the data directory.
    Returns sorted list of Path objects.
    """
    folders = [f for f in data_dir.iterdir() if f.is_dir() and re.match(r'\d{4}-\d{2}', f.name)]
    return sorted(folders)

def load_clean_csv(filepath, required_cols):
    if not filepath.exists():
        # logging.warning(f"File not found: {filepath.name}. Skipping.")
        return None
            
    try:
        df = pd.read_csv(filepath, low_memory=False)
        if df.empty: return None
            
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            logging.warning(f"File {filepath} is missing cols: {missing}.")
        
        return df
    except Exception as e:
        logging.error(f"Error loading {filepath}: {e}")
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

def create_player_id_map(data_dir, season_folders):
    logging.info("Creating Player ID Map across all seasons...")
    all_player_dfs = []
    
    required_cols = ['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION']

    for folder in season_folders:
        for friendly_name in ["Home", "Away:Road", "Last 5 Games"]:
            filepath = folder / f"NBA Player Stats {friendly_name}.csv"
            df = load_clean_csv(filepath, required_cols=required_cols)
            if df is not None:
                all_player_dfs.append(df[required_cols])
    
    if not all_player_dfs:
        logging.critical("CRITICAL: No valid player stat files found in any season folder.")
        return None
        
    player_map_df = pd.concat(all_player_dfs)
    player_map_df.drop_duplicates(subset=['PLAYER_ID'], inplace=True)
    
    player_map_df['Player_Clean'] = player_map_df['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).lower().strip())
    return player_map_df

def process_master_player_stats(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_player_stats (Multi-Season) ---")
    
    master_dfs = []

    for folder in season_folders:
        season_id = folder.name
        # logging.info(f"Processing Player Stats for {season_id}...")
        
        try:
            api_player_stats = []
            for file_prefix, friendly_name in PLAYER_STAT_PREFIX_MAP.items():
                filepath = folder / f"NBA Player Stats {friendly_name}.csv"
                df = load_clean_csv(filepath, required_cols=['PLAYER_ID', 'PLAYER_NAME'])
                if df is not None:
                    df = df.add_prefix(f"{file_prefix}_")
                    df.rename(columns={f"{file_prefix}_PLAYER_ID": "PLAYER_ID", f"{file_prefix}_PLAYER_NAME": "PLAYER_NAME"}, inplace=True)
                    api_player_stats.append(df)

            if not api_player_stats: continue
                
            season_player_df = api_player_stats[0]
            for df in api_player_stats[1:]:
                season_player_df = pd.merge(season_player_df, df, on=["PLAYER_ID", "PLAYER_NAME"], how="outer")
            
            # Load Quarter Stats
            for q in range(1, 5):
                filepath = folder / f"NBA Player Q{q}.csv"
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
                    season_player_df = pd.merge(season_player_df, df_q, on="PLAYER_ID", how="left")

            # Bball-Ref Stats
            bball_ref_df = load_clean_csv(folder / "NBA Player Per Game Averages.csv", required_cols=['Player', 'PTS'])
            if bball_ref_df is not None:
                bball_ref_df['Player_Clean'] = bball_ref_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                bball_ref_df = bball_ref_df.rename(columns=BBREF_COLUMN_MAP)
                
                # Match IDs
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
                season_player_df = pd.merge(season_player_df, bball_ref_df[cols_exist], on="PLAYER_ID", how="left")
                
                # Advanced Stats
                adv_df = load_clean_csv(folder / "NBA Player Advanced Stats.csv", required_cols=['Player', 'USG%'])
                if adv_df is not None:
                    adv_df['Player_Clean'] = adv_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                    adv_df['PLAYER_ID'] = adv_df['Player_Clean'].apply(find_match)
                    adv_df = adv_df[adv_df['PLAYER_ID'].notna()].drop_duplicates(subset=['PLAYER_ID'])
                    
                    adv_cols = [c for c in ['PLAYER_ID', 'TS%', 'USG%', 'PER'] if c in adv_df.columns]
                    season_player_df = pd.merge(season_player_df, adv_df[adv_cols], on="PLAYER_ID", how="left", suffixes=('', '_adv'))

            # Add Season ID
            season_player_df['SEASON_ID'] = season_id
            master_dfs.append(season_player_df)
            
        except Exception as e:
            logging.error(f"Error processing player stats for {folder}: {e}")

    if master_dfs:
        # Concatenate all seasons
        full_master_df = pd.concat(master_dfs, ignore_index=True)
        # Merge with ID map to ensure clean names
        full_master_df = pd.merge(player_id_map[['PLAYER_ID', 'Player_Clean', 'TEAM_ID', 'TEAM_ABBREVIATION']], full_master_df, on="PLAYER_ID", how="right")
        full_master_df.rename(columns={'Player_Clean': 'clean_name'}, inplace=True)
        
        full_master_df.to_csv(output_dir / "master_player_stats.csv", index=False)
        logging.info("Saved master_player_stats.csv")

def process_master_team_stats(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_team_stats (Multi-Season) ---")
    
    # We only really need the LATEST team stats for today's predictions
    # But for historical training, we might want past season team stats attached to past games.
    # For simplicity in this version, we will prioritize the CURRENT season for the 'master' file,
    # or simple stacking.
    
    all_season_teams = []
    
    team_id_to_abbr = player_id_map[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates().set_index('TEAM_ID')['TEAM_ABBREVIATION'].to_dict()

    for folder in season_folders:
        season_id = folder.name
        season_team_dfs = []
        
        for filepath in folder.glob("NBA Team *.csv"):
            file_type = sniff_file_type(filepath)
            
            if file_type == 'teamrankings':
                df = load_clean_csv(filepath, required_cols=['Team'])
                if df is None: continue
                metric_name = get_metric_from_filename(filepath.name)
                if not metric_name: continue
                
                # Dynamic Year Column Picker
                year_cols = [col for col in df.columns if re.match(r'202\d', str(col))]
                val_col = max(year_cols, key=lambda x: int(x)) if year_cols else (df.columns[2] if len(df.columns) > 2 else None)

                if not val_col: continue

                df['TEAM_ABBREVIATION'] = df['Team'].map(TEAM_NAME_MAP)
                df = df[df['TEAM_ABBREVIATION'].notna()]
                df[metric_name] = pd.to_numeric(df[val_col].astype(str).str.replace(r'[%,]', '', regex=True), errors='coerce')
                season_team_dfs.append(df[['TEAM_ABBREVIATION', metric_name]])

            elif file_type == 'nba_api':
                df = load_clean_csv(filepath, required_cols=['TEAM_ID'])
                if df is None: continue
                metric_name = get_metric_from_filename(filepath.name)
                prefix = re.sub(r'[^A-Z_]', '', metric_name.upper()[:4])
                df['TEAM_ABBREVIATION'] = df['TEAM_ID'].map(team_id_to_abbr)
                df = df[df['TEAM_ABBREVIATION'].notna()]
                cols = [col for col in df.columns if col not in ['TEAM_ABBREVIATION', 'TEAM_ID', 'TEAM_NAME']]
                df.rename(columns={col: f"{prefix}_{col}" for col in cols}, inplace=True)
                season_team_dfs.append(df)

        if season_team_dfs:
            # Merge all metrics for THIS season
            season_master = pd.DataFrame(player_id_map['TEAM_ABBREVIATION'].unique(), columns=['TEAM_ABBREVIATION']).dropna()
            for df in season_team_dfs:
                season_master = pd.merge(season_master, df, on='TEAM_ABBREVIATION', how='outer')
            
            season_master['SEASON_ID'] = season_id
            all_season_teams.append(season_master)

    if all_season_teams:
        master_team_df = pd.concat(all_season_teams, ignore_index=True)
        master_team_df.to_csv(output_dir / "master_team_stats.csv", index=False)
        logging.info("Saved master_team_stats.csv")

def calculate_historical_vacancy(bs_df, player_df):
    logging.info("--- Calculating Historical Usage Vacancy (Split G/F) ---")
    # Vacancy is complex with multi-season. 
    # We will process vacancy strictly within season groups to avoid data bleed.
    
    vacancy_results = []
    
    # Group by Season to calculate vacancy correctly per year
    for season_id, season_bs in bs_df.groupby('SEASON_ID'):
        season_player_stats = player_df[player_df['SEASON_ID'] == season_id]
        if season_player_stats.empty:
            # Fallback to any player stats if season missing
            season_player_stats = player_df
            
        usg_col = 'USG_PROXY' if 'USG_PROXY' in season_player_stats.columns else 'USG%'
        if usg_col not in season_player_stats.columns: 
            vacancy_results.append(season_bs)
            continue

        player_usg_map = season_player_stats[['PLAYER_ID', usg_col]].drop_duplicates(subset=['PLAYER_ID']).set_index('PLAYER_ID')[usg_col].to_dict()
        pos_map = season_player_stats[['PLAYER_ID', 'Pos']].set_index('PLAYER_ID')['Pos'].to_dict()
        
        def is_guard(pid):
            pos = str(pos_map.get(pid, ''))
            return 1 if ('G' in pos) else 0
        
        season_bs = season_bs.sort_values('GAME_DATE')
        season_vacancy_records = []
        
        for team in season_bs['TEAM_ABBREVIATION'].dropna().unique():
            team_games = season_bs[season_bs['TEAM_ABBREVIATION'] == team].copy()
            if team_games.empty: continue

            min_matrix = team_games.pivot_table(index='Game_ID', columns='PLAYER_ID', values='MIN', aggfunc='max').fillna(0)
            rolling_mins = min_matrix.rolling(window=10, min_periods=1).mean().shift(1).fillna(0)
            is_missing = (rolling_mins > 12.0) & (min_matrix == 0)
            
            if not is_missing.any().any(): continue

            team_ids = min_matrix.columns
            team_usgs = np.array([player_usg_map.get(pid, 0.0) for pid in team_ids])
            is_guard_mask = np.array([is_guard(pid) for pid in team_ids])
            is_front_mask = 1 - is_guard_mask
            
            missing_usg = (is_missing.astype(float) * team_usgs).sum(axis=1)
            missing_min = (is_missing.astype(float) * rolling_mins).sum(axis=1)
            missing_usg_g = (is_missing.astype(float) * (team_usgs * is_guard_mask)).sum(axis=1)
            missing_usg_f = (is_missing.astype(float) * (team_usgs * is_front_mask)).sum(axis=1)
            
            season_vacancy_records.append(pd.DataFrame({
                'Game_ID': min_matrix.index, 
                'TEAM_ABBREVIATION': team,
                'TEAM_MISSING_USG': missing_usg, 
                'TEAM_MISSING_MIN': missing_min,
                'MISSING_USG_G': missing_usg_g, 
                'MISSING_USG_F': missing_usg_f
            }))

        if season_vacancy_records:
            full_vacancy = pd.concat(season_vacancy_records, ignore_index=True)
            # Merge back only to this season's chunk
            season_bs = pd.merge(season_bs, full_vacancy, on=['Game_ID', 'TEAM_ABBREVIATION'], how='left')
        else:
            for c in ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']:
                season_bs[c] = 0.0
                
        vacancy_results.append(season_bs)

    final_df = pd.concat(vacancy_results, ignore_index=True)
    cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    final_df[cols] = final_df[cols].fillna(0.0).round(2)
    return final_df

def process_master_box_scores(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_box_scores (Multi-Season) ---")
    
    all_box_scores = []
    
    for folder in season_folders:
        season_id = folder.name
        try:
            bs_df = load_clean_csv(folder / "NBA Player Box Scores.csv", required_cols=['Player_ID', 'Game_ID', 'PTS'])
            if bs_df is None: continue

            bs_df.rename(columns={'Player_ID': 'PLAYER_ID'}, inplace=True)
            bs_df.dropna(subset=['PLAYER_ID'], inplace=True)
            bs_df['PLAYER_ID'] = bs_df['PLAYER_ID'].astype(int)
            if 'GAME_DATE' in bs_df.columns: bs_df['GAME_DATE'] = pd.to_datetime(bs_df['GAME_DATE'], errors='coerce')

            id_map = player_id_map[['PLAYER_ID', 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'Player_Clean']].drop_duplicates(subset=['PLAYER_ID'])
            bs_df = pd.merge(bs_df, id_map, on='PLAYER_ID', how='left')
            
            # Merge Position from Master Player Stats (filtered by Season)
            player_stats_path = output_dir / "master_player_stats.csv"
            if player_stats_path.exists():
                p_stats = pd.read_csv(player_stats_path)
                # Filter for this season to get accurate role for that year
                p_stats_szn = p_stats[p_stats['SEASON_ID'] == season_id]
                if not p_stats_szn.empty:
                    p_stats_szn = p_stats_szn[['PLAYER_ID', 'Pos']].drop_duplicates(subset=['PLAYER_ID'])
                    bs_df = pd.merge(bs_df, p_stats_szn, on='PLAYER_ID', how='left')

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

            # PER 36
            per_36_cols = ['PTS', 'REB', 'AST', 'PRA', 'FG3M', 'STL', 'BLK', 'TOV']
            for col in per_36_cols:
                if col in bs_df.columns:
                    bs_df[f'{col}_PER36'] = np.where(bs_df['MIN'] > 0, (bs_df[col] / bs_df['MIN']) * 36, 0.0).round(2)

            bs_df['SEASON_ID'] = season_id
            all_box_scores.append(bs_df)
            
        except Exception as e:
            logging.error(f"Error processing box scores for {season_id}: {e}")

    if not all_box_scores: return

    master_bs_df = pd.concat(all_box_scores, ignore_index=True)

    # Vacancy Calc (Updated to handle multi-season)
    if (output_dir / "master_player_stats.csv").exists():
        master_bs_df = calculate_historical_vacancy(master_bs_df, pd.read_csv(output_dir / "master_player_stats.csv"))
    
    # Add Opponent
    def get_opponent(matchup):
        if not isinstance(matchup, str): return "UNKNOWN"
        return matchup.split(" vs. ")[-1] if " vs. " in matchup else matchup.split(" @ ")[-1] if " @ " in matchup else "UNKNOWN"
    master_bs_df['OPPONENT_ABBREV'] = master_bs_df['MATCHUP'].apply(get_opponent)
    
    master_bs_df.to_csv(output_dir / "master_box_scores.csv", index=False)
    logging.info(f"Saved master_box_scores.csv ({len(master_bs_df)} rows)")

def process_vs_opponent_stats(data_dir, output_dir):
    # This remains largely the same, but uses the new master file
    logging.info("--- Starting: process_vs_opponent_stats ---")
    try:
        if not (output_dir / "master_box_scores.csv").exists(): return
        df = pd.read_csv(output_dir / "master_box_scores.csv", low_memory=False)
        
        agg_cols = {k: 'mean' for k in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 'PRA', 'PR', 'PA', 'RA', 'FANTASY_PTS', 'MIN'] if k in df.columns}
        if 'Game_ID' in df.columns: agg_cols['Game_ID'] = 'count'
        
        # Group by Player and Opponent (Averages over ALL seasons provided)
        vs_opp_df = df.groupby(['PLAYER_ID', 'PLAYER_NAME', 'OPPONENT_ABBREV']).agg(agg_cols).reset_index()
        if 'Game_ID' in vs_opp_df.columns: vs_opp_df.rename(columns={'Game_ID': 'GAMES_PLAYED'}, inplace=True)
        
        vs_opp_df.round(2).to_csv(output_dir / "master_vs_opponent.csv", index=False)
        logging.info("Saved master_vs_opponent.csv")
    except Exception as e:
        logging.error(f"Error in process_vs_opponent_stats: {e}")

def process_dvp_stats(output_dir):
    logging.info("--- Starting: process_dvp_stats ---")
    try:
        bs_path = output_dir / "master_box_scores.csv"
        if not bs_path.exists(): return

        df = pd.read_csv(bs_path, low_memory=False)
        
        # Filter for CURRENT SEASON ONLY for DVP
        # You generally don't want 2024 defense stats polluting 2025 predictions
        # unless you purposefully want a multi-year average. 
        # For now, let's take the latest season if possible.
        latest_season = df['SEASON_ID'].max()
        df = df[df['SEASON_ID'] == latest_season].copy()
        logging.info(f"Calculating DvP using only latest season data: {latest_season}")

        if 'Pos' not in df.columns or 'OPPONENT_ABBREV' not in df.columns: return

        def normalize_pos(pos):
            if not isinstance(pos, str): return 'UNKNOWN'
            p = pos.split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p
        
        df['Primary_Pos'] = df['Pos'].apply(normalize_pos)
        valid_positions = ['PG', 'SG', 'SF', 'PF', 'C']
        df = df[df['Primary_Pos'].isin(valid_positions)]

        stat_cols = ['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA', 'STL', 'BLK', 'TOV']
        agg_dict = {}
        for col in stat_cols:
            if f'{col}_PER36' in df.columns:
                agg_dict[f'{col}_PER36'] = 'mean'
            elif col in df.columns:
                agg_dict[col] = 'mean'
        
        if not agg_dict: return

        dvp_df = df.groupby(['OPPONENT_ABBREV', 'Primary_Pos']).agg(agg_dict).reset_index()
        
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
    logging.info(f"========= STARTING NBA DATA ETL SCRIPT (MULTI-SEASON) =========")
    
    DATA_DIR = cfg.DATA_DIR
    OUTPUT_DIR = cfg.DATA_DIR
    
    # 1. Identify Seasons
    season_folders = get_season_folders(DATA_DIR)
    if not season_folders:
        logging.critical(f"No season folders (e.g. 2024-25) found in {DATA_DIR}")
        return
    logging.info(f"Found seasons: {[f.name for f in season_folders]}")

    # 2. Build Maps
    player_id_map = create_player_id_map(DATA_DIR, season_folders)
    if player_id_map is None: return

    # 3. Process Aggregates
    process_master_player_stats(player_id_map, season_folders, OUTPUT_DIR)
    process_master_team_stats(player_id_map, season_folders, OUTPUT_DIR)
    process_master_box_scores(player_id_map, season_folders, OUTPUT_DIR)
    
    # 4. Derivative Stats
    process_vs_opponent_stats(DATA_DIR, OUTPUT_DIR)
    process_dvp_stats(OUTPUT_DIR)

    end_time = pd.Timestamp.now()
    logging.info(f"========= ETL COMPLETE. TIME: {end_time - start_time} =========")

if __name__ == "__main__":
    main()