import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from rapidfuzz import process, fuzz
from unidecode import unidecode
import warnings

# Import config and Data Contract
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols

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

def load_clean_data(filepath_stem, required_cols=[]):
    """
    Smart loader: Prefers Parquet, falls back to CSV.
    filepath_stem: Path object without extension OR string.
    """
    if isinstance(filepath_stem, Path):
        path_str = str(filepath_stem)
        base = re.sub(r'\.(csv|parquet)$', '', path_str)
    else:
        base = str(filepath_stem)
    
    parquet_path = Path(base + ".parquet")
    csv_path = Path(base + ".csv")

    try:
        df = None
        if parquet_path.exists():
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            df = pd.read_csv(csv_path, low_memory=False)
        else:
            return None
            
        if df is not None and not df.empty and required_cols:
            missing = [col for col in required_cols if col not in df.columns]
            if missing:
                pass 
        
        return df
    except Exception as e:
        logging.error(f"Error loading {base}: {e}")
        return None

def sniff_file_type(filepath):
    fname = filepath.name
    if 'NBA Team' in fname:
        if 'Defense' in fname or 'General' in fname: return 'nba_api'
        return 'teamrankings'
    return 'nba_api' 

def get_metric_from_filename(filename, prefix="NBA Team "):
    clean_name = re.sub(r'\.(csv|parquet)$', '', filename)
    if not clean_name.startswith(prefix):
        return None
    return clean_name[len(prefix):]

def create_player_id_map(data_dir, season_folders):
    logging.info("Creating Player ID Map across all seasons...")
    all_player_dfs = []
    required_cols = [Cols.PLAYER_ID, 'PLAYER_NAME', 'TEAM_ID', 'TEAM_ABBREVIATION']

    for folder in season_folders:
        for friendly_name in ["Home", "Away:Road", "Last 5 Games"]:
            file_stem = folder / f"NBA Player Stats {friendly_name}"
            df = load_clean_data(file_stem, required_cols=required_cols)
            
            if df is not None:
                df[Cols.PLAYER_ID] = pd.to_numeric(df[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
                existing_cols = [c for c in required_cols if c in df.columns]
                all_player_dfs.append(df[existing_cols])
    
    if not all_player_dfs:
        logging.critical("CRITICAL: No valid player stat files found in any season folder.")
        return None
        
    player_map_df = pd.concat(all_player_dfs)
    player_map_df.drop_duplicates(subset=[Cols.PLAYER_ID], inplace=True)
    player_map_df['Player_Clean'] = player_map_df['PLAYER_NAME'].apply(lambda x: unidecode(str(x)).lower().strip())
    return player_map_df

def process_master_player_stats(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_player_stats (Separate Files) ---")
    
    for folder in season_folders:
        season_id = folder.name
        try:
            api_player_stats = []
            for file_prefix, friendly_name in PLAYER_STAT_PREFIX_MAP.items():
                file_stem = folder / f"NBA Player Stats {friendly_name}"
                df = load_clean_data(file_stem, required_cols=[Cols.PLAYER_ID, 'PLAYER_NAME'])
                if df is not None:
                    df[Cols.PLAYER_ID] = pd.to_numeric(df[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
                    df = df.add_prefix(f"{file_prefix}_")
                    df.rename(columns={f"{file_prefix}_{Cols.PLAYER_ID}": Cols.PLAYER_ID, f"{file_prefix}_PLAYER_NAME": "PLAYER_NAME"}, inplace=True)
                    api_player_stats.append(df)

            if not api_player_stats:
                continue
                
            season_player_df = api_player_stats[0]
            for df in api_player_stats[1:]:
                season_player_df = pd.merge(season_player_df, df, on=[Cols.PLAYER_ID, "PLAYER_NAME"], how="outer")
            
            # Quarter Stats
            for q in range(1, 5):
                file_stem = folder / f"NBA Player Q{q}"
                df_q = load_clean_data(file_stem, required_cols=[Cols.PLAYER_ID, 'PTS', 'MIN'])
                if df_q is not None:
                    df_q[Cols.PLAYER_ID] = pd.to_numeric(df_q[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
                    cols_to_norm = ['MIN', 'FGM', 'FGA', 'FG3M', 'FG3A', 'FTM', 'FTA', 'OREB', 'DREB', 'REB', 'AST', 'TOV', 'STL', 'BLK', 'PF', 'PTS']
                    cols_to_norm = [c for c in cols_to_norm if c in df_q.columns]
                    
                    if 'GP' in df_q.columns:
                        df_q = df_q[df_q['GP'] > 0].copy()
                        for col in cols_to_norm:
                            df_q[col] = (df_q[col] / df_q['GP']).round(2)

                    df_q = df_q.add_prefix(f"Q{q}_")
                    df_q.rename(columns={f"Q{q}_{Cols.PLAYER_ID}": Cols.PLAYER_ID}, inplace=True)
                    season_player_df = pd.merge(season_player_df, df_q, on=Cols.PLAYER_ID, how="left")

            # Bball-Ref Stats
            bball_ref_stem = folder / "NBA Player Per Game Averages"
            bball_ref_df = load_clean_data(bball_ref_stem, required_cols=['Player', 'PTS'])
            if bball_ref_df is not None:
                bball_ref_df['Player_Clean'] = bball_ref_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                bball_ref_df = bball_ref_df.rename(columns=BBREF_COLUMN_MAP)
                
                id_map_clean = player_id_map[[Cols.PLAYER_ID, 'Player_Clean']].drop_duplicates(subset=['Player_Clean'])
                name_to_id = id_map_clean.set_index('Player_Clean')[Cols.PLAYER_ID].to_dict()
                
                def find_match(name):
                    if not name: return None
                    match = process.extractOne(name, name_to_id.keys(), scorer=fuzz.token_sort_ratio, score_cutoff=90)
                    return name_to_id.get(match[0]) if match else None

                bball_ref_df[Cols.PLAYER_ID] = bball_ref_df['Player_Clean'].apply(find_match)
                bball_ref_df = bball_ref_df[bball_ref_df[Cols.PLAYER_ID].notna()]
                bball_ref_df[Cols.PLAYER_ID] = bball_ref_df[Cols.PLAYER_ID].astype(int)
                bball_ref_df.drop_duplicates(subset=[Cols.PLAYER_ID], inplace=True)
                
                season_cols = [Cols.PLAYER_ID, 'Pos', 'SEASON_G', 'SEASON_PTS', 'SEASON_TRB', 'SEASON_AST', 'SEASON_STL', 'SEASON_BLK', 'SEASON_FG3M']
                cols_exist = [col for col in season_cols if col in bball_ref_df.columns]
                season_player_df = pd.merge(season_player_df, bball_ref_df[cols_exist], on=Cols.PLAYER_ID, how="left")
                
                # Advanced
                adv_stem = folder / "NBA Player Advanced Stats"
                adv_df = load_clean_data(adv_stem, required_cols=['Player', 'USG%'])
                if adv_df is not None:
                    adv_df['Player_Clean'] = adv_df['Player'].apply(lambda x: unidecode(str(x)).lower().strip())
                    adv_df[Cols.PLAYER_ID] = adv_df['Player_Clean'].apply(find_match)
                    adv_df = adv_df[adv_df[Cols.PLAYER_ID].notna()].drop_duplicates(subset=[Cols.PLAYER_ID])
                    adv_df[Cols.PLAYER_ID] = adv_df[Cols.PLAYER_ID].astype(int)
                    
                    adv_cols = [c for c in [Cols.PLAYER_ID, 'TS%', 'USG%', 'PER'] if c in adv_df.columns]
                    season_player_df = pd.merge(season_player_df, adv_df[adv_cols], on=Cols.PLAYER_ID, how="left", suffixes=('', '_adv'))

            season_player_df['SEASON_ID'] = season_id
            
            # Clean names
            season_player_df = pd.merge(player_id_map[[Cols.PLAYER_ID, 'Player_Clean', 'TEAM_ID', 'TEAM_ABBREVIATION']], season_player_df, on=Cols.PLAYER_ID, how="right")
            season_player_df.rename(columns={'Player_Clean': 'clean_name'}, inplace=True)
            
            out_name = f"master_player_stats_{season_id}.parquet"
            season_player_df.to_parquet(output_dir / out_name, index=False)
            logging.info(f"Saved {out_name}")
            
        except Exception as e:
            logging.error(f"Error processing player stats for {folder}: {e}")

def process_master_team_stats(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_team_stats (Separate Files) ---")
    
    team_id_to_abbr = player_id_map[['TEAM_ID', 'TEAM_ABBREVIATION']].drop_duplicates().set_index('TEAM_ID')['TEAM_ABBREVIATION'].to_dict()

    for folder in season_folders:
        season_id = folder.name
        season_team_dfs = []
        
        files = list(folder.glob("NBA Team *.csv")) + list(folder.glob("NBA Team *.parquet"))
        
        for filepath in files:
            file_type = sniff_file_type(filepath)
            df = load_clean_data(filepath.parent / filepath.stem)
            if df is None: continue

            if file_type == 'teamrankings':
                metric_name = get_metric_from_filename(filepath.name)
                if not metric_name: continue
                
                year_cols = [col for col in df.columns if re.match(r'202\d', str(col))]
                val_col = max(year_cols, key=lambda x: str(x)) if year_cols else (df.columns[2] if len(df.columns) > 2 else None)

                if not val_col: continue

                df['TEAM_ABBREVIATION'] = df['Team'].map(TEAM_NAME_MAP)
                df = df[df['TEAM_ABBREVIATION'].notna()]
                df[metric_name] = pd.to_numeric(df[val_col].astype(str).str.replace(r'[%,]', '', regex=True), errors='coerce')
                season_team_dfs.append(df[['TEAM_ABBREVIATION', metric_name]])

            elif file_type == 'nba_api':
                metric_name = get_metric_from_filename(filepath.name)
                prefix = re.sub(r'[^A-Z_]', '', metric_name.upper()[:4])
                
                if 'TEAM_ID' not in df.columns: continue
                
                df['TEAM_ABBREVIATION'] = df['TEAM_ID'].map(team_id_to_abbr)
                df = df[df['TEAM_ABBREVIATION'].notna()]
                cols = [col for col in df.columns if col not in ['TEAM_ABBREVIATION', 'TEAM_ID', 'TEAM_NAME']]
                df.rename(columns={col: f"{prefix}_{col}" for col in cols}, inplace=True)
                season_team_dfs.append(df)

        if season_team_dfs:
            season_master = pd.DataFrame(player_id_map['TEAM_ABBREVIATION'].unique(), columns=['TEAM_ABBREVIATION']).dropna()
            for df in season_team_dfs:
                season_master = pd.merge(season_master, df, on='TEAM_ABBREVIATION', how='outer')
            
            season_master['SEASON_ID'] = season_id
            
            out_name = f"master_team_stats_{season_id}.parquet"
            season_master.to_parquet(output_dir / out_name, index=False)
            logging.info(f"Saved {out_name}")

def calculate_historical_vacancy(bs_df, player_df):
    logging.info("--- Initializing Historical Vacancy Columns (Placeholder) ---")
    vacancy_cols = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in vacancy_cols:
        if c not in bs_df.columns:
            bs_df[c] = 0.0
        else:
            bs_df[c] = bs_df[c].fillna(0.0)
    return bs_df

def process_master_box_scores(player_id_map, season_folders, output_dir):
    logging.info("--- Starting: process_master_box_scores (Separate Files) ---")
    
    for folder in season_folders:
        season_id = folder.name
        try:
            file_stem = folder / "NBA Player Box Scores"
            bs_df = load_clean_data(file_stem)
            
            if bs_df is None or bs_df.empty: 
                continue

            # Standardize Column Names
            rename_map = {}
            if 'Player_ID' in bs_df.columns and Cols.PLAYER_ID not in bs_df.columns:
                rename_map['Player_ID'] = Cols.PLAYER_ID
            if 'Game_ID' in bs_df.columns and Cols.GAME_ID not in bs_df.columns:
                rename_map['Game_ID'] = Cols.GAME_ID
            
            if rename_map:
                bs_df.rename(columns=rename_map, inplace=True)

            # Drop redundant columns to prevent '_x' / '_y' suffixes from merges
            cols_to_drop = ['PLAYER_NAME', 'TEAM_ABBREVIATION', 'Player_Clean']
            bs_df.drop(columns=[c for c in cols_to_drop if c in bs_df.columns], inplace=True)

            bs_df.dropna(subset=[Cols.PLAYER_ID], inplace=True)
            bs_df[Cols.PLAYER_ID] = bs_df[Cols.PLAYER_ID].astype(int)
            
            # Ensure GAME_ID is numeric if present
            if Cols.GAME_ID in bs_df.columns:
                bs_df[Cols.GAME_ID] = pd.to_numeric(bs_df[Cols.GAME_ID], errors='coerce').fillna(0).astype(int)
            
            # Handle Date
            if 'GAME_DATE' in bs_df.columns: 
                bs_df[Cols.DATE] = pd.to_datetime(bs_df['GAME_DATE'], errors='coerce')
            
            # Merge Player Info
            id_map = player_id_map[[Cols.PLAYER_ID, 'PLAYER_NAME', 'TEAM_ABBREVIATION', 'Player_Clean']].drop_duplicates(subset=[Cols.PLAYER_ID])
            bs_df = pd.merge(bs_df, id_map, on=Cols.PLAYER_ID, how='left')
            
            # Merge Position Info
            p_stats_path = output_dir / f"master_player_stats_{season_id}.parquet"
            if p_stats_path.exists():
                p_stats = pd.read_parquet(p_stats_path)
                if Cols.PLAYER_ID in p_stats.columns:
                    p_stats[Cols.PLAYER_ID] = pd.to_numeric(p_stats[Cols.PLAYER_ID], errors='coerce').fillna(0).astype(int)
                    p_stats_szn = p_stats[[Cols.PLAYER_ID, 'Pos']].drop_duplicates(subset=[Cols.PLAYER_ID])
                    bs_df = pd.merge(bs_df, p_stats_szn, on=Cols.PLAYER_ID, how='left')

            # Numeric Conversions
            numeric_cols = [
                'PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 
                'FGM', 'FGA', 'FTA', 'FTM', 'OREB', 'DREB', 'MIN'
            ]
            for col in numeric_cols:
                if col in bs_df.columns: 
                    bs_df[col] = pd.to_numeric(bs_df[col], errors='coerce').fillna(0)
            
            # Derived Stats
            bs_df['PRA'] = bs_df['PTS'] + bs_df['REB'] + bs_df['AST']
            bs_df['PR'] = bs_df['PTS'] + bs_df['REB']
            bs_df['PA'] = bs_df['PTS'] + bs_df['AST']
            bs_df['RA'] = bs_df['REB'] + bs_df['AST']
            bs_df['STK'] = bs_df['STL'] + bs_df['BLK']
            bs_df['FANTASY_PTS'] = bs_df['PTS'] + (bs_df['REB']*1.2) + (bs_df['AST']*1.5) + (bs_df['STL']*3) + (bs_df['BLK']*3) - bs_df['TOV']
            
            # Double/Triple Doubles
            dd_cols = ['PTS', 'REB', 'AST', 'STL', 'BLK']
            if all(c in bs_df.columns for c in dd_cols):
                counts = bs_df[dd_cols].ge(10).sum(axis=1)
                bs_df['DD'] = counts.ge(2).astype(int)
                bs_df['TD'] = counts.ge(3).astype(int)
            else:
                bs_df['DD'] = 0
                bs_df['TD'] = 0

            # Advanced Metrics
            ts_denom = 2 * (bs_df['FGA'] + 0.44 * bs_df['FTA'])
            bs_df['TS_PCT'] = np.where(ts_denom > 0, bs_df['PTS'] / ts_denom, 0.0)
            
            usg_num = (bs_df['FGA'] + 0.44 * bs_df['FTA'] + bs_df['TOV'])
            bs_df['USG_PROXY'] = np.where(bs_df['MIN'] > 0, usg_num / bs_df['MIN'], 0.0)

            per_36_cols = ['PTS', 'REB', 'AST', 'PRA', 'FG3M', 'STL', 'BLK', 'TOV']
            for col in per_36_cols:
                if col in bs_df.columns:
                    bs_df[f'{col}_PER36'] = np.where(bs_df['MIN'] > 0, (bs_df[col] / bs_df['MIN']) * 36, 0.0).round(2)

            bs_df['SEASON_ID'] = season_id
            
            if p_stats_path.exists():
                bs_df = calculate_historical_vacancy(bs_df, pd.read_parquet(p_stats_path))
            
            def get_opponent(matchup):
                if not isinstance(matchup, str): return "UNKNOWN"
                return matchup.split(" vs. ")[-1] if " vs. " in matchup else matchup.split(" @ ")[-1] if " @ " in matchup else "UNKNOWN"
            
            if 'MATCHUP' in bs_df.columns:
                bs_df['OPPONENT_ABBREV'] = bs_df['MATCHUP'].apply(get_opponent)
            else:
                bs_df['OPPONENT_ABBREV'] = "UNK"

            subset_cols = [Cols.PLAYER_ID, Cols.DATE]
            if Cols.GAME_ID in bs_df.columns:
                subset_cols.insert(1, Cols.GAME_ID)
            
            bs_df.drop_duplicates(subset=subset_cols, keep='last', inplace=True)
            
            out_name = f"master_box_scores_{season_id}.parquet"
            bs_df.to_parquet(output_dir / out_name, index=False)
            logging.info(f"Saved {out_name} ({len(bs_df)} rows)")
            
        except Exception as e:
            logging.error(f"Error processing box scores for {season_id}: {e}", exc_info=True)

def process_vs_opponent_stats(data_dir, output_dir):
    logging.info("--- Starting: process_vs_opponent_stats ---")
    all_files = sorted(output_dir.glob("master_box_scores_*.parquet"))
    if not all_files: return

    dfs = []
    for f in all_files:
        try:
            dfs.append(pd.read_parquet(f))
        except: pass
    
    if not dfs: return
    df = pd.concat(dfs, ignore_index=True)
    
    agg_cols = {k: 'mean' for k in ['PTS', 'REB', 'AST', 'STL', 'BLK', 'FG3M', 'TOV', 'PRA', 'PR', 'PA', 'RA', 'FANTASY_PTS', 'MIN', 'DD', 'TD'] if k in df.columns}
    
    count_col = Cols.GAME_ID if Cols.GAME_ID in df.columns else 'Game_ID'
    if count_col in df.columns: 
        agg_cols[count_col] = 'count'
    
    vs_opp_df = df.groupby([Cols.PLAYER_ID, 'PLAYER_NAME', 'OPPONENT_ABBREV']).agg(agg_cols).reset_index()
    if count_col in vs_opp_df.columns: 
        vs_opp_df.rename(columns={count_col: 'GAMES_PLAYED'}, inplace=True)
    
    vs_opp_df.round(2).to_parquet(output_dir / "master_vs_opponent.parquet", index=False)
    logging.info("Saved master_vs_opponent.parquet")

def process_dvp_stats(output_dir):
    """
    Calculates Defense vs Position (DvP) stats PER SEASON to avoid historical leakage.
    Improved: Iterates through ALL master_box_scores_{season}.parquet files, calculates 
    DVP for that specific season, and aggregates them.
    """
    logging.info("--- Starting: process_dvp_stats (Season-Aware) ---")
    files = sorted(output_dir.glob("master_box_scores_*.parquet"))
    if not files: return
    
    all_dvp_dfs = []

    for file_path in files:
        try:
            # Extract season_id from filename: master_box_scores_2024-25.parquet
            match = re.search(r'\d{4}-\d{2}', file_path.name)
            season_id = match.group(0) if match else "UNKNOWN"
            logging.info(f"Calculating DvP for Season: {season_id}")
            
            df = pd.read_parquet(file_path)
            
            required = ['Pos', 'OPPONENT_ABBREV']
            if Cols.DATE in df.columns: required.append(Cols.DATE)
            
            if not all(c in df.columns for c in required):
                logging.warning(f"Skipping {season_id}: Missing required columns.")
                continue

            def normalize_pos(pos):
                if not isinstance(pos, str): return 'UNKNOWN'
                p = pos.split('-')[0].upper().strip()
                if p == 'G': return 'SG' # Default Guard bucket
                if p == 'F': return 'PF' # Default Forward bucket
                return p
            
            df['Primary_Pos'] = df['Pos'].apply(normalize_pos)
            valid_positions = ['PG', 'SG', 'SF', 'PF', 'C']
            df = df[df['Primary_Pos'].isin(valid_positions)].copy()

            # Sort for expanding window calculation
            if Cols.DATE in df.columns:
                df.sort_values(by=[Cols.PLAYER_ID, Cols.DATE], inplace=True)

            stat_cols = ['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA', 'STL', 'BLK', 'TOV']
            
            # 1. Calculate Player's Avg Entering the Game (To baseline performance)
            # GroupBy Player within this season
            for col in stat_cols:
                if col in df.columns:
                    # Expanding mean excluding current game (shift 1)
                    exp_series = df.groupby(Cols.PLAYER_ID)[col].expanding().mean()
                    df[f'{col}_AVG'] = exp_series.groupby(level=0).shift(1).reset_index(level=0, drop=True)
                    
            # Drop rows where history is NaN (first game of season for player)
            # DVP needs at least one game of history to establish baseline performance
            df.dropna(subset=[f'{c}_AVG' for c in stat_cols if c in df.columns], inplace=True)

            # 2. Calculate Differential (Actual - Expected)
            for col in stat_cols:
                if col in df.columns:
                    df[f'{col}_DIFF'] = df[col] - df[f'{col}_AVG']

            # 3. Aggregate Differences by Opponent & Position for this season
            diff_cols = {f'{col}_DIFF': 'mean' for col in stat_cols if col in df.columns}
            if not diff_cols: continue
            
            dvp_diffs = df.groupby(['OPPONENT_ABBREV', 'Primary_Pos']).agg(diff_cols).reset_index()

            # 4. Calculate League Average Baseline for Position for this season
            league_pos_baselines = df.groupby('Primary_Pos')[stat_cols].mean().reset_index()
            
            rename_map = {c: f"{c}_BASE" for c in stat_cols}
            league_pos_baselines.rename(columns=rename_map, inplace=True)
            
            # 5. Combine
            merged_dvp = pd.merge(dvp_diffs, league_pos_baselines, on='Primary_Pos', how='inner')
            
            season_dvp = pd.DataFrame()
            season_dvp['SEASON_ID'] = season_id
            season_dvp['OPPONENT_ABBREV'] = merged_dvp['OPPONENT_ABBREV']
            season_dvp['Primary_Pos'] = merged_dvp['Primary_Pos']
            
            for col in stat_cols:
                if f'{col}_DIFF' in merged_dvp.columns and f'{col}_BASE' in merged_dvp.columns:
                    # DVP_{STAT} = Baseline + Diff
                    season_dvp[f'DVP_{col}'] = merged_dvp[f'{col}_BASE'] + merged_dvp[f'{col}_DIFF']
            
            all_dvp_dfs.append(season_dvp)
            
        except Exception as e:
            logging.error(f"Error processing DVP for {file_path.name}: {e}", exc_info=True)

    if all_dvp_dfs:
        final_dvp_all = pd.concat(all_dvp_dfs, ignore_index=True)
        final_dvp_all.round(2).to_parquet(output_dir / "master_dvp_stats.parquet", index=False)
        logging.info(f"Saved master_dvp_stats.parquet (Multi-Season: {len(final_dvp_all)} rows)")

def process_q1_history(output_dir):
    """
    1. Aggregates daily Q1 logs -> master_q1_stats.parquet
    2. Aggregates daily Q2 logs + Q1 logs -> master_1h_stats.parquet
    """
    logging.info("--- Processing Quarter/Half History ---")
    
    # --- PROCESS Q1 ---
    q1_files = sorted(output_dir.rglob("daily_q1_stats_*.parquet"))
    full_q1 = pd.DataFrame()
    
    if q1_files:
        dfs = []
        for f in q1_files:
            try: dfs.append(pd.read_parquet(f))
            except: pass
        if dfs:
            full_q1 = pd.concat(dfs, ignore_index=True)
            # Cleanup Q1
            if 'GAME_DATE' in full_q1.columns:
                full_q1['GAME_DATE'] = pd.to_datetime(full_q1['GAME_DATE']).dt.normalize()
            
            # Robust drop duplicates
            dedup_subset = [Cols.PLAYER_ID, 'GAME_DATE']
            if Cols.GAME_ID in full_q1.columns:
                dedup_subset.append(Cols.GAME_ID)
                
            full_q1 = full_q1.drop_duplicates(subset=dedup_subset, keep='last')
            
            # Numeric conversion
            cols_num = ['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'MIN']
            for c in cols_num:
                if c in full_q1.columns: full_q1[c] = pd.to_numeric(full_q1[c], errors='coerce').fillna(0)
            
            full_q1.to_parquet(cfg.MASTER_Q1_FILE, index=False)
            logging.info(f"Saved Master Q1: {len(full_q1)} rows")

    # --- PROCESS 1H (Q1 + Q2) ---
    q2_files = sorted(output_dir.rglob("daily_q2_stats_*.parquet"))
    
    if q1_files and q2_files:
        logging.info("Building 1st Half (1H) stats from Q1 + Q2...")
        
        # Load Q2
        dfs_q2 = []
        for f in q2_files:
            try: dfs_q2.append(pd.read_parquet(f))
            except: pass
            
        if dfs_q2:
            full_q2 = pd.concat(dfs_q2, ignore_index=True)
            if 'GAME_DATE' in full_q2.columns:
                full_q2['GAME_DATE'] = pd.to_datetime(full_q2['GAME_DATE']).dt.normalize()
            
            dedup_subset = [Cols.PLAYER_ID, 'GAME_DATE']
            if Cols.GAME_ID in full_q2.columns:
                dedup_subset.append(Cols.GAME_ID)
            
            full_q2 = full_q2.drop_duplicates(subset=dedup_subset, keep='last')
            
            cols_num = ['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'MIN']
            for c in cols_num:
                if c in full_q2.columns: full_q2[c] = pd.to_numeric(full_q2[c], errors='coerce').fillna(0)

            # Merge Q1 + Q2 on Player/Date
            # Use GAME_ID if available for tighter merge
            merge_on = [Cols.PLAYER_ID, 'GAME_DATE', 'PLAYER_NAME', 'TEAM_ABBREVIATION']
            if Cols.GAME_ID in full_q1.columns and Cols.GAME_ID in full_q2.columns:
                merge_on.append(Cols.GAME_ID)

            merged = pd.merge(
                full_q1, full_q2, 
                on=merge_on, 
                how='inner', 
                suffixes=('_q1', '_q2')
            )
            
            # Calculate 1H Stats
            stats_to_sum = ['PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 'MIN']
            for col in stats_to_sum:
                if f'{col}_q1' in merged.columns and f'{col}_q2' in merged.columns:
                    merged[col] = merged[f'{col}_q1'] + merged[f'{col}_q2']
            
            # Save 1H Master
            # Keep only standard cols + ID
            keep_cols = [Cols.PLAYER_ID, 'GAME_DATE', 'PLAYER_NAME', 'TEAM_ABBREVIATION']
            if Cols.GAME_ID in merged.columns:
                keep_cols.append(Cols.GAME_ID)
            keep_cols += stats_to_sum
            
            final_1h = merged[keep_cols].copy()
            
            final_1h.to_parquet(cfg.MASTER_1H_FILE, index=False)
            logging.info(f"Saved Master 1H: {len(final_1h)} rows")
    else:
        logging.warning("Missing Q2 files. Cannot build 1H history.")