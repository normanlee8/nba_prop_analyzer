import pandas as pd
import logging
import re
from pathlib import Path
from prop_analyzer import config as cfg
from prop_analyzer.utils.text import preprocess_name_for_fuzzy_match

_INJURY_CACHE = None
_INJURY_WARNING_SHOWN = False

def load_static_data():
    """
    Loads master player and team stats.
    Merges ALL found season files (e.g. 2024-25 and 2025-26), 
    prioritizing the latest season's data for deduplication.
    """
    logging.info("--- Loading Static Data Files ---")
    try:
        # 1. Load Player Stats
        player_files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_PLAYER_PATTERN))
        if not player_files:
            logging.error("No master_player_stats files found.")
            return None, None, 100.0
            
        player_dfs = []
        for f in player_files:
            try:
                df = pd.read_csv(f)
                player_dfs.append(df)
            except Exception as e:
                logging.warning(f"Error reading {f}: {e}")
        
        if player_dfs:
            player_stats_df = pd.concat(player_dfs, ignore_index=True)
            # Deduplicate: Keep LAST occurrence (which corresponds to latest file due to sorted() above)
            if 'PLAYER_ID' in player_stats_df.columns:
                player_stats_df = player_stats_df.drop_duplicates(subset=['PLAYER_ID'], keep='last')
            
            if 'clean_name' in player_stats_df.columns:
                player_stats_df['processed_name'] = player_stats_df['clean_name'].apply(preprocess_name_for_fuzzy_match)
        else:
            player_stats_df = pd.DataFrame()

        # 2. Load Team Stats
        team_files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_TEAM_PATTERN))
        if team_files:
            team_dfs = []
            for f in team_files:
                try:
                    df = pd.read_csv(f)
                    team_dfs.append(df)
                except Exception as e:
                    logging.warning(f"Error reading {f}: {e}")
            
            if team_dfs:
                team_stats_df = pd.concat(team_dfs, ignore_index=True)
                if 'TEAM_ABBREVIATION' in team_stats_df.columns:
                    # Keep latest stats for each team
                    team_stats_df = team_stats_df.drop_duplicates(subset=['TEAM_ABBREVIATION'], keep='last')
                    team_stats_df.set_index('TEAM_ABBREVIATION', inplace=True)
            else:
                team_stats_df = pd.DataFrame()
        else:
            team_stats_df = pd.DataFrame()
        
        pace_col = team_stats_df.get('Possessions per Game')
        league_pace_avg = pace_col.mean() if pace_col is not None else 100.0
        
        return player_stats_df, team_stats_df, league_pace_avg
    except Exception as e:
        logging.critical(f"FATAL: Failed to load static master files: {e}", exc_info=True)
        return None, None, 100.0

def load_box_scores(player_ids=None):
    """
    Loads ALL master_box_scores_*.csv files and combines them.
    This gives the model the full multi-season history it needs.
    """
    try:
        # Glob all season files
        files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_BOX_SCORES_PATTERN))
        if not files:
            logging.warning("No master_box_scores files found.")
            return None

        dfs = []
        for f in files:
            try:
                # Memory Optimization: Read in chunks if filtering
                if player_ids is not None:
                    id_set = set(player_ids)
                    chunks = []
                    for chunk in pd.read_csv(f, chunksize=50000, low_memory=False):
                        filtered = chunk[chunk['PLAYER_ID'].isin(id_set)]
                        if not filtered.empty:
                            chunks.append(filtered)
                    if chunks:
                        dfs.append(pd.concat(chunks))
                else:
                    dfs.append(pd.read_csv(f, low_memory=False))
            except Exception as e:
                logging.warning(f"Failed to read {f}: {e}")

        if not dfs: return None
        
        box_scores_df = pd.concat(dfs, ignore_index=True)

        box_scores_df['GAME_DATE'] = pd.to_datetime(box_scores_df['GAME_DATE'], errors='coerce').dt.normalize()
        box_scores_df.dropna(subset=['GAME_DATE'], inplace=True)
        box_scores_df.sort_values(by='GAME_DATE', ascending=False, inplace=True)
        
        return box_scores_df
    except Exception as e:
        logging.critical(f"FATAL: Failed to load box scores: {e}", exc_info=True)
        return None

def load_vs_opponent_data():
    path = cfg.MASTER_VS_OPP_FILE
    if not path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        cols_to_rename = {}
        for c in df.columns:
            if c not in ['PLAYER_ID', 'OPPONENT_ABBREV', 'PLAYER_NAME', 'GAMES_PLAYED'] and not c.startswith('VS_OPP_'):
                cols_to_rename[c] = f"VS_OPP_{c}"
        if cols_to_rename:
            df.rename(columns=cols_to_rename, inplace=True)
        return df
    except: return pd.DataFrame()

def get_cached_injury_data():
    global _INJURY_CACHE, _INJURY_WARNING_SHOWN
    
    if _INJURY_CACHE is not None: 
        return _INJURY_CACHE
    
    search_paths = []
    
    if cfg.DATA_DIR.exists():
        season_folders = sorted([f for f in cfg.DATA_DIR.iterdir() if f.is_dir() and re.match(r'\d{4}-\d{2}', f.name)], reverse=True)
        if season_folders:
            search_paths.append(season_folders[0] / "daily_injuries.csv")
    
    search_paths.append(cfg.DATA_DIR / "daily_injuries.csv")
    search_paths.append(Path("daily_injuries.csv"))
    
    for p in search_paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if 'Status_Clean' not in df.columns and 'Injury Status' in df.columns:
                    df['Status_Clean'] = df['Injury Status'].apply(
                        lambda x: 'OUT' if 'out' in str(x).lower() else 'GTD' if 'question' in str(x).lower() else 'UNKNOWN'
                    )
                _INJURY_CACHE = df
                return df
            except Exception as e:
                logging.warning(f"Failed to read injury file {p}: {e}")
            
    if not _INJURY_WARNING_SHOWN:
        logging.warning("daily_injuries.csv not found in any season folder.")
        _INJURY_WARNING_SHOWN = True
        
    return None