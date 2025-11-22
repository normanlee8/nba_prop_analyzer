import pandas as pd
import logging
from pathlib import Path
from prop_analyzer import config as cfg
from prop_analyzer.utils.text import preprocess_name_for_fuzzy_match

_INJURY_CACHE = None
_INJURY_WARNING_SHOWN = False

def load_static_data():
    """
    Loads master player and team stats.
    Returns: (player_stats_df, team_stats_df, league_pace_avg)
    """
    logging.info("--- Loading Static Data Files ---")
    try:
        if not cfg.MASTER_PLAYER_FILE.exists():
            logging.error(f"File not found: {cfg.MASTER_PLAYER_FILE}")
            return None, None, 100.0

        player_stats_df = pd.read_csv(cfg.MASTER_PLAYER_FILE)
        # Pre-calculate fuzzy match keys for speed
        if 'clean_name' in player_stats_df.columns:
            player_stats_df['processed_name'] = player_stats_df['clean_name'].apply(preprocess_name_for_fuzzy_match)
        
        if cfg.MASTER_TEAM_FILE.exists():
            team_stats_df = pd.read_csv(cfg.MASTER_TEAM_FILE)
            if 'TEAM_ABBREVIATION' in team_stats_df.columns:
                team_stats_df.set_index('TEAM_ABBREVIATION', inplace=True)
        else:
            logging.warning(f"File not found: {cfg.MASTER_TEAM_FILE}")
            team_stats_df = pd.DataFrame()
        
        # Calculate League Pace
        pace_col = team_stats_df.get('Possessions per Game')
        league_pace_avg = pace_col.mean() if pace_col is not None else 100.0
        
        return player_stats_df, team_stats_df, league_pace_avg
    except Exception as e:
        logging.critical(f"FATAL: Failed to load static master files: {e}", exc_info=True)
        return None, None, 100.0

def load_box_scores(player_ids=None):
    """
    Loads box scores, optionally filtering for specific players (for optimization).
    """
    try:
        if not cfg.MASTER_BOX_SCORES_FILE.exists():
            logging.warning("master_box_scores.csv not found.")
            return None

        # Memory Optimization: Read in chunks if filtering
        if player_ids is not None:
            chunks = []
            # Convert IDs to set for O(1) lookup
            id_set = set(player_ids)
            
            for chunk in pd.read_csv(cfg.MASTER_BOX_SCORES_FILE, chunksize=50000, low_memory=False):
                filtered = chunk[chunk['PLAYER_ID'].isin(id_set)]
                if not filtered.empty:
                    chunks.append(filtered)
            
            if not chunks: 
                return pd.DataFrame()
            box_scores_df = pd.concat(chunks)
        else:
            # Load full file
            box_scores_df = pd.read_csv(cfg.MASTER_BOX_SCORES_FILE, low_memory=False)
        
        if box_scores_df.empty: return None

        # Ensure Date Parsing
        box_scores_df['GAME_DATE'] = pd.to_datetime(box_scores_df['GAME_DATE'], errors='coerce').dt.normalize()
        box_scores_df.dropna(subset=['GAME_DATE'], inplace=True)
        box_scores_df.sort_values(by='GAME_DATE', ascending=False, inplace=True)
        
        return box_scores_df
    except Exception as e:
        logging.critical(f"FATAL: Failed to load box scores: {e}", exc_info=True)
        return None

def load_vs_opponent_data():
    """Loads the H2H stats file."""
    path = cfg.MASTER_VS_OPP_FILE
    if not path.exists(): return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        # Ensure columns have VS_OPP_ prefix if they don't already
        cols_to_rename = {}
        for c in df.columns:
            if c not in ['PLAYER_ID', 'OPPONENT_ABBREV', 'PLAYER_NAME', 'GAMES_PLAYED'] and not c.startswith('VS_OPP_'):
                cols_to_rename[c] = f"VS_OPP_{c}"
        
        if cols_to_rename:
            df.rename(columns=cols_to_rename, inplace=True)
            
        return df
    except: return pd.DataFrame()

def get_cached_injury_data():
    """
    Returns the injury dataframe, cached in memory to prevent disk I/O loops.
    """
    global _INJURY_CACHE, _INJURY_WARNING_SHOWN
    
    if _INJURY_CACHE is not None: 
        return _INJURY_CACHE
    
    # Try loading from config path, then local fallback
    paths = [cfg.DATA_DIR / "daily_injuries.csv", Path("daily_injuries.csv")]
    
    for p in paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                # Ensure Status_Clean exists
                if 'Status_Clean' not in df.columns and 'Injury Status' in df.columns:
                    df['Status_Clean'] = df['Injury Status'].apply(
                        lambda x: 'OUT' if 'out' in str(x).lower() else 'GTD' if 'question' in str(x).lower() else 'UNKNOWN'
                    )
                
                _INJURY_CACHE = df
                return df
            except Exception as e:
                logging.warning(f"Failed to read injury file {p}: {e}")
            
    if not _INJURY_WARNING_SHOWN:
        logging.warning("daily_injuries.csv not found. Vacancy logic will be disabled.")
        _INJURY_WARNING_SHOWN = True
        
    return None