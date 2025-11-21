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
    """
    logging.info("--- Loading Static Data Files ---")
    try:
        if not cfg.MASTER_PLAYER_FILE.exists():
            logging.error(f"File not found: {cfg.MASTER_PLAYER_FILE}")
            return None, None, 100.0

        player_stats_df = pd.read_csv(cfg.MASTER_PLAYER_FILE)
        player_stats_df['processed_name'] = player_stats_df['clean_name'].apply(preprocess_name_for_fuzzy_match)
        
        team_stats_df = pd.read_csv(cfg.MASTER_TEAM_FILE)
        if 'TEAM_ABBREVIATION' in team_stats_df.columns:
            team_stats_df.set_index('TEAM_ABBREVIATION', inplace=True)
        
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

        chunks = []
        for chunk in pd.read_csv(cfg.MASTER_BOX_SCORES_FILE, chunksize=50000, low_memory=False):
            if player_ids is not None:
                chunk = chunk[chunk['PLAYER_ID'].isin(player_ids)]
            chunks.append(chunk)
        
        if not chunks: return pd.DataFrame()
        
        box_scores_df = pd.concat(chunks)
        if box_scores_df.empty: return None

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
        cols_to_rename = {c: f"VS_OPP_{c}" for c in df.columns if c not in ['PLAYER_ID', 'OPPONENT_ABBREV', 'PLAYER_NAME']}
        df.rename(columns=cols_to_rename, inplace=True)
        return df
    except: return pd.DataFrame()

def get_cached_injury_data():
    global _INJURY_CACHE, _INJURY_WARNING_SHOWN
    if _INJURY_CACHE is not None: return _INJURY_CACHE
    
    paths = [cfg.DATA_DIR / "daily_injuries.csv", Path("daily_injuries.csv")]
    for p in paths:
        if p.exists():
            try:
                df = pd.read_csv(p)
                if 'Status_Clean' not in df.columns and 'Injury Status' in df.columns:
                    df['Status_Clean'] = df['Injury Status'].apply(
                        lambda x: 'OUT' if 'out' in str(x).lower() else 'GTD' if 'question' in str(x).lower() else 'UNKNOWN'
                    )
                _INJURY_CACHE = df
                return df
            except: pass
            
    if not _INJURY_WARNING_SHOWN:
        logging.warning("daily_injuries.csv not found.")
        _INJURY_WARNING_SHOWN = True
    return None