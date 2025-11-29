import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader

def add_rolling_stats_history(df):
    """
    Calculates historical rolling features on the full box score dataset.
    This creates a snapshot of 'stats so far' for every point in time.
    """
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    
    stats_to_roll = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'FG3M', 'STL', 'BLK', 'TOV', 'FANTASY_PTS']
    for col in stats_to_roll:
        if col not in df.columns: df[col] = 0.0

    grouped = df.groupby('PLAYER_ID')

    # Rolling Stats (Inclusive of current row)
    for col in stats_to_roll:
        # Season Average (Expanding)
        df[f'{col}_SZN_AVG'] = grouped[col].expanding().mean().values
        
        # L5 Avg
        df[f'{col}_L5_AVG'] = grouped[col].rolling(window=5, min_periods=1).mean().values
        
        # L10 Std
        df[f'{col}_L10_STD'] = grouped[col].rolling(window=10, min_periods=3).std().values
        
        # EWMA 
        df[f'{col}_L5_EWMA'] = grouped[col].ewm(alpha=0.15, adjust=False).mean().values

    if 'USG_PROXY' in df.columns:
        df['SZN_USG_PROXY'] = grouped['USG_PROXY'].expanding().mean().values
        df['L5_USG_PROXY'] = grouped['USG_PROXY'].rolling(window=5).mean().values
        
    return df

def build_feature_set(props_df):
    """
    Constructs the feature matrix (X) for a list of props.
    Uses point-in-time lookup to prevent data leakage for backtests.
    """
    logging.info("Building feature set with Point-in-Time safety...")
    
    # 1. Load Data
    box_scores = loader.load_box_scores()
    
    # Load other static files
    player_stats_static, team_stats, league_pace = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    
    dvp_df = None
    if cfg.MASTER_DVP_FILE.exists():
        dvp_df = pd.read_csv(cfg.MASTER_DVP_FILE)

    # 2. Map Player Names to IDs
    if 'PLAYER_ID' not in props_df.columns:
        if player_stats_static is not None:
            name_map = player_stats_static.set_index('clean_name')['PLAYER_ID'].to_dict()
            
            # Normalize input names
            props_df['clean_name'] = props_df['Player Name'].apply(lambda x: str(x).lower().strip())
            
            # --- FIX: Manual Name Mapping ---
            # Maps: Input Name (lowercase) -> Master File Name (lowercase)
            manual_map = {
                'deuce mcbride': 'miles mcbride',
                'cam johnson': 'cameron johnson',
                'lu dort': 'luguentz dort',
                'pj washington': 'p.j. washington'
            }
            props_df['clean_name'] = props_df['clean_name'].replace(manual_map)
            
            # Perform mapping
            props_df['PLAYER_ID'] = props_df['clean_name'].map(name_map)
            
            # Handle Unmapped Players
            missing_ids = props_df[props_df['PLAYER_ID'].isna()]
            if not missing_ids.empty:
                logging.warning(f"Dropping {len(missing_ids)} props due to unrecognized player names: {missing_ids['Player Name'].unique()}")
                props_df = props_df.dropna(subset=['PLAYER_ID']).copy()
            
            if props_df.empty:
                logging.error("No valid props remaining after name mapping.")
                return pd.DataFrame()

            # Force conversion to int64 to match box scores
            props_df['PLAYER_ID'] = props_df['PLAYER_ID'].astype('int64')
            
        else:
            logging.error("Cannot map names: Player stats file missing.")
            return pd.DataFrame()

    # 3. Time-Travel Feature Engineering
    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating dynamic historical stats...")
        
        # Ensure box scores ID is strictly int64
        if box_scores['PLAYER_ID'].dtype != 'int64':
             box_scores['PLAYER_ID'] = box_scores['PLAYER_ID'].astype('int64')

        # Calculate stats for every game in history
        history_df = add_rolling_stats_history(box_scores.copy())
        
        # Prepare for Merge
        props_df['GAME_DATE'] = pd.to_datetime(props_df['GAME_DATE'])
        history_df['GAME_DATE'] = pd.to_datetime(history_df['GAME_DATE'])
        
        # Sort strictly for merge_asof
        props_df = props_df.sort_values('GAME_DATE')
        history_df = history_df.sort_values('GAME_DATE')
        
        # Merge Asof (Historical Rolling Stats)
        features_df = pd.merge_asof(
            props_df,
            history_df,
            on='GAME_DATE',
            by='PLAYER_ID',
            direction='backward',
            allow_exact_matches=False,
            suffixes=('', '_hist')
        )
        
        # --- CRITICAL FIX START ---
        # Force merge of static stats (Season Avg, etc.) even when box scores are used.
        # This prevents the model from seeing 0.0 for 'SZN Avg'.
        if player_stats_static is not None:
            # Only select columns that don't already exist (or the join key) to avoid conflicts
            cols_to_use = [c for c in player_stats_static.columns if c not in features_df.columns or c == 'PLAYER_ID']
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on='PLAYER_ID', how='left')
        # --- CRITICAL FIX END ---
        
        logging.info(f"Point-in-time merge complete. Rows: {len(features_df)}")
    else:
        logging.warning("No box scores found. Falling back to static stats (Potentially Unsafe for Backtesting).")
        features_df = pd.merge(props_df, player_stats_static, on='PLAYER_ID', how='left')

    # 4. Merge Team and Opponent Stats
    if 'TEAM_ABBREVIATION' not in features_df.columns and 'Team' in features_df.columns:
        features_df['TEAM_ABBREVIATION'] = features_df['Team']
        
    if team_stats is not None:
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        features_df = pd.merge(features_df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        features_df = pd.merge(features_df, opp_stats_renamed, left_on='Opponent', right_index=True, how='left')

    # 5. Merge DVP
    if dvp_df is not None:
        if 'Pos' not in features_df.columns:
             features_df['Pos'] = 'PG' 
             
        def normalize_pos(p):
            p = str(p).split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p if p in ['PG','SG','SF','PF','C'] else 'PG'
            
        features_df['Primary_Pos'] = features_df['Pos'].apply(normalize_pos)
        
        features_df = pd.merge(
            features_df, 
            dvp_df, 
            left_on=['Opponent', 'Primary_Pos'], 
            right_on=['OPPONENT_ABBREV', 'Primary_Pos'], 
            how='left'
        )

    # 6. Merge H2H
    if not vs_opp_df.empty:
        features_df = pd.merge(
            features_df,
            vs_opp_df,
            left_on=['PLAYER_ID', 'Opponent'],
            right_on=['PLAYER_ID', 'OPPONENT_ABBREV'],
            how='left'
        )

    # 7. Final Polish
    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['GAME_PACE'] = features_df['TEAM_Possessions per Game']
        
    cols_to_fill = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_df