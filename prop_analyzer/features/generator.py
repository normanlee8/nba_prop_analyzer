import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader

def add_rolling_stats_history(df):
    """
    Calculates historical rolling features on the full box score dataset.
    Used for Time-Travel lookups during Inference.
    """
    # Sort by Player + Date
    # Ensure columns exist before sort
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns in box scores. Cols found: {df.columns}")
        return df

    df = df.sort_values(by=[Cols.PLAYER_ID, Cols.DATE]).reset_index(drop=True)
    
    # Stats to roll (Must match dataset.py)
    stats_to_roll = [
        'PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 
        'FG3M', 'STL', 'BLK', 'TOV', 'FANTASY_PTS', 
        'FGA', 'FG3A', 'DD', 'TD'
    ]
    for col in stats_to_roll:
        if col not in df.columns: df[col] = 0.0

    grouped = df.groupby(Cols.PLAYER_ID)

    for col in stats_to_roll:
        # Note: For inference, we calculate the value 'as of' the last game played.
        # merge_asof will later grab the appropriate row.
        
        # SZN Avg (Expanding Mean)
        df[f'{col}_{Cols.SZN_AVG}'] = grouped[col].expanding().mean().values
        
        # L5 Avg (Rolling 5)
        df[f'{col}_{Cols.L5_AVG}'] = grouped[col].rolling(window=5, min_periods=1).mean().values
        
        # L10 Std Dev
        df[f'{col}_L10_STD'] = grouped[col].rolling(window=10, min_periods=3).std().values
        
        # EWMA
        df[f'{col}_L5_EWMA'] = grouped[col].ewm(alpha=0.15, adjust=False).mean().values

    # Advanced Stats
    if 'USG_PROXY' in df.columns:
        df['SZN_USG_PROXY'] = grouped['USG_PROXY'].expanding().mean().values
        df['L5_USG_PROXY'] = grouped['USG_PROXY'].rolling(window=5).mean().values
        
    if 'TS_PCT' in df.columns:
        df['SZN_TS_PCT'] = grouped['TS_PCT'].expanding().mean().values
        
    return df

def build_feature_set(props_df):
    logging.info("Building feature set with Point-in-Time safety...")
    
    # 1. Load Data
    box_scores = loader.load_box_scores()
    player_stats_static, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    
    dvp_df = None
    if cfg.MASTER_DVP_FILE.exists():
        # --- FIX: Read Parquet instead of CSV ---
        try:
            dvp_df = pd.read_parquet(cfg.MASTER_DVP_FILE)
        except Exception as e:
            logging.error(f"Failed to read DVP Parquet: {e}")
            dvp_df = None
        # ----------------------------------------

    # 2. Map Player Names to IDs
    # Using standardized names from parser.py/Cols
    if Cols.PLAYER_ID not in props_df.columns:
        if player_stats_static is not None:
            # Map clean_name -> ID
            # Note: ETL now produces clean_name in master_player_stats
            name_map = player_stats_static.set_index('clean_name')[Cols.PLAYER_ID].to_dict()
            
            # Helper to clean input names
            props_df['clean_name'] = props_df[Cols.PLAYER_NAME].apply(lambda x: str(x).lower().strip())
            
            # Apply Manual Fixes (Still useful for edge cases/typos in props input)
            manual_map = {
                'deuce mcbride': 'miles mcbride',
                'cam johnson': 'cameron johnson',
                'lu dort': 'luguentz dort',
                'pj washington': 'p.j. washington',
                'jimmy butler': 'jimmy butler iii',
                'herb jones': 'herbert jones',
                'robert williams': 'robert williams iii',
                'trey murphy': 'trey murphy iii',
                'kelly oubre': 'kelly oubre jr.',
                'michael porter': 'michael porter jr.'
            }
            props_df['clean_name'] = props_df['clean_name'].replace(manual_map)
            
            props_df[Cols.PLAYER_ID] = props_df['clean_name'].map(name_map)
            
            # Log failures
            missing_ids = props_df[props_df[Cols.PLAYER_ID].isna()]
            if not missing_ids.empty:
                logging.warning(f"Dropping {len(missing_ids)} props - Unrecognized names: {missing_ids[Cols.PLAYER_NAME].unique()}")
                props_df = props_df.dropna(subset=[Cols.PLAYER_ID]).copy()
            
            if props_df.empty:
                logging.error("No valid props remaining after name mapping.")
                return pd.DataFrame()

            props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        else:
            logging.error("Cannot map names: Player stats file missing.")
            return pd.DataFrame()

    # 3. Time-Travel Feature Engineering
    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating dynamic historical stats...")
        
        # Ensure Types match for merge
        box_scores[Cols.PLAYER_ID] = box_scores[Cols.PLAYER_ID].fillna(0).astype('int64')
        props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        
        # Standardize Date in box scores (if not already done by loader/etl)
        if Cols.DATE in box_scores.columns:
            box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE])
        elif 'GAME_DATE' in box_scores.columns:
             box_scores[Cols.DATE] = pd.to_datetime(box_scores['GAME_DATE'])

        # Calculate history
        history_df = add_rolling_stats_history(box_scores.copy())
        
        # Prepare Merge
        props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE])
        history_df[Cols.DATE] = pd.to_datetime(history_df[Cols.DATE])
        
        props_df = props_df.sort_values(Cols.DATE)
        history_df = history_df.sort_values(Cols.DATE)
        
        # Merge AsOf (Backward)
        # For a prop on Date T, finds the last row in history where Date <= T.
        features_df = pd.merge_asof(
            props_df,
            history_df,
            on=Cols.DATE,
            by=Cols.PLAYER_ID,
            direction='backward',
            suffixes=('', '_hist')
        )
        
        # Backfill static stats for any gaps
        if player_stats_static is not None:
            # Only bring in columns we don't have yet
            cols_to_use = [c for c in player_stats_static.columns 
                           if c not in features_df.columns or c == Cols.PLAYER_ID]
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on=Cols.PLAYER_ID, how='left')

    else:
        logging.warning("No box scores found. Falling back to static stats only.")
        features_df = pd.merge(props_df, player_stats_static, on=Cols.PLAYER_ID, how='left')

    # 4. Merge Team and Opponent Stats
    # Standardize Team column for merge (ETL produces master_team_stats with 'TEAM_ABBREVIATION')
    if 'TEAM_ABBREVIATION' not in features_df.columns and Cols.TEAM in features_df.columns:
        features_df['TEAM_ABBREVIATION'] = features_df[Cols.TEAM]
        
    if team_stats is not None:
        # Team Stats
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        if 'TEAM_TEAM_ABBREVIATION' in team_stats_renamed.columns:
             # Fix if index was reset weirdly
             team_stats_renamed = team_stats_renamed.rename(columns={'TEAM_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})
             
        features_df = pd.merge(features_df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        # Opponent Stats
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        features_df = pd.merge(features_df, opp_stats_renamed, left_on=Cols.OPPONENT, right_index=True, how='left')

    # 5. Merge DVP
    if dvp_df is not None:
        # Get Position from Static Stats if not in features
        if 'Pos' not in features_df.columns and player_stats_static is not None:
             if Cols.PLAYER_ID in player_stats_static.columns:
                 pos_map = player_stats_static.set_index(Cols.PLAYER_ID)['Pos'].to_dict()
                 features_df['Pos'] = features_df[Cols.PLAYER_ID].map(pos_map).fillna('PG')

        def normalize_pos(p):
            p = str(p).split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p if p in ['PG','SG','SF','PF','C'] else 'PG'
            
        features_df['Primary_Pos'] = features_df.get('Pos', 'PG').apply(normalize_pos)
        
        # Ensure String Type for Parquet compatibility
        features_df['Primary_Pos'] = features_df['Primary_Pos'].astype(str)
        dvp_df['Primary_Pos'] = dvp_df['Primary_Pos'].astype(str)

        # Strict Merge (Matches dataset.py logic)
        features_df = pd.merge(
            features_df, 
            dvp_df, 
            left_on=[Cols.OPPONENT, 'Primary_Pos'], 
            right_on=['OPPONENT_ABBREV', 'Primary_Pos'], 
            how='left'
        )

    # 6. Merge H2H
    if not vs_opp_df.empty:
        features_df = pd.merge(
            features_df,
            vs_opp_df,
            left_on=[Cols.PLAYER_ID, Cols.OPPONENT],
            right_on=[Cols.PLAYER_ID, 'OPPONENT_ABBREV'],
            how='left'
        )

    # 7. Final Polish
    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['GAME_PACE'] = features_df['TEAM_Possessions per Game']
        
    # Vacancy Flags (Set to 0 if not calculated live)
    cols_to_fill = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_df