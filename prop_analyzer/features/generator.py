import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader

def add_rolling_stats_history(df, stats_to_roll=None):
    """
    Calculates historical rolling features on a dataset (Full Game, Q1, or 1H).
    CRITICAL FIX: All rolling stats are shifted by 1 to represent "stats entering the game".
    """
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns. Cols found: {df.columns}")
        return df

    # Ensure strictly sorted by Player then Date for correct shifting
    df = df.sort_values(by=[Cols.PLAYER_ID, Cols.DATE]).reset_index(drop=True)
    
    if stats_to_roll is None:
        stats_to_roll = [
            'PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 
            'FG3M', 'STL', 'BLK', 'TOV', 'FANTASY_PTS', 
            'FGA', 'FG3A', 'DD', 'TD', 
            'FTM', 'OREB' 
        ]
        
    # Ensure stats exist (fill missing with 0 to prevent errors)
    for col in stats_to_roll:
        if col not in df.columns: 
            df[col] = 0.0

    grouped = df.groupby(Cols.PLAYER_ID)

    # --- CRITICAL UPDATE: Minutes Volatility ---
    if 'MIN' in df.columns:
        # Ensure MIN is numeric
        if df['MIN'].dtype == 'object':
             try:
                 df['MIN'] = pd.to_numeric(df['MIN'], errors='coerce').fillna(0.0)
             except:
                 pass

        # 1. Season Average Minutes (Entering Game)
        df['MIN_SZN_AVG'] = grouped['MIN'].expanding().mean().shift(1).values
        
        # 2. Last 5 Average Minutes (Recent Role)
        df['MIN_L5_AVG'] = grouped['MIN'].rolling(window=5, min_periods=1).mean().shift(1).values
        
        # 3. Minutes Volatility (Standard Deviation of last 5 games)
        # We fill NaNs with 8.0 (high volatility) to penalize players with < 2 games history
        df['MIN_L5_STD'] = grouped['MIN'].rolling(window=5, min_periods=2).std().shift(1).fillna(8.0).values

    for col in stats_to_roll:
        # SZN Avg (Expanding Mean)
        df[f'{col}_{Cols.SZN_AVG}'] = grouped[col].expanding().mean().shift(1).values
        
        # L5 Avg (Rolling 5)
        df[f'{col}_{Cols.L5_AVG}'] = grouped[col].rolling(window=5, min_periods=1).mean().shift(1).values
        
        # L10 Std Dev
        df[f'{col}_L10_STD'] = grouped[col].rolling(window=10, min_periods=3).std().shift(1).values
        
        # EWMA (Exponential Weighted Moving Average)
        df[f'{col}_L5_EWMA'] = grouped[col].ewm(alpha=0.15, adjust=False).mean().shift(1).values

    # Advanced Stats (Only if present)
    if 'USG_PROXY' in df.columns:
        df['SZN_USG_PROXY'] = grouped['USG_PROXY'].expanding().mean().shift(1).values
        df['L5_USG_PROXY'] = grouped['USG_PROXY'].rolling(window=5).mean().shift(1).values
        
    if 'TS_PCT' in df.columns:
        df['SZN_TS_PCT'] = grouped['TS_PCT'].expanding().mean().shift(1).values
        
    return df

def generate_features(props_df):
    """
    Main entry point for generating features for today's props.
    """
    logging.info("Building feature set with Point-in-Time safety (Leakage Fixed)...")
    
    # 1. Load Data
    box_scores = loader.load_box_scores()
    q1_history = loader.load_master_q1_history()
    h1_history = loader.load_master_1h_history()
    
    player_stats_static, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    
    dvp_df = None
    if cfg.MASTER_DVP_FILE.exists():
        try:
            dvp_df = pd.read_parquet(cfg.MASTER_DVP_FILE)
        except Exception as e:
            logging.error(f"Failed to read DVP Parquet: {e}")
            dvp_df = None

    # 2. Map Player Names to IDs
    if Cols.PLAYER_ID not in props_df.columns:
        if player_stats_static is not None:
            # Create cleaner name map
            name_map = player_stats_static.set_index('clean_name')[Cols.PLAYER_ID].to_dict()
            props_df['clean_name'] = props_df[Cols.PLAYER_NAME].apply(lambda x: str(x).lower().strip())
            
            # Manual Mapping overrides
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
                'michael porter': 'michael porter jr.',
                'nick richards': 'nick richards', 
                'gg jackson': 'gg jackson ii'
            }
            props_df['clean_name'] = props_df['clean_name'].replace(manual_map)
            props_df[Cols.PLAYER_ID] = props_df['clean_name'].map(name_map)
            
            props_df = props_df.dropna(subset=[Cols.PLAYER_ID]).copy()
            if props_df.empty: 
                logging.warning("No players matched ID map. Check naming conventions.")
                return pd.DataFrame()

            props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        else:
            logging.error("Static player stats missing. Cannot map IDs.")
            return pd.DataFrame()

    # 3. Time-Travel Feature Engineering (Full Game)
    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating Full Game rolling stats...")
        
        box_scores[Cols.PLAYER_ID] = box_scores[Cols.PLAYER_ID].fillna(0).astype('int64')
        props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        
        if Cols.DATE in box_scores.columns:
            box_scores[Cols.DATE] = pd.to_datetime(box_scores[Cols.DATE])
        elif 'GAME_DATE' in box_scores.columns:
             box_scores[Cols.DATE] = pd.to_datetime(box_scores['GAME_DATE'])

        # Calculate history with shifts
        history_df = add_rolling_stats_history(box_scores.copy())
        
        props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE])
        history_df[Cols.DATE] = pd.to_datetime(history_df[Cols.DATE])
        
        props_df = props_df.sort_values(Cols.DATE)
        history_df = history_df.sort_values(Cols.DATE)
        
        # Merge point-in-time stats
        features_df = pd.merge_asof(
            props_df, history_df, on=Cols.DATE, by=Cols.PLAYER_ID,
            direction='backward', suffixes=('', '_hist')
        )
        
        if player_stats_static is not None:
            # CLEAN MERGE: Drop columns that already exist in features_df from the right side
            cols_to_use = [c for c in player_stats_static.columns 
                           if c not in features_df.columns or c == Cols.PLAYER_ID]
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on=Cols.PLAYER_ID, how='left')
    else:
        features_df = pd.merge(props_df, player_stats_static, on=Cols.PLAYER_ID, how='left')

    # 4. Time-Travel Feature Engineering (Q1)
    if q1_history is not None and not q1_history.empty:
        logging.info("Calculating Q1 rolling stats...")
        q1_history[Cols.PLAYER_ID] = q1_history[Cols.PLAYER_ID].fillna(0).astype('int64')
        if Cols.DATE in q1_history.columns:
            q1_history[Cols.DATE] = pd.to_datetime(q1_history[Cols.DATE])
        
        # Ensure Combo Stats
        for base, combo in [('PTS', 'PRA'), ('PTS', 'PR'), ('PTS', 'PA'), ('REB', 'RA')]:
            if combo not in q1_history.columns and base in q1_history.columns:
                q1_history[combo] = 0 
                
        # Recalculate combos for safety if components exist
        if {'PTS','REB','AST'}.issubset(q1_history.columns):
            q1_history['PRA'] = q1_history['PTS'] + q1_history['REB'] + q1_history['AST']
            q1_history['PR'] = q1_history['PTS'] + q1_history['REB']
            q1_history['PA'] = q1_history['PTS'] + q1_history['AST']
            q1_history['RA'] = q1_history['REB'] + q1_history['AST']
        
        q1_rolled = add_rolling_stats_history(
            q1_history.copy(), 
            stats_to_roll=['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA']
        )
        
        # Rename Q1 columns
        cols_to_rename = {}
        for col in q1_rolled.columns:
            if '_SZN_' in col or '_L5_' in col or '_L10_' in col:
                cols_to_rename[col] = f"Q1_{col}"
        q1_rolled.rename(columns=cols_to_rename, inplace=True)
        
        q1_rolled = q1_rolled.sort_values(Cols.DATE)
        features_df = features_df.sort_values(Cols.DATE)
        
        q1_feats_only = q1_rolled[[Cols.PLAYER_ID, Cols.DATE] + list(cols_to_rename.values())]
        
        features_df = pd.merge_asof(
            features_df, q1_feats_only,
            on=Cols.DATE, by=Cols.PLAYER_ID,
            direction='backward'
        )

    # 5. Time-Travel Feature Engineering (1H)
    if h1_history is not None and not h1_history.empty:
        logging.info("Calculating 1H rolling stats...")
        h1_history[Cols.PLAYER_ID] = h1_history[Cols.PLAYER_ID].fillna(0).astype('int64')
        if Cols.DATE in h1_history.columns:
            h1_history[Cols.DATE] = pd.to_datetime(h1_history[Cols.DATE])
            
        # Recalculate combos
        if {'PTS','REB','AST'}.issubset(h1_history.columns):
            h1_history['PRA'] = h1_history['PTS'] + h1_history['REB'] + h1_history['AST']
            h1_history['PR'] = h1_history['PTS'] + h1_history['REB']
            h1_history['PA'] = h1_history['PTS'] + h1_history['AST']
            h1_history['RA'] = h1_history['REB'] + h1_history['AST']
             
        h1_rolled = add_rolling_stats_history(
            h1_history.copy(), 
            stats_to_roll=['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA']
        )
        
        # Rename 1H columns
        cols_to_rename = {}
        for col in h1_rolled.columns:
            if '_SZN_' in col or '_L5_' in col or '_L10_' in col:
                cols_to_rename[col] = f"1H_{col}"
        h1_rolled.rename(columns=cols_to_rename, inplace=True)
        
        h1_rolled = h1_rolled.sort_values(Cols.DATE)
        features_df = features_df.sort_values(Cols.DATE)
        
        h1_feats = h1_rolled[[Cols.PLAYER_ID, Cols.DATE] + list(cols_to_rename.values())]
        
        features_df = pd.merge_asof(
            features_df, h1_feats,
            on=Cols.DATE, by=Cols.PLAYER_ID,
            direction='backward'
        )

    # 6. Merge Team/Opponent
    if 'TEAM_ABBREVIATION' not in features_df.columns and Cols.TEAM in features_df.columns:
        features_df['TEAM_ABBREVIATION'] = features_df[Cols.TEAM]
        
    if team_stats is not None:
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        if 'TEAM_TEAM_ABBREVIATION' in team_stats_renamed.columns:
             team_stats_renamed = team_stats_renamed.rename(columns={'TEAM_TEAM_ABBREVIATION': 'TEAM_ABBREVIATION'})
        
        # Filter duplicates
        cols_team = [c for c in team_stats_renamed.columns if c not in features_df.columns or c == 'TEAM_ABBREVIATION']
        if 'TEAM_ABBREVIATION' in cols_team:
             features_df = pd.merge(features_df, team_stats_renamed[cols_team], on='TEAM_ABBREVIATION', how='left')
        else:
             features_df = pd.merge(features_df, team_stats_renamed[cols_team], left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        # Filter duplicates for Opponent
        # Note: We merge on OPPONENT column
        if 'OPP_TEAM_ABBREVIATION' in opp_stats_renamed.columns:
             opp_stats_renamed = opp_stats_renamed.rename(columns={'OPP_TEAM_ABBREVIATION': 'OPP_ABBREV'})
        
        cols_opp = [c for c in opp_stats_renamed.columns if c not in features_df.columns]
        
        # Check if we can merge
        features_df = pd.merge(features_df, opp_stats_renamed[cols_opp], left_on=Cols.OPPONENT, right_index=True, how='left')

    # 7. Merge DVP
    if dvp_df is not None:
        # Standardize Position
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
        features_df['Primary_Pos'] = features_df['Primary_Pos'].astype(str)
        
        if 'Primary_Pos' in dvp_df.columns:
            dvp_df['Primary_Pos'] = dvp_df['Primary_Pos'].astype(str)
            
            # CLEAN MERGE: Drop potential duplicates from DVP (like ID or Name if they exist)
            cols_to_use_dvp = [c for c in dvp_df.columns if c not in features_df.columns or c in ['OPPONENT_ABBREV', 'Primary_Pos']]
            
            features_df = pd.merge(
                features_df, dvp_df[cols_to_use_dvp], 
                left_on=[Cols.OPPONENT, 'Primary_Pos'], 
                right_on=['OPPONENT_ABBREV', 'Primary_Pos'], 
                how='left'
            )

    # 8. Merge H2H
    if vs_opp_df is not None and not vs_opp_df.empty:
        # CLEAN MERGE: Exclude Redundant Columns from vs_opp_df
        # We only want the STATS, not the Player Name or Team or Position again.
        
        # Identify columns to drop (identifiers already in features_df)
        drop_cols = [
            'PLAYER_NAME', 'Player Name', 'TEAM_ABBREVIATION', 'TEAM', 'Team', 
            'MATCHUP', 'Matchup', 'GAME_DATE', 'DATE'
        ]
        
        # Filter vs_opp_df columns to only keep what we need + keys
        right_cols = [c for c in vs_opp_df.columns if c not in drop_cols]
        # Ensure we keep the join keys
        if Cols.PLAYER_ID not in right_cols: right_cols.append(Cols.PLAYER_ID)
        if 'OPPONENT_ABBREV' not in right_cols and 'OPPONENT_ABBREV' in vs_opp_df.columns: right_cols.append('OPPONENT_ABBREV')
        
        features_df = pd.merge(
            features_df, vs_opp_df[right_cols],
            left_on=[Cols.PLAYER_ID, Cols.OPPONENT],
            right_on=[Cols.PLAYER_ID, 'OPPONENT_ABBREV'],
            how='left',
            suffixes=('', '_h2h') # Add explicit suffix just in case stats overlap
        )

    # 9. Final Polish
    if 'TEAM_Possessions per Game' in features_df.columns:
        features_df['GAME_PACE'] = features_df['TEAM_Possessions per Game']
        
    cols_to_fill = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in cols_to_fill:
        if c not in features_df.columns: features_df[c] = 0.0
        features_df[c] = features_df[c].fillna(0.0)

    logging.info(f"Feature set built. Final Shape: {features_df.shape}")
    return features_df