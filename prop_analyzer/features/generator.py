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
    """
    if Cols.PLAYER_ID not in df.columns or Cols.DATE not in df.columns:
        logging.error(f"Missing ID/Date columns. Cols found: {df.columns}")
        return df

    df = df.sort_values(by=[Cols.PLAYER_ID, Cols.DATE]).reset_index(drop=True)
    
    if stats_to_roll is None:
        stats_to_roll = [
            'PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 
            'FG3M', 'STL', 'BLK', 'TOV', 'FANTASY_PTS', 
            'FGA', 'FG3A', 'DD', 'TD', 
            'FTM', 'OREB' 
        ]
        
    # Ensure stats exist
    for col in stats_to_roll:
        if col not in df.columns: df[col] = 0.0

    grouped = df.groupby(Cols.PLAYER_ID)

    for col in stats_to_roll:
        # SZN Avg (Expanding Mean)
        df[f'{col}_{Cols.SZN_AVG}'] = grouped[col].expanding().mean().values
        
        # L5 Avg (Rolling 5)
        df[f'{col}_{Cols.L5_AVG}'] = grouped[col].rolling(window=5, min_periods=1).mean().values
        
        # L10 Std Dev
        df[f'{col}_L10_STD'] = grouped[col].rolling(window=10, min_periods=3).std().values
        
        # EWMA
        df[f'{col}_L5_EWMA'] = grouped[col].ewm(alpha=0.15, adjust=False).mean().values

    # Advanced Stats (Only if present)
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
    q1_history = loader.load_master_q1_history()
    h1_history = loader.load_master_1h_history() # <--- NEW LOAD
    
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
            name_map = player_stats_static.set_index('clean_name')[Cols.PLAYER_ID].to_dict()
            props_df['clean_name'] = props_df[Cols.PLAYER_NAME].apply(lambda x: str(x).lower().strip())
            
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
            
            props_df = props_df.dropna(subset=[Cols.PLAYER_ID]).copy()
            if props_df.empty: return pd.DataFrame()

            props_df[Cols.PLAYER_ID] = props_df[Cols.PLAYER_ID].astype('int64')
        else:
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

        history_df = add_rolling_stats_history(box_scores.copy())
        
        props_df[Cols.DATE] = pd.to_datetime(props_df[Cols.DATE])
        history_df[Cols.DATE] = pd.to_datetime(history_df[Cols.DATE])
        
        props_df = props_df.sort_values(Cols.DATE)
        history_df = history_df.sort_values(Cols.DATE)
        
        features_df = pd.merge_asof(
            props_df, history_df, on=Cols.DATE, by=Cols.PLAYER_ID,
            direction='backward', suffixes=('', '_hist')
        )
        
        if player_stats_static is not None:
            cols_to_use = [c for c in player_stats_static.columns 
                           if c not in features_df.columns or c == Cols.PLAYER_ID]
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on=Cols.PLAYER_ID, how='left')
    else:
        features_df = pd.merge(props_df, player_stats_static, on=Cols.PLAYER_ID, how='left')

    # 4. Time-Travel Feature Engineering (Q1)
    if not q1_history.empty:
        logging.info("Calculating Q1 rolling stats...")
        q1_history[Cols.PLAYER_ID] = q1_history[Cols.PLAYER_ID].fillna(0).astype('int64')
        if Cols.DATE in q1_history.columns:
            q1_history[Cols.DATE] = pd.to_datetime(q1_history[Cols.DATE])
        
        # --- Calculate Combo Stats for Q1 ---
        if 'PRA' not in q1_history.columns and 'PTS' in q1_history.columns:
             q1_history['PRA'] = q1_history['PTS'] + q1_history['REB'] + q1_history['AST']
        if 'PR' not in q1_history.columns and 'PTS' in q1_history.columns:
             q1_history['PR'] = q1_history['PTS'] + q1_history['REB']
        if 'PA' not in q1_history.columns and 'PTS' in q1_history.columns:
             q1_history['PA'] = q1_history['PTS'] + q1_history['AST']
        if 'RA' not in q1_history.columns and 'REB' in q1_history.columns:
             q1_history['RA'] = q1_history['REB'] + q1_history['AST']
        
        q1_rolled = add_rolling_stats_history(
            q1_history.copy(), 
            stats_to_roll=['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA']
        )
        
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

    # 5. Time-Travel Feature Engineering (1H) - NEW BLOCK
    if not h1_history.empty:
        logging.info("Calculating 1H rolling stats...")
        h1_history[Cols.PLAYER_ID] = h1_history[Cols.PLAYER_ID].fillna(0).astype('int64')
        if Cols.DATE in h1_history.columns:
            h1_history[Cols.DATE] = pd.to_datetime(h1_history[Cols.DATE])
            
        # --- Calculate Combo Stats for 1H ---
        if 'PRA' not in h1_history.columns and 'PTS' in h1_history.columns:
             h1_history['PRA'] = h1_history['PTS'] + h1_history['REB'] + h1_history['AST']
        if 'PR' not in h1_history.columns and 'PTS' in h1_history.columns:
             h1_history['PR'] = h1_history['PTS'] + h1_history['REB']
        if 'PA' not in h1_history.columns and 'PTS' in h1_history.columns:
             h1_history['PA'] = h1_history['PTS'] + h1_history['AST']
        if 'RA' not in h1_history.columns and 'REB' in h1_history.columns:
             h1_history['RA'] = h1_history['REB'] + h1_history['AST']
             
        h1_rolled = add_rolling_stats_history(
            h1_history.copy(), 
            stats_to_roll=['PTS', 'REB', 'AST', 'FG3M', 'PRA', 'PR', 'PA', 'RA']
        )
        
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
        features_df = pd.merge(features_df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        features_df = pd.merge(features_df, opp_stats_renamed, left_on=Cols.OPPONENT, right_index=True, how='left')

    # 7. Merge DVP
    if dvp_df is not None:
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
        dvp_df['Primary_Pos'] = dvp_df['Primary_Pos'].astype(str)

        features_df = pd.merge(
            features_df, dvp_df, 
            left_on=[Cols.OPPONENT, 'Primary_Pos'], 
            right_on=['OPPONENT_ABBREV', 'Primary_Pos'], 
            how='left'
        )

    # 8. Merge H2H
    if not vs_opp_df.empty:
        features_df = pd.merge(
            features_df, vs_opp_df,
            left_on=[Cols.PLAYER_ID, Cols.OPPONENT],
            right_on=[Cols.PLAYER_ID, 'OPPONENT_ABBREV'],
            how='left'
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