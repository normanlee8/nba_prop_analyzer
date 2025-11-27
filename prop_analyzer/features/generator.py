import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader

def get_latest_season_stats(df):
    """
    Filters the master dataframe to keep only the latest season row for each player.
    """
    if 'SEASON_ID' not in df.columns:
        return df
    
    # Sort by Player and Season (descending) so the latest season is first
    df = df.sort_values(by=['PLAYER_ID', 'SEASON_ID'], ascending=[True, False])
    
    # Drop duplicates, keeping the first (latest) one
    df_latest = df.drop_duplicates(subset=['PLAYER_ID'], keep='first').copy()
    
    return df_latest

def build_feature_set(props_df):
    """
    Constructs the feature matrix (X) for a list of props.
    """
    logging.info("Building feature set for inference...")
    
    # 1. Load Static Data
    player_stats, team_stats, league_pace = loader.load_static_data()
    
    if player_stats is None or player_stats.empty:
        logging.critical("Master Player Stats is empty. Cannot build features.")
        return pd.DataFrame()

    # --- CRITICAL FIX: Filter for Latest Season Only ---
    # We only want 2025-26 stats for today's predictions, not 2024-25
    player_stats = get_latest_season_stats(player_stats)
    logging.info(f"Filtered player stats to latest season. Active players: {len(player_stats)}")

    # 2. Load Vs Opponent Data
    vs_opp_df = loader.load_vs_opponent_data()
    
    # 3. Load DVP Data (Defense vs Position)
    dvp_df = None
    if (cfg.DATA_DIR / "master_dvp_stats.csv").exists():
        dvp_df = pd.read_csv(cfg.DATA_DIR / "master_dvp_stats.csv")

    # 4. Merge Data onto Props
    # Map text names to IDs if needed
    if 'PLAYER_ID' not in props_df.columns:
        # Simple map using the player_stats file
        name_map = player_stats.set_index('clean_name')['PLAYER_ID'].to_dict()
        props_df['clean_name'] = props_df['Player Name'].apply(lambda x: str(x).lower().strip())
        props_df['PLAYER_ID'] = props_df['clean_name'].map(name_map)

    # Merge Base Player Stats
    # We use inner merge? No, left, to keep props even if stats missing (we handle missing later)
    df = pd.merge(props_df, player_stats, on='PLAYER_ID', how='left', suffixes=('', '_szn'))
    
    # Merge Team Stats (Implied Team Totals, Pace, etc)
    if team_stats is not None and 'TEAM_ABBREVIATION' in df.columns:
        team_stats_renamed = team_stats.add_prefix('TEAM_')
        df = pd.merge(df, team_stats_renamed, left_on='TEAM_ABBREVIATION', right_index=True, how='left')
        
    # Merge Opponent Stats (Defensive metrics)
    if team_stats is not None and 'Opponent' in df.columns:
        opp_stats_renamed = team_stats.add_prefix('OPP_')
        df = pd.merge(df, opp_stats_renamed, left_on='Opponent', right_index=True, how='left')

    # Merge DVP
    if dvp_df is not None and 'Opponent' in df.columns and 'Pos' in df.columns:
        # We need to normalize position again just like in ETL
        def normalize_pos(p):
            p = str(p).split('-')[0].upper().strip()
            if p == 'G': return 'SG'
            if p == 'F': return 'PF'
            return p if p in ['PG','SG','SF','PF','C'] else 'PG'
            
        df['Primary_Pos'] = df['Pos'].apply(normalize_pos)
        
        # Merge on Opponent + Position
        # Note: DVP file columns are [OPPONENT_ABBREV, Primary_Pos, DVP_PTS, ...]
        df = pd.merge(
            df, 
            dvp_df, 
            left_on=['Opponent', 'Primary_Pos'], 
            right_on=['OPPONENT_ABBREV', 'Primary_Pos'], 
            how='left'
        )

    # Merge VS Opponent History (H2H)
    if not vs_opp_df.empty:
        # Rename cols to avoid collision
        df = pd.merge(
            df,
            vs_opp_df,
            left_on=['PLAYER_ID', 'Opponent'],
            right_on=['PLAYER_ID', 'OPPONENT_ABBREV'],
            how='left'
        )

    # 5. Feature Engineering / Calculation
    # Calculate derived features (e.g. implied totals)
    if 'TEAM_Possessions per Game' in df.columns:
        df['GAME_PACE'] = df['TEAM_Possessions per Game'] # Simplified pace model
        
    # Vacancy Filling
    cols_to_fill = ['TEAM_MISSING_USG', 'TEAM_MISSING_MIN', 'MISSING_USG_G', 'MISSING_USG_F']
    for c in cols_to_fill:
        if c not in df.columns: df[c] = 0.0
        df[c] = df[c].fillna(0.0)

    logging.info(f"Feature set built. Shape: {df.shape}")
    return df