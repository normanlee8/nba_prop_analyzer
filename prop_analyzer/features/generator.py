import pandas as pd
import numpy as np
import logging
from prop_analyzer import config as cfg
from prop_analyzer.features import definitions as feat_defs
from prop_analyzer.data import loader

def add_rolling_stats_history(df):
    """
    Calculates historical rolling features on the full box score dataset.
    """
    df = df.sort_values(by=['PLAYER_ID', 'GAME_DATE']).reset_index(drop=True)
    
    stats_to_roll = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'FG3M', 'STL', 'BLK', 'TOV', 'FANTASY_PTS']
    for col in stats_to_roll:
        if col not in df.columns: df[col] = 0.0

    grouped = df.groupby('PLAYER_ID')

    for col in stats_to_roll:
        df[f'{col}_SZN_AVG'] = grouped[col].expanding().mean().values
        df[f'{col}_L5_AVG'] = grouped[col].rolling(window=5, min_periods=1).mean().values
        df[f'{col}_L10_STD'] = grouped[col].rolling(window=10, min_periods=3).std().values
        df[f'{col}_L5_EWMA'] = grouped[col].ewm(alpha=0.15, adjust=False).mean().values

    if 'USG_PROXY' in df.columns:
        df['SZN_USG_PROXY'] = grouped['USG_PROXY'].expanding().mean().values
        df['L5_USG_PROXY'] = grouped['USG_PROXY'].rolling(window=5).mean().values
        
    return df

def build_feature_set(props_df):
    logging.info("Building feature set with Point-in-Time safety...")
    
    # 1. Load Data
    box_scores = loader.load_box_scores()
    player_stats_static, team_stats, league_pace = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    
    dvp_df = None
    if cfg.MASTER_DVP_FILE.exists():
        dvp_df = pd.read_csv(cfg.MASTER_DVP_FILE)

    # 2. Map Player Names to IDs
    if 'PLAYER_ID' not in props_df.columns:
        if player_stats_static is not None:
            name_map = player_stats_static.set_index('clean_name')['PLAYER_ID'].to_dict()
            props_df['clean_name'] = props_df['Player Name'].apply(lambda x: str(x).lower().strip())
            
            # --- FIX: Manual Name Mapping ---
            manual_map = {
                'deuce mcbride': 'miles mcbride',
                'cam johnson': 'cameron johnson',
                'lu dort': 'luguentz dort',
                'pj washington': 'p.j. washington',
                'jimmy butler': 'jimmy butler iii' 
            }
            props_df['clean_name'] = props_df['clean_name'].replace(manual_map)
            
            props_df['PLAYER_ID'] = props_df['clean_name'].map(name_map)
            
            missing_ids = props_df[props_df['PLAYER_ID'].isna()]
            if not missing_ids.empty:
                logging.warning(f"Dropping {len(missing_ids)} props due to unrecognized player names: {missing_ids['Player Name'].unique()}")
                props_df = props_df.dropna(subset=['PLAYER_ID']).copy()
            
            if props_df.empty:
                logging.error("No valid props remaining after name mapping.")
                return pd.DataFrame()

            props_df['PLAYER_ID'] = props_df['PLAYER_ID'].astype('int64')
        else:
            logging.error("Cannot map names: Player stats file missing.")
            return pd.DataFrame()

    # 3. Time-Travel Feature Engineering
    if box_scores is not None and not box_scores.empty:
        logging.info("Calculating dynamic historical stats...")
        
        box_scores['PLAYER_ID'] = box_scores['PLAYER_ID'].fillna(0).astype('int64')
        props_df['PLAYER_ID'] = props_df['PLAYER_ID'].fillna(0).astype('int64')

        history_df = add_rolling_stats_history(box_scores.copy())
        
        props_df['GAME_DATE'] = pd.to_datetime(props_df['GAME_DATE'])
        history_df['GAME_DATE'] = pd.to_datetime(history_df['GAME_DATE'])
        
        props_df = props_df.sort_values('GAME_DATE')
        history_df = history_df.sort_values('GAME_DATE')
        
        features_df = pd.merge_asof(
            props_df,
            history_df,
            on='GAME_DATE',
            by='PLAYER_ID',
            direction='backward',
            allow_exact_matches=False,
            suffixes=('', '_hist')
        )
        
        # --- FIX: ROBUST REPAIR & BACKFILL ---
        if player_stats_static is not None:
            cols_to_use = [c for c in player_stats_static.columns if c not in features_df.columns or c == 'PLAYER_ID']
            features_df = pd.merge(features_df, player_stats_static[cols_to_use], on='PLAYER_ID', how='left')
            
            # A. REPAIR: Calculate Missing Season Totals from Home/Away Splits
            if 'SEASON_G' in features_df.columns:
                features_df['SEASON_G'] = features_df['SEASON_G'].fillna(
                    features_df.get('HOME_GP', 0) + features_df.get('AWAY_GP', 0)
                ).replace(0, 1)

            metrics_map = {
                'PTS': 'SEASON_PTS', 'REB': 'SEASON_TRB', 'AST': 'SEASON_AST',
                'STL': 'SEASON_STL', 'BLK': 'SEASON_BLK', 'FG3M': 'SEASON_FG3M',
                'TOV': 'SEASON_TOV'
            }
            
            if 'SEASON_TOV' not in features_df.columns:
                features_df['SEASON_TOV'] = features_df.get('HOME_TOV', 0) + features_df.get('AWAY_TOV', 0)

            for short_name, col_name in metrics_map.items():
                if col_name in features_df.columns:
                    home_col = f'HOME_{short_name}'
                    away_col = f'AWAY_{short_name}'
                    if home_col in features_df.columns and away_col in features_df.columns:
                        features_df[col_name] = features_df[col_name].fillna(
                            features_df[home_col].fillna(0) + features_df[away_col].fillna(0)
                        )

            # B. BACKFILL: Main Rolling Stats
            for short_name, static_col in metrics_map.items():
                rolling_col = f'{short_name}_SZN_AVG'
                if static_col in features_df.columns:
                    if rolling_col not in features_df.columns: features_df[rolling_col] = 0.0
                    
                    avg_val = features_df[static_col] / features_df['SEASON_G']
                    features_df[rolling_col] = features_df[rolling_col].fillna(avg_val)
                    
                    features_df[rolling_col] = features_df.apply(
                        lambda r: (r[static_col] / r['SEASON_G']) if r[rolling_col] == 0.0 else r[rolling_col], 
                        axis=1
                    )

            # C. BACKFILL: Derived Full Game Stats (Fixes 'float' attribute error)
            if 'PTS_SZN_AVG' in features_df.columns:
                p = features_df['PTS_SZN_AVG']
                r = features_df['REB_SZN_AVG']
                a = features_df['AST_SZN_AVG']
                s = features_df.get('STL_SZN_AVG', 0)
                b = features_df.get('BLK_SZN_AVG', 0)
                t = features_df.get('TOV_SZN_AVG', 0)
                
                # --- FIX: Explicit checks to avoid .fillna() on non-Series ---
                
                # PRA
                calc_pra = p + r + a
                if 'PRA_SZN_AVG' in features_df.columns:
                    features_df['PRA_SZN_AVG'] = features_df['PRA_SZN_AVG'].replace(0.0, np.nan).fillna(calc_pra)
                else:
                    features_df['PRA_SZN_AVG'] = calc_pra
                
                # PR
                calc_pr = p + r
                if 'PR_SZN_AVG' in features_df.columns:
                    features_df['PR_SZN_AVG'] = features_df['PR_SZN_AVG'].replace(0.0, np.nan).fillna(calc_pr)
                else:
                    features_df['PR_SZN_AVG'] = calc_pr

                # PA
                calc_pa = p + a
                if 'PA_SZN_AVG' in features_df.columns:
                    features_df['PA_SZN_AVG'] = features_df['PA_SZN_AVG'].replace(0.0, np.nan).fillna(calc_pa)
                else:
                    features_df['PA_SZN_AVG'] = calc_pa

                # RA
                calc_ra = r + a
                if 'RA_SZN_AVG' in features_df.columns:
                    features_df['RA_SZN_AVG'] = features_df['RA_SZN_AVG'].replace(0.0, np.nan).fillna(calc_ra)
                else:
                    features_df['RA_SZN_AVG'] = calc_ra
                
                # FANTASY
                fantasy_calc = p + (1.2 * r) + (1.5 * a) + (3 * s) + (3 * b) - t
                if 'FANTASY_PTS_SZN_AVG' in features_df.columns:
                    features_df['FANTASY_PTS_SZN_AVG'] = features_df['FANTASY_PTS_SZN_AVG'].replace(0.0, np.nan).fillna(fantasy_calc)
                else:
                    features_df['FANTASY_PTS_SZN_AVG'] = fantasy_calc

            # D. BACKFILL: Quarter & Half Stats
            segments = ['Q1', 'Q2', 'Q3', 'Q4']
            base_metrics = ['PTS', 'REB', 'AST']
            
            for seg in segments:
                for met in base_metrics:
                    static_col = f"{seg}_{met}" 
                    target_col = f"{seg}_{met}_SZN_AVG"
                    
                    if static_col in features_df.columns:
                        if target_col not in features_df.columns: features_df[target_col] = np.nan
                        features_df[target_col] = features_df[target_col].fillna(features_df[static_col])

                # PRA for Quarters
                pra_target = f"{seg}_PRA_SZN_AVG"
                p_avg = features_df.get(f"{seg}_PTS_SZN_AVG", 0)
                r_avg = features_df.get(f"{seg}_REB_SZN_AVG", 0)
                a_avg = features_df.get(f"{seg}_AST_SZN_AVG", 0)
                
                # --- FIX: Check for column existence ---
                calc_q_pra = p_avg + r_avg + a_avg
                if pra_target in features_df.columns:
                    features_df[pra_target] = features_df[pra_target].fillna(calc_q_pra)
                else:
                    features_df[pra_target] = calc_q_pra

            # E. BACKFILL: Halves (Sum of Quarters)
            halves = {'1H': ['Q1', 'Q2'], '2H': ['Q3', 'Q4']}
            for half, q_list in halves.items():
                for met in ['PTS', 'REB', 'AST', 'PRA']:
                    target_col = f"{half}_{met}_SZN_AVG"
                    val = 0
                    for q in q_list:
                        val += features_df.get(f"{q}_{met}_SZN_AVG", 0)
                    
                    if target_col not in features_df.columns: features_df[target_col] = np.nan
                    features_df[target_col] = features_df[target_col].fillna(val)

        logging.info(f"Point-in-time merge complete. Rows: {len(features_df)}")
    else:
        logging.warning("No box scores found. Falling back to static stats.")
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