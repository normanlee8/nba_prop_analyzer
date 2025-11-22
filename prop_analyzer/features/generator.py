import pandas as pd
import numpy as np
from datetime import timedelta
from prop_analyzer.features import calculator as calc
from prop_analyzer.data import loader

def get_historical_vs_opponent_stats(box_scores_df, player_id, opponent_team_abbr, prop_game_date):
    stats_of_interest = [
        'PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV', 'PRA', 'PR', 'PA', 'RA', 'STK', 'FANTASY_PTS',
        'TS_PCT', 'EFG_PCT', 'USG_PROXY'
    ]
    hist_stats = {f"HIST_VS_OPP_{stat}_AVG": np.nan for stat in stats_of_interest}
    hist_stats["HIST_VS_OPP_GAMES"] = 0

    try:
        player_games_df = box_scores_df[box_scores_df['PLAYER_ID'] == player_id].copy()
        prop_date_dt = pd.to_datetime(prop_game_date).normalize()
        player_games_df = player_games_df[player_games_df['GAME_DATE'] < prop_date_dt]
        
        opponent_games_df = player_games_df[player_games_df['MATCHUP'].str.contains(f" {opponent_team_abbr}", na=False)]
        num_hist_games = len(opponent_games_df)
        hist_stats["HIST_VS_OPP_GAMES"] = num_hist_games

        if num_hist_games > 0:
            for stat in stats_of_interest:
                if stat in opponent_games_df.columns:
                    hist_stats[f"HIST_VS_OPP_{stat}_AVG"] = round(opponent_games_df[stat].mean(), 2)
        return hist_stats
    except: return hist_stats

def build_feature_vector(player_data, player_id, prop_category, prop_line, 
                         player_team_abbr, opponent_team_abbr, is_home, prop_game_date,
                         box_scores_df, team_stats_df, vs_opp_df, full_roster_df=None,
                         dvp_df=None): # <--- Added dvp_df argument
    
    # 1. Base Metrics
    player_metrics = calc.calculate_player_metrics(
        box_scores_df, player_id, prop_category, is_home, prop_game_date
    )
    
    # 2. Fatigue & Rest
    fatigue_metrics = calc.get_schedule_fatigue_metrics(player_id, box_scores_df, prop_game_date)
    days_rest = calc.determine_rest_factor(player_id, box_scores_df, prop_game_date)
    
    # 3. Team Data
    player_team_row = team_stats_df.loc[player_team_abbr] if player_team_abbr in team_stats_df.index else pd.Series()
    opponent_team_row = team_stats_df.loc[opponent_team_abbr] if opponent_team_abbr in team_stats_df.index else pd.Series()

    # 4. Vacancy Logic (Updated for Split)
    prop_date_dt = pd.to_datetime(prop_game_date).normalize()
    current_game_record = box_scores_df[(box_scores_df['PLAYER_ID'] == player_id) & (box_scores_df['GAME_DATE'] == prop_date_dt)]
    
    # Default values
    missing_usg, missing_min = 0.0, 0.0
    vac_g, vac_f = 0.0, 0.0

    if not current_game_record.empty and 'TEAM_MISSING_USG' in current_game_record.columns:
        # Historical Mode (Training)
        missing_usg = current_game_record.iloc[0]['TEAM_MISSING_USG']
        missing_min = current_game_record.iloc[0]['TEAM_MISSING_MIN']
        if 'MISSING_USG_G' in current_game_record.columns:
            vac_g = current_game_record.iloc[0]['MISSING_USG_G']
            vac_f = current_game_record.iloc[0]['MISSING_USG_F']
    else:
        # Live Mode (To be updated in calc, for now we use legacy calc fallback)
        inj_df = loader.get_cached_injury_data()
        missing_usg, missing_min = calc.calculate_live_vacancy(player_team_abbr, full_roster_df, inj_df)
        # Note: Live split vacancy not implemented in Phase 2 yet, defaulting to 0

    # 5. Feature Construction
    features = {
        'Prop Line': prop_line,
        'SZN Avg': player_metrics['szn_avg'],
        'L3 Avg': player_metrics['l3_avg'],   
        'L5 EWMA': player_metrics['l5_avg'],
        'Loc Avg': player_metrics.get('loc_avg', player_metrics['szn_avg']),
        'CoV %': player_metrics['cov_pct'],
        'SZN Games': player_metrics['season_games'],
        'Selected Std Dev': player_metrics['szn_std_dev'],
        'L10_STD_DEV': player_metrics.get('l10_std_dev', player_metrics['szn_std_dev']),
        'GAMES_IN_L5': fatigue_metrics['games_in_l5'],
        'IS_B2B': fatigue_metrics['is_b2b'],
        'Days Rest': days_rest,
        'TEAM_MISSING_USG': missing_usg,
        'TEAM_MISSING_MIN': missing_min,
        'MISSING_USG_G': vac_g, # New
        'MISSING_USG_F': vac_f, # New
        'SZN_TS_PCT': player_metrics['szn_avg_ts'], 'SZN_EFG_PCT': player_metrics['szn_avg_efg'], 'SZN_USG_PROXY': player_metrics['szn_avg_usg'],
        'L5_TS_PCT': player_metrics['l5_avg_ts'], 'L5_EFG_PCT': player_metrics['l5_avg_efg'], 'L5_USG_PROXY': player_metrics['l5_avg_usg'],
        'LOC_TS_PCT': player_metrics['loc_avg_ts'], 'LOC_EFG_PCT': player_metrics['loc_avg_efg'], 'LOC_USG_PROXY': player_metrics['loc_avg_usg'],
        'TS_DIFF': player_metrics['l5_avg_ts'] - player_metrics['szn_avg_ts'],
        'EFG_DIFF': player_metrics['l5_avg_efg'] - player_metrics['szn_avg_efg'],
        'USG_DIFF': player_metrics['l5_avg_usg'] - player_metrics['szn_avg_usg'],
        'PBP_OnCourt_PlusMinus': player_data.get('PBP_OnCourt_PlusMinus', 0.0),
        'PBP_OnCourt_USG_PCT': player_data.get('PBP_OnCourt_USG_PCT', 0.0),
        'PBP_OnCourt_TOV_PCT': player_data.get('PBP_OnCourt_TOV_PCT', 0.0),
        'PBP_OnCourt_3PAr': player_data.get('PBP_OnCourt_3PAr', 0.0),
        'SEASON_TS_PLUS': player_data.get('TS+', 0.0), 
        'SEASON_eFG_PLUS': player_data.get('eFG+', 0.0), 
        'SEASON_FGA_PCT_3P': player_data.get('%FGA 3P', 0.0), 
    }

    # 6. Inject Quarters
    for q in range(1, 5):
        for stat in ['PTS', 'REB', 'AST', 'MIN']:
            col = f"Q{q}_{stat}"
            features[col] = player_data.get(col, 0.0)

    # 7. Inject H2H (Vs Opponent)
    for col in ['VS_OPP_PTS', 'VS_OPP_REB', 'VS_OPP_AST', 'VS_OPP_STL', 'VS_OPP_BLK', 'VS_OPP_FG3M', 'VS_OPP_TOV', 'VS_OPP_PRA', 'VS_OPP_PR', 'VS_OPP_PA', 'VS_OPP_RA', 'VS_OPP_FANTASY_PTS', 'VS_OPP_MIN', 'VS_OPP_GAMES_PLAYED']:
        features[col] = np.nan

    if not vs_opp_df.empty:
        opp_stats = vs_opp_df[(vs_opp_df['PLAYER_ID'] == player_id) & (vs_opp_df['OPPONENT_ABBREV'] == opponent_team_abbr)]
        if not opp_stats.empty:
            row_stats = opp_stats.iloc[0]
            for col in opp_stats.columns:
                if col in features: features[col] = row_stats[col]

    # 8. Inject Team Stats
    if not player_team_row.empty:
        for col, val in player_team_row.items(): features[f'TEAM_{col}'] = val
    if not opponent_team_row.empty:
        for col, val in opponent_team_row.items(): features[f'OPP_{col}'] = val

    # 9. Inject DvP (Defense vs Position) - NEW
    if dvp_df is not None:
        # Get Player Position (Simple)
        pos = str(player_data.get('Pos', 'PG')).split('-')[0]
        if pos not in ['PG', 'SG', 'SF', 'PF', 'C']: pos = 'PG' # Fallback
        
        # Lookup
        dvp_row = dvp_df[(dvp_df['OPPONENT_ABBREV'] == opponent_team_abbr) & (dvp_df['Primary_Pos'] == pos)]
        
        if not dvp_row.empty:
            stats = dvp_row.iloc[0]
            # Inject all matching DVP columns
            for col in stats.index:
                if col.startswith('DVP_'):
                    features[col] = stats[col]

    # 10. Legacy History Fallback
    hist_vs_opp = get_historical_vs_opponent_stats(box_scores_df, player_id, opponent_team_abbr, prop_game_date)
    features.update(hist_vs_opp)

    return features, player_metrics, fatigue_metrics