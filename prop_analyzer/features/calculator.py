import pandas as pd
import numpy as np
from datetime import timedelta
from rapidfuzz import process, fuzz
from prop_analyzer import config as cfg
from prop_analyzer.utils.common import get_nba_season_id
from prop_analyzer.utils.text import preprocess_name_for_fuzzy_match

def calculate_player_metrics(box_scores_df, player_id, prop_category, is_home, prop_game_date):
    # Map sub-game props to full game props for trend analysis
    proxy_map = {
        'Q1_PTS': 'PTS', 'Q1_REB': 'REB', 'Q1_AST': 'AST', 'Q1_PRA': 'PRA',
        'Q2_PTS': 'PTS', 'Q2_REB': 'REB', 'Q2_AST': 'AST',
        'Q3_PTS': 'PTS', 'Q3_REB': 'REB', 'Q3_AST': 'AST',
        'Q4_PTS': 'PTS', 'Q4_REB': 'REB', 'Q4_AST': 'AST',
        '1H_PTS': 'PTS', '1H_REB': 'REB', '1H_AST': 'AST', '1H_PRA': 'PRA',
    }
    prop_col = proxy_map.get(prop_category, prop_category)

    # Filter for Player & Date
    player_history = box_scores_df[box_scores_df['PLAYER_ID'] == player_id].copy()
    prop_date_dt = pd.to_datetime(prop_game_date).normalize()
    player_history = player_history[player_history['GAME_DATE'] < prop_date_dt]
    
    # Season Isolation
    target_season_id = get_nba_season_id(prop_date_dt)
    if 'SEASON_ID' in player_history.columns:
        player_season_df = player_history[player_history['SEASON_ID'] == target_season_id].copy()
    else:
        season_start = prop_date_dt - timedelta(days=270)
        player_season_df = player_history[player_history['GAME_DATE'] > season_start].copy()

    defaults = {
        'szn_avg': 0, 'l3_avg': 0, 'l5_avg': 0, 'l10_std_dev': 0, 
        'szn_std_dev': 0, 'cov_pct': 0, 'season_games': 0,
        'szn_avg_ts': 0, 'l5_avg_ts': 0, 'loc_avg_ts': 0,
        'szn_avg_efg': 0, 'l5_avg_efg': 0, 'loc_avg_efg': 0,
        'szn_avg_usg': 0, 'l5_avg_usg': 0, 'loc_avg_usg': 0,
        'loc_avg': 0
    }

    if prop_col not in player_season_df.columns:
        return {'prop_col': prop_col} | defaults
    
    player_season_df.dropna(subset=[prop_col], inplace=True)
    player_season_df.sort_values(by='GAME_DATE', ascending=False, inplace=True)
    
    games_played = len(player_season_df)
    stats = {'prop_col': prop_col, 'season_games': games_played}
    
    if games_played == 0:
        return defaults | stats

    all_stats = player_season_df[prop_col].values

    # Bayesian Smoothed Season Avg
    sample_mean = np.mean(all_stats)
    n_prior = cfg.BAYESIAN_PRIOR_WEIGHT 
    prior_mean = cfg.BAYESIAN_PRIORS.get(prop_category, sample_mean) 
    stats['szn_avg'] = round(((n_prior * prior_mean) + (games_played * sample_mean)) / (n_prior + games_played), 2)
    
    # Advanced SZN Stats
    stats['szn_avg_ts'] = round(player_season_df['TS_PCT'].mean(), 4) if 'TS_PCT' in player_season_df else 0
    stats['szn_avg_efg'] = round(player_season_df['EFG_PCT'].mean(), 4) if 'EFG_PCT' in player_season_df else 0
    stats['szn_avg_usg'] = round(player_season_df['USG_PROXY'].mean(), 4) if 'USG_PROXY' in player_season_df else 0

    # Volatility
    stats['szn_std_dev'] = round(np.std(all_stats), 2)
    stats['cov_pct'] = round((stats['szn_std_dev'] / stats['szn_avg']) * 100, 1) if stats['szn_avg'] != 0 else 100.0

    # Location Splits
    loc_filter = 'vs.' if is_home else '@'
    loc_stats_df = player_season_df[player_season_df['MATCHUP'].str.contains(loc_filter, na=False)]
    stats['loc_avg'] = round(loc_stats_df[prop_col].mean(), 2) if not loc_stats_df.empty else stats['szn_avg']
    stats['loc_avg_ts'] = round(loc_stats_df['TS_PCT'].mean(), 4) if not loc_stats_df.empty and 'TS_PCT' in loc_stats_df else stats['szn_avg_ts']
    stats['loc_avg_efg'] = round(loc_stats_df['EFG_PCT'].mean(), 4) if not loc_stats_df.empty and 'EFG_PCT' in loc_stats_df else stats['szn_avg_efg']
    stats['loc_avg_usg'] = round(loc_stats_df['USG_PROXY'].mean(), 4) if not loc_stats_df.empty and 'USG_PROXY' in loc_stats_df else stats['szn_avg_usg']

    # Rolling Windows
    player_season_df_asc = player_season_df.sort_values(by='GAME_DATE', ascending=True)
    
    # L3
    stats['l3_avg'] = round(np.mean(player_season_df[prop_col].head(3).values), 2)
    
    # L5 EWMA
    ewma = player_season_df_asc[prop_col].ewm(alpha=(1.0 - cfg.EWMA_DECAY_FACTOR), adjust=False).mean()
    stats['l5_avg'] = round(ewma.iloc[-1], 2) if not ewma.empty else stats['szn_avg']
    
    # L10 Std Dev
    l10_vals = player_season_df[prop_col].head(10).values
    stats['l10_std_dev'] = round(np.std(l10_vals), 2) if len(l10_vals) >= 5 else stats['szn_std_dev']

    # Rolling Advanced
    decay_alpha = 1.0 - cfg.EWMA_DECAY_FACTOR
    for metric in ['TS_PCT', 'EFG_PCT', 'USG_PROXY']:
        key = metric.lower().replace('_pct', '').replace('_proxy', '') # ts, efg, usg
        if metric in player_season_df_asc:
            ewma_metric = player_season_df_asc[metric].ewm(alpha=decay_alpha, adjust=False).mean()
            stats[f'l5_avg_{key}'] = round(ewma_metric.iloc[-1], 4) if not ewma_metric.empty else stats[f'szn_avg_{key}']
        else:
            stats[f'l5_avg_{key}'] = stats[f'szn_avg_{key}']

    return stats

def determine_rest_factor(player_id, box_scores_df, prop_game_date):
    prop_date = pd.to_datetime(prop_game_date).normalize()
    player_games = box_scores_df[box_scores_df['PLAYER_ID'] == player_id]
    
    if player_games.empty: return 7
    
    past_games = player_games[player_games['GAME_DATE'] < prop_date]
    if past_games.empty: return 7
        
    last_game_date = past_games['GAME_DATE'].max()
    if pd.notna(last_game_date):
        return min((prop_date - last_game_date).days, 7)
    return 7

def get_schedule_fatigue_metrics(player_id, box_scores_df, prop_game_date):
    prop_date = pd.to_datetime(prop_game_date).normalize()
    start_window = prop_date - timedelta(days=5)
    
    player_games = box_scores_df[box_scores_df['PLAYER_ID'] == player_id]
    recent = player_games[(player_games['GAME_DATE'] >= start_window) & (player_games['GAME_DATE'] < prop_date)]
    
    is_b2b = 1 if any(recent['GAME_DATE'] == (prop_date - timedelta(days=1))) else 0
    return {'games_in_l5': len(recent), 'is_b2b': is_b2b}

def calculate_live_vacancy(team_abbr, full_roster_df, inj_df):
    """
    Calculates the total missing Usage and Minutes for a team based on the injury report.
    Returns: (missing_usg, missing_min, missing_usg_g, missing_usg_f)
    """
    if inj_df is None or inj_df.empty or full_roster_df is None or full_roster_df.empty:
        return 0.0, 0.0, 0.0, 0.0

    # 1. Filter Injuries for this Team
    team_injuries = inj_df[inj_df['Team'] == team_abbr] if 'Team' in inj_df.columns else pd.DataFrame()
    if team_injuries.empty or 'Status_Clean' not in team_injuries.columns:
        return 0.0, 0.0, 0.0, 0.0

    out_players = team_injuries[team_injuries['Status_Clean'].isin(['OUT', 'DOUBTFUL'])]
    if out_players.empty:
        return 0.0, 0.0, 0.0, 0.0

    # 2. Filter Roster for this Team
    team_roster = full_roster_df[full_roster_df['TEAM_ABBREVIATION'] == team_abbr].copy()
    if team_roster.empty: 
        return 0.0, 0.0, 0.0, 0.0
    
    # 3. Dynamic Column Selection
    # Determine Usage Column
    if 'USG_PROXY' in team_roster.columns: usg_col = 'USG_PROXY'
    elif 'USG%' in team_roster.columns: usg_col = 'USG%'
    elif 'SEASON_USG' in team_roster.columns: usg_col = 'SEASON_USG'
    else:
        usg_col = 'USG_TEMP'
        team_roster[usg_col] = 0.20

    # Determine Minutes Column
    if 'HOME_MIN' in team_roster.columns: min_col = 'HOME_MIN'
    elif 'Home_MIN' in team_roster.columns: min_col = 'Home_MIN'
    elif 'SEASON_MIN' in team_roster.columns: min_col = 'SEASON_MIN'
    elif 'MIN' in team_roster.columns: min_col = 'MIN'
    else:
        min_col = 'MIN_TEMP'
        team_roster[min_col] = 0.0
        
    # Determine Position Column (For Split Vacancy)
    pos_col = 'Pos' if 'Pos' in team_roster.columns else None

    # 4. Create Lookup Map
    team_roster['match_name'] = team_roster['clean_name'].fillna('')
    
    cols_to_pull = [usg_col, min_col]
    if pos_col: cols_to_pull.append(pos_col)
    
    roster_map = team_roster.set_index('match_name')[cols_to_pull].to_dict('index')
    roster_names = list(roster_map.keys())
    
    missing_usg = 0.0
    missing_min = 0.0
    missing_usg_g = 0.0
    missing_usg_f = 0.0
    
    for _, row in out_players.iterrows():
        p_name = str(row.get('Player', ''))
        clean_inj_name = preprocess_name_for_fuzzy_match(p_name)
        match = process.extractOne(clean_inj_name, roster_names, scorer=fuzz.token_sort_ratio, score_cutoff=85)
        
        if match:
            matched_name = match[0]
            stats = roster_map[matched_name]
            
            avg_min = float(stats.get(min_col, 0))
            # Only count vacancy if the player plays significant minutes
            if avg_min > 12.0:
                u_val = float(stats.get(usg_col, 0))
                
                missing_usg += u_val
                missing_min += avg_min
                
                # Split Logic
                if pos_col:
                    raw_pos = str(stats.get(pos_col, '')).upper()
                    # Heuristic: If position contains 'G', it's a Guard. Else Forward/Center
                    if 'G' in raw_pos:
                        missing_usg_g += u_val
                    else:
                        missing_usg_f += u_val

    return round(missing_usg, 2), round(missing_min, 2), round(missing_usg_g, 2), round(missing_usg_f, 2)