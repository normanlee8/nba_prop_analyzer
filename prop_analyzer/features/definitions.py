# Centralized definition of all features used in the model
# Prevents mismatch between Training columns and Inference columns

QUARTER_FEATURES = [
    'Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_MIN',
    'Q2_PTS', 'Q2_REB', 'Q2_AST', 'Q2_MIN',
    'Q3_PTS', 'Q3_REB', 'Q3_AST', 'Q3_MIN',
    'Q4_PTS', 'Q4_REB', 'Q4_AST', 'Q4_MIN'
]

VS_OPP_FEATURES = [
    'VS_OPP_PTS', 'VS_OPP_REB', 'VS_OPP_AST', 'VS_OPP_STL', 'VS_OPP_BLK', 
    'VS_OPP_FG3M', 'VS_OPP_TOV', 'VS_OPP_PRA', 'VS_OPP_PR', 'VS_OPP_PA', 
    'VS_OPP_RA', 'VS_OPP_FANTASY_PTS', 'VS_OPP_MIN', 'VS_OPP_GAMES_PLAYED'
]

HIST_FEATURES = [
    'HIST_VS_OPP_GAMES', 'HIST_VS_OPP_PTS_AVG', 'HIST_VS_OPP_REB_AVG',
    'HIST_VS_OPP_AST_AVG', 'HIST_VS_OPP_FG3M_AVG', 'HIST_VS_OPP_STL_AVG',
    'HIST_VS_OPP_BLK_AVG', 'HIST_VS_OPP_TOV_AVG', 'HIST_VS_OPP_PRA_AVG',
    'HIST_VS_OPP_PR_AVG', 'HIST_VS_OPP_PA_AVG', 'HIST_VS_OPP_RA_AVG',
    'HIST_VS_OPP_STK_AVG', 'HIST_VS_OPP_FANTASY_PTS_AVG',
    'HIST_VS_OPP_TS_PCT_AVG', 'HIST_VS_OPP_EFG_PCT_AVG',
    'HIST_VS_OPP_USG_PROXY_AVG'
]

BASE_FEATURE_COLS = [
    'Prop Line', 'SZN Avg', 'L3 Avg', 'L5 EWMA', 'Loc Avg', 'CoV %',
    'SZN Games', 'Selected Std Dev', 'Days Rest',
    'L10_STD_DEV', 'GAMES_IN_L5', 'IS_B2B',
    'TEAM_MISSING_USG', 'TEAM_MISSING_MIN',
    'SZN_TS_PCT', 'SZN_EFG_PCT', 'SZN_USG_PROXY',
    'L5_TS_PCT', 'L5_EFG_PCT', 'L5_USG_PROXY',
    'LOC_TS_PCT', 'LOC_EFG_PCT', 'LOC_USG_PROXY',
    'TS_DIFF', 'EFG_DIFF', 'USG_DIFF',
    'PBP_OnCourt_PlusMinus', 'PBP_OnCourt_USG_PCT',
    'PBP_OnCourt_TOV_PCT', 'PBP_OnCourt_3PAr',
    'SEASON_TS_PLUS', 'SEASON_eFG_PLUS', 'SEASON_FGA_PCT_3P',
] + HIST_FEATURES + VS_OPP_FEATURES + QUARTER_FEATURES

# Defines which features are relevant for which prop type
PROP_FEATURE_MAP = {
    'PTS': ['PTS', 'FANTASY_PTS', 'PRA', 'PR', 'PA'],
    'REB': ['REB', 'FANTASY_PTS', 'PRA', 'PR', 'RA'],
    'AST': ['AST', 'FANTASY_PTS', 'PRA', 'PA', 'RA'],
    'FG3M': ['FG3M'],
    'STL': ['STL', 'FANTASY_PTS', 'STK'],
    'BLK': ['BLK', 'FANTASY_PTS', 'STK'],
    'TOV': ['TOV', 'FANTASY_PTS'],
    'PRA': ['PTS', 'REB', 'AST', 'FANTASY_PTS', 'PRA', 'PR', 'PA', 'RA'],
    'PR': ['PTS', 'REB', 'FANTASY_PTS', 'PRA', 'PR', 'PA'],
    'PA': ['PTS', 'AST', 'FANTASY_PTS', 'PRA', 'PA', 'RA'],
    'RA': ['REB', 'AST', 'FANTASY_PTS', 'PRA', 'RA', 'PR'],
    'STK': ['STL', 'BLK', 'FANTASY_PTS', 'STK'],
    'FANTASY_PTS': ['PTS', 'REB', 'AST', 'STL', 'BLK', 'TOV', 'FANTASY_PTS', 'PRA', 'PR', 'PA', 'RA', 'STK'],
    'Q1_PTS': ['PTS', 'FANTASY_PTS'], 
    'Q1_REB': ['REB', 'FANTASY_PTS'],
    'Q1_AST': ['AST', 'FANTASY_PTS'],
    'Q1_PRA': ['PTS', 'REB', 'AST', 'PRA'],
    '1H_PTS': ['PTS', 'FANTASY_PTS'],
    '1H_REB': ['REB', 'FANTASY_PTS'],
    '1H_AST': ['AST', 'FANTASY_PTS'],
    '1H_PRA': ['PTS', 'REB', 'AST', 'PRA'],
}