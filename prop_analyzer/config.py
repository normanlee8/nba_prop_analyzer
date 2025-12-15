from pathlib import Path

# --- PATHS ---
BASE_DIR = Path(".")

# Data Directories
DATA_DIR = BASE_DIR / "prop_data"
MODEL_DIR = BASE_DIR / "prop_models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"
GRADED_DIR = OUTPUT_DIR / "graded_history"

# Ensure key directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(INPUT_DIR / "records").mkdir(parents=True, exist_ok=True)
GRADED_DIR.mkdir(parents=True, exist_ok=True)

# Specific File Paths
INPUT_PROPS_TXT = INPUT_DIR / "props_input.txt"
PROPS_FILE = INPUT_DIR / "props_today.csv"

# Final results
PROCESSED_OUTPUT_SYSTEM = OUTPUT_DIR / "processed_props.parquet" 
PROCESSED_OUTPUT_XLSX = OUTPUT_DIR / "processed_props.xlsx"

# --- MASTER DATA FILES (ALL PARQUET) ---
MASTER_PLAYER_FILE = DATA_DIR / "master_player_stats_2025-26.parquet"
MASTER_PLAYER_PATTERN = "master_player_stats_*.parquet"

MASTER_TEAM_FILE = DATA_DIR / "master_team_stats_2025-26.parquet"
MASTER_TEAM_PATTERN = "master_team_stats_*.parquet"

MASTER_BOX_SCORES_FILE = DATA_DIR / "master_box_scores_2025-26.parquet"
MASTER_BOX_SCORES_PATTERN = "master_box_scores_*.parquet"

# NEW: Master Q1 History for Grading
MASTER_Q1_FILE = DATA_DIR / "master_q1_stats.parquet"

MASTER_VS_OPP_FILE = DATA_DIR / "master_vs_opponent.parquet"
MASTER_DVP_FILE = DATA_DIR / "master_dvp_stats.parquet"
MASTER_TRAINING_FILE = DATA_DIR / "master_training_dataset.parquet"

# --- DATA CONTRACT (SCHEMA) ---
class Cols:
    PLAYER_NAME = 'Player Name'
    PLAYER_ID = 'PLAYER_ID'
    TEAM = 'Team'
    OPPONENT = 'Opponent'
    MATCHUP = 'Matchup'
    DATE = 'GAME_DATE'
    
    PROP_TYPE = 'Prop Category'
    PROP_LINE = 'Prop Line'
    
    PREDICTION = 'Model_Pred'
    CONFIDENCE = 'Model_Conf'
    EDGE_TYPE = 'Edge_Type'
    TIER = 'Tier'
    
    ACTUAL_VAL = 'Actual Value'
    RESULT = 'Result'
    CORRECTNESS = 'Correctness'
    
    SZN_AVG = 'SZN_AVG'
    L5_AVG = 'L5_AVG'
    
    @classmethod
    def get_required_input_cols(cls):
        return [cls.PLAYER_NAME, cls.TEAM, cls.OPPONENT, cls.MATCHUP, cls.PROP_TYPE, cls.PROP_LINE, cls.DATE]

# --- THRESHOLDS ---
MIN_PROB_FOR_S_TIER = 0.585
MIN_EDGE_FOR_S_TIER = 1.5
MIN_EDGE_FOR_A_TIER = 1.0
LIVE_MIN_PROB_THRESHOLD = 0.65
LIVE_BLOWOUT_THRESHOLD = 20
BAYESIAN_PRIOR_WEIGHT = 12
EWMA_DECAY_FACTOR = 0.85
MIN_GAMES_FOR_ANALYSIS = 5

# --- PRIORS ---
# Added priors for new props (FTM, OREB)
BAYESIAN_PRIORS = {
    'PTS': 12.0, 'REB': 4.0, 'AST': 3.0, 'FG3M': 1.5,
    'STL': 0.8, 'BLK': 0.5, 'TOV': 1.5, 'PRA': 18.0,
    'PR': 16.0, 'PA': 15.0, 'RA': 7.0, 'STK': 1.3,
    'FANTASY_PTS': 25.0,
    'OREB': 0.8, 'FTM': 2.5,  # NEW
    'Q1_PTS': 4.0, 'Q1_REB': 1.5, 'Q1_AST': 1.0, 'Q1_PRA': 6.5,
    '1H_PTS': 7.0, '1H_REB': 2.5, '1H_AST': 2.0, '1H_PRA': 11.5,
    'FGA': 15.0, 'FG3A': 6.0, 
    'DD': 0.10, 'TD': 0.03
}

# --- PROP MAPPING ---
MASTER_PROP_MAP = {
    # Core
    'Points': 'PTS', 'pts': 'PTS',
    'Rebounds': 'REB', 'reb': 'REB',
    'Assists': 'AST', 'ast': 'AST',
    'Blocks': 'BLK', 'blk': 'BLK',
    'Steals': 'STL', 'stl': 'STL',
    'Turnovers': 'TOV', 'tov': 'TOV',
    
    # Shooting / Scoring Specifics (Added FT Made)
    '3-Pointers Made': 'FG3M', '3-Point HITS': 'FG3M', '3 Pointers Made': 'FG3M', 'fg3m': 'FG3M',
    'FG Attempted': 'FGA', 'Field Goals Attempted': 'FGA',
    '3s Attempted': 'FG3A', '3-Pointers Attempted': 'FG3A',
    'FT Made': 'FTM', 'Free Throws Made': 'FTM',
    
    # Rebounding Specifics (Added OREB)
    'Offensive Rebounds': 'OREB',
    
    # Combos
    'Double Doubles': 'DD', 'Double Double': 'DD',
    'Triple Doubles': 'TD', 'Triple Double': 'TD',
    'Pts + Rebs + Asts': 'PRA', 'Pts+Rebs+Asts': 'PRA', 'pra': 'PRA',
    'Rebounds + Assists': 'RA', 'ra': 'RA',
    'Points + Rebounds': 'PR', 'pr': 'PR',
    'Points + Assists': 'PA', 'pa': 'PA',
    'Steals + Blocks': 'STK', 'Stls + Blks': 'STK', 'stk': 'STK', 'Blocks + Steals': 'STK',
    'Fantasy Points': 'FANTASY_PTS', 'Fantasy Score': 'FANTASY_PTS', 'fantasy points': 'FANTASY_PTS',
    
    # Quarter Props (Added Q1 3-Pointers)
    '1st Quarter Points': 'Q1_PTS', '1Q Points': 'Q1_PTS',
    '1st Quarter Rebounds': 'Q1_REB', '1Q Rebounds': 'Q1_REB',
    '1st Quarter Assists': 'Q1_AST', '1Q Assists': 'Q1_AST',
    '1st Quarter PRA': 'Q1_PRA', '1Q Pts + Rebs + Asts': 'Q1_PRA',
    '1Q 3-Pointers Made': 'Q1_FG3M', '1st Quarter 3-Pointers Made': 'Q1_FG3M',
    
    # Half Props (Added 1H 3-Pointers)
    '1st Half Points': '1H_PTS', '1H Points': '1H_PTS',
    '1st Half Rebounds': '1H_REB', '1H Rebounds': '1H_REB',
    '1st Half Assists': '1H_AST', '1H Assists': '1H_AST',
    '1st Half PRA': '1H_PRA', '1H Pts + Rebs + Asts': '1H_PRA',
    '1H 3-Pointers Made': '1H_FG3M',
}

# Added new supported props so the model knows to look for them
SUPPORTED_PROPS = [
    'PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV',
    'FGA', 'FG3A', 'FTM', 'OREB', 'DD', 'TD',
    'PRA', 'PR', 'PA', 'RA', 'STK', 'FANTASY_PTS',
    'Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_PRA', 'Q1_FG3M',
    '1H_PTS', '1H_REB', '1H_AST', '1H_PRA', '1H_FG3M'
]