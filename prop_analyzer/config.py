from pathlib import Path

# --- PATHS ---
# Base directory of execution (usually where you run the script from)
BASE_DIR = Path(".")

# Data Directories
DATA_DIR = BASE_DIR / "prop_data"
MODEL_DIR = BASE_DIR / "prop_models"
INPUT_DIR = BASE_DIR / "input"
OUTPUT_DIR = BASE_DIR / "output"

# Ensure key directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
INPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Specific File Paths
PROPS_FILE = INPUT_DIR / "props_today.csv"
PROCESSED_OUTPUT = OUTPUT_DIR / "processed_props.csv"
UNPROCESSED_OUTPUT = OUTPUT_DIR / "unprocessed_props.csv"
INPUT_PROPS_TXT = INPUT_DIR / "props_input.txt"
LIVE_INPUT_TXT = INPUT_DIR / "live_input.txt"

# Master Data Files
MASTER_PLAYER_FILE = DATA_DIR / "master_player_stats.csv"
MASTER_TEAM_FILE = DATA_DIR / "master_team_stats.csv"
MASTER_BOX_SCORES_FILE = DATA_DIR / "master_box_scores.csv"
MASTER_VS_OPP_FILE = DATA_DIR / "master_vs_opponent.csv"

# --- NEW: ODDS API CONFIG ---
# Get a free key at https://the-odds-api.com/
ODDS_API_KEY = "082486d25139c715fef49236b58234a6"  # <--- PASTE YOUR KEY INSIDE THESE QUOTES
ODDS_SPORTSBOOKS = "draftkings,fanduel,mgm,caesars,betrivers" # Books to check
ODDS_MARKETS = [
    "player_points", "player_rebounds", "player_assists", 
    "player_threes", "player_points_rebounds_assists", 
    "player_points_rebounds", "player_points_assists", 
    "player_rebounds_assists", "player_steals", "player_blocks", 
    "player_turnovers"
]

# --- THRESHOLDS ---
# Pre-game Logic
MIN_PROB_FOR_S_TIER = 0.58
MIN_EDGE_FOR_S_TIER = 1.5
MIN_EDGE_FOR_A_TIER = 0.08

# Live Logic
LIVE_MIN_PROB_THRESHOLD = 0.65
LIVE_BLOWOUT_THRESHOLD = 18
LIVE_PACE_WEIGHT = 0.5
LIVE_HOT_HAND_WEIGHT = 0.25

# Feature Engineering
BAYESIAN_PRIOR_WEIGHT = 12
EWMA_DECAY_FACTOR = 0.85
MIN_GAMES_FOR_ANALYSIS = 5

# --- PRIORS ---
BAYESIAN_PRIORS = {
    'PTS': 12.0, 'REB': 4.0, 'AST': 3.0, 'FG3M': 1.5,
    'STL': 0.8, 'BLK': 0.5, 'TOV': 1.5, 'PRA': 18.0,
    'PR': 16.0, 'PA': 15.0, 'RA': 7.0, 'STK': 1.3,
    'FANTASY_PTS': 25.0,
    'Q1_PTS': 4.0, 'Q1_REB': 1.5, 'Q1_AST': 1.0, 'Q1_PRA': 6.5,
    '1H_PTS': 7.0, '1H_REB': 2.5, '1H_AST': 2.0, '1H_PRA': 11.5,
}

# --- PROP MAPPING ---
MASTER_PROP_MAP = {
    'Points': 'PTS', 'pts': 'PTS',
    'Rebounds': 'REB', 'reb': 'REB',
    'Assists': 'AST', 'ast': 'AST',
    'Blocks': 'BLK', 'blk': 'BLK',
    'Steals': 'STL', 'stl': 'STL',
    'Turnovers': 'TOV', 'tov': 'TOV',
    '3-Pointers Made': 'FG3M', '3-Point HITS': 'FG3M', '3 Pointers Made': 'FG3M', 'fg3m': 'FG3M',
    'Pts + Rebs + Asts': 'PRA', 'Pts+Rebs+Asts': 'PRA', 'pra': 'PRA',
    'Rebounds + Assists': 'RA', 'ra': 'RA',
    'Points + Rebounds': 'PR', 'pr': 'PR',
    'Points + Assists': 'PA', 'pa': 'PA',
    'Steals + Blocks': 'STK', 'Stls + Blks': 'STK', 'stk': 'STK',
    'Fantasy Points': 'FANTASY_PTS', 'Fantasy Score': 'FANTASY_PTS', 'fantasy points': 'FANTASY_PTS',
    
    # Quarter/Half Props
    '1st Quarter Points': 'Q1_PTS', '1Q Points': 'Q1_PTS', '1st Quarter Pts': 'Q1_PTS',
    '1st Quarter Rebounds': 'Q1_REB', '1Q Rebounds': 'Q1_REB',
    '1st Quarter Assists': 'Q1_AST', '1Q Assists': 'Q1_AST',
    '1st Quarter PRA': 'Q1_PRA', '1Q Pts + Rebs + Asts': 'Q1_PRA',
    '1st Half Points': '1H_PTS', '1H Points': '1H_PTS',
    '1st Half Rebounds': '1H_REB', '1H Rebounds': '1H_REB',
    '1st Half Assists': '1H_AST', '1H Assists': '1H_AST',
    '1st Half PRA': '1H_PRA', '1H Pts + Rebs + Asts': '1H_PRA',

    # Auxiliary
    '3s Attempted': 'FG3A', 'fg3a': 'FG3A',
    'Field Goals Made': 'FGM', 'fgm': 'FGM',
    'Field Goals Attempted': 'FGA', 'fga': 'FGA',
    'Free Throws Made': 'FTM', 'ftm': 'FTM',
    'Free Throws Attempted': 'FTA', 'fta': 'FTA',
    'Offensive Rebounds': 'OREB', 'oreb': 'OREB',
    'Defensive Rebounds': 'DREB', 'dreb': 'DREB',
}

SUPPORTED_PROPS = [
    'PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'STK', 'FANTASY_PTS',
    'Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_PRA',
    '1H_PTS', '1H_REB', '1H_AST', '1H_PRA'
]

UNSUPPORTED_PROP_FILTERS = [
    'Double Doubles', 'Triple Doubles', 'DD', 'TD',
    'SPECIAL_DOUBLE_DOUBLE', 'SPECIAL_TRIPLE_DOUBLE',
    '1Q_FG3M', 'PTS_EACH_QTR', 'FIRST_FG_ATT', 'FIRST_FG3_ATT',
    'GAME_HIGH_SCORE', 'TEAM_HIGH_SCORE',
    'First 5 Min', 'Margin'
]