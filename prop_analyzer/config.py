from pathlib import Path

# --- PATHS ---
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
(INPUT_DIR / "records").mkdir(parents=True, exist_ok=True)

# Specific File Paths
# Input text file for copy-pasting props
INPUT_PROPS_TXT = INPUT_DIR / "props_input.txt"
# Parsed CSV for analysis
PROPS_FILE = INPUT_DIR / "props_today.csv"
# Final results
PROCESSED_OUTPUT = OUTPUT_DIR / "processed_props.csv"

# Master Data Files (Defaults to Current Season)
# We also define patterns to load multiple seasons
MASTER_PLAYER_FILE = DATA_DIR / "master_player_stats_2025-26.csv"
MASTER_PLAYER_PATTERN = "master_player_stats_*.csv"

MASTER_TEAM_FILE = DATA_DIR / "master_team_stats_2025-26.csv"
MASTER_TEAM_PATTERN = "master_team_stats_*.csv"

MASTER_BOX_SCORES_FILE = DATA_DIR / "master_box_scores_2025-26.csv"
MASTER_BOX_SCORES_PATTERN = "master_box_scores_*.csv"

MASTER_VS_OPP_FILE = DATA_DIR / "master_vs_opponent.csv"
MASTER_DVP_FILE = DATA_DIR / "master_dvp_stats.csv"

# --- THRESHOLDS ---
# Pre-game Logic
MIN_PROB_FOR_S_TIER = 0.585
MIN_EDGE_FOR_S_TIER = 1.5
MIN_EDGE_FOR_A_TIER = 1.0

# Live Logic
LIVE_MIN_PROB_THRESHOLD = 0.65
LIVE_BLOWOUT_THRESHOLD = 20

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
    '1st Quarter Points': 'Q1_PTS', '1Q Points': 'Q1_PTS',
    '1st Quarter Rebounds': 'Q1_REB', '1Q Rebounds': 'Q1_REB',
    '1st Quarter Assists': 'Q1_AST', '1Q Assists': 'Q1_AST',
    '1st Quarter PRA': 'Q1_PRA', '1Q Pts + Rebs + Asts': 'Q1_PRA',
    '1st Half Points': '1H_PTS', '1H Points': '1H_PTS',
    '1st Half Rebounds': '1H_REB', '1H Rebounds': '1H_REB',
    '1st Half Assists': '1H_AST', '1H Assists': '1H_AST',
    '1st Half PRA': '1H_PRA', '1H Pts + Rebs + Asts': '1H_PRA',
}

SUPPORTED_PROPS = [
    'PTS', 'REB', 'AST', 'FG3M', 'STL', 'BLK', 'TOV',
    'PRA', 'PR', 'PA', 'RA', 'STK', 'FANTASY_PTS',
    'Q1_PTS', 'Q1_REB', 'Q1_AST', 'Q1_PRA',
    '1H_PTS', '1H_REB', '1H_AST', '1H_PRA'
]