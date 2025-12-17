import pandas as pd
import os
import sys
import logging
from datetime import datetime
import colorama
from colorama import Fore, Style

# Initialize colorama
colorama.init(autoreset=True)

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prop_analyzer import config as cfg
from prop_analyzer.features.generator import generate_features
from prop_analyzer.models.inference import predict_props

# --- CONFIGURATION ---
INPUT_PROPS_PATH = os.path.join(cfg.BASE_DIR, 'input', 'props_today.csv')
OUTPUT_FILE = os.path.join(cfg.BASE_DIR, 'output', 'processed_props.xlsx')
LOG_DIR = os.path.join(cfg.BASE_DIR, 'logs')

# Parlay Safety Thresholds
MAX_PARLAY_SPREAD_PCT = 0.40  # Max allowed volatility (Spread / Line)
MIN_PARLAY_PROB = 0.56       # Minimum implied probability for parlay inclusion

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        filename=os.path.join(LOG_DIR, 'analysis_pregame.log'),
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)

def load_todays_props():
    if not os.path.exists(INPUT_PROPS_PATH):
        logging.error(f"props_today.csv not found at {INPUT_PROPS_PATH}")
        sys.exit(1)
    
    df = pd.read_csv(INPUT_PROPS_PATH)
    # Ensure clean formatting
    df['Prop Line'] = pd.to_numeric(df['Prop Line'], errors='coerce')
    df = df.dropna(subset=['Prop Line', 'Player Name'])
    return df

def build_parlay(df, n_picks=6):
    """
    Selects the best n_picks ensuring:
    1. Highest Win Probability
    2. Low Volatility (Spread_Pct)
    3. Game Independence (1 pick per game)
    """
    # Filter candidates
    candidates = df[
        (df['Tier'].isin(['S Tier', 'A Tier', 'B Tier'])) &
        (df['Spread_Pct'] <= MAX_PARLAY_SPREAD_PCT) &
        (df['Win_Prob'] >= MIN_PARLAY_PROB)
    ].copy()
    
    # Sort by Confidence (Win_Prob) -> Edge (Diff%)
    candidates = candidates.sort_values(by=['Win_Prob', 'Diff%'], ascending=[False, False])
    
    selected_picks = []
    used_teams = set()
    
    for _, row in candidates.iterrows():
        # Identify the game (Team vs Opponent)
        team = row.get('Team', 'UNK')
        opp = row.get('Opponent', 'UNK')
        
        # Check if this game is already covered
        if team in used_teams or opp in used_teams:
            continue
            
        selected_picks.append(row)
        used_teams.add(team)
        used_teams.add(opp)
        
        if len(selected_picks) >= n_picks:
            break
            
    return pd.DataFrame(selected_picks)

def main():
    setup_logging()
    logging.info(">>> STARTING DAILY PROP ANALYSIS <<<")
    
    # 1. Load Input
    raw_props = load_todays_props()
    logging.info(f"Loaded {len(raw_props)} props from input.")
    
    # 2. Generate Features
    logging.info("Generating features (aggregating stats)...")
    try:
        features_df = generate_features(raw_props)
    except Exception as e:
        logging.error(f"Feature generation failed: {e}")
        sys.exit(1)
        
    if features_df.empty:
        logging.warning("No features generated. Exiting.")
        sys.exit(0)

    # 3. Run Inference (Models)
    logging.info("Running ML Models...")
    predictions = predict_props(features_df)
    
    if predictions.empty:
        logging.warning("No predictions made.")
        sys.exit(0)

    # 4. Post-Processing & Sorting
    # Sort primarily by Tier Rank, then Win Probability
    tier_map = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3, 'Void': 4}
    predictions['Tier_Rank'] = predictions['Tier'].map(tier_map)
    
    final_df = predictions.sort_values(
        by=['Tier_Rank', 'Win_Prob', 'Diff%'], 
        ascending=[True, False, False]
    ).drop(columns=['Tier_Rank'])

    # 5. Export to Excel
    try:
        final_df.to_excel(OUTPUT_FILE, index=False)
        logging.info(f"Analysis saved to {OUTPUT_FILE}")
    except Exception as e:
        logging.error(f"Failed to save Excel output: {e}")

    # 6. Terminal Output
    print("\n" + "="*60)
    print(f"{Fore.CYAN}>>> TOP RECOMMENDED PLAYS (S & A Tier) <<<{Style.RESET_ALL}")
    print("="*60)
    
    top_plays = final_df[final_df['Tier'].isin(['S Tier', 'A Tier'])].head(15)
    
    if top_plays.empty:
        print(f"{Fore.YELLOW}No S or A Tier plays found today.{Style.RESET_ALL}")
    else:
        print(f"{'Player':<20} | {'Prop':<15} | {'Line':<5} | {'Pick':<5} | {'Prob':<5} | {'Spread%':<7} | {'Tier'}")
        print("-" * 85)
        for _, row in top_plays.iterrows():
            # Color code the tier
            tier_color = Fore.GREEN if row['Tier'] == 'S Tier' else Fore.YELLOW
            # Color code probability
            prob_str = f"{row['Win_Prob']:.1%}"
            
            print(f"{row['Player Name']:<20} | {row['Prop Category']:<15} | {row['Prop Line']:<5} | "
                  f"{row['Edge_Type']:<5} | {prob_str:<5} | {row['Spread_Pct']:<7.1%} | "
                  f"{tier_color}{row['Tier']}{Style.RESET_ALL}")

    # 7. Parlay Builder Output
    print("\n" + "="*60)
    print(f"{Fore.MAGENTA}>>> SUGGESTED 6-LEG PARLAY (Low Volatility / Independent Games) <<<{Style.RESET_ALL}")
    print("="*60)
    
    parlay_picks = build_parlay(final_df, n_picks=6)
    
    if parlay_picks.empty:
        print("Not enough high-confidence independent picks for a parlay today.")
    else:
        print(f"{'Player':<20} | {'Team':<4} | {'Opp':<4} | {'Prop':<12} | {'Pick':<5} | {'Conf':<6}")
        print("-" * 70)
        for _, row in parlay_picks.iterrows():
            print(f"{row['Player Name']:<20} | {row.get('Team', 'UNK'):<4} | {row.get('Opponent', 'UNK'):<4} | "
                  f"{row['Prop Category']:<12} | {row['Edge_Type']:<5} | {row['Win_Prob']:.1%}")
        
        # Calculate combined probability (naive)
        combined_prob = parlay_picks['Win_Prob'].prod()
        # Implied odds (1/prob)
        implied_odds = (1 / combined_prob) if combined_prob > 0 else 0
        print("-" * 70)
        print(f"Est. Parlay Win Probability: {Fore.GREEN}{combined_prob:.2%}{Style.RESET_ALL}")
        print(f"Fair Implied Odds: +{int(implied_odds * 100) if implied_odds > 0 else 0}")

    print("\n")
    logging.info("Analysis run complete.")

if __name__ == "__main__":
    main()