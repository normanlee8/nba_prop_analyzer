import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from rapidfuzz import process

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import loader, odds_client
from prop_analyzer.features import generator
from prop_analyzer.models import registry, inference
from prop_analyzer.utils import common, text

# --- HELPER FUNCTIONS ---

def get_decimal_odds(american_odds):
    """Converts American Odds (-110, +150) to Decimal Odds (1.91, 2.50)."""
    try:
        odds = int(american_odds)
        if odds > 0:
            return 1 + (odds / 100)
        else:
            return 1 + (100 / abs(odds))
    except (ValueError, TypeError):
        return 1.91 # Default to -110 if invalid

def calculate_ev(win_prob, american_odds):
    """
    Calculates Expected Value (ROI).
    Formula: (Prob_Win * Profit) - (Prob_Loss * Stake)
    Simplified: (Prob_Win * Decimal_Odds) - 1
    """
    decimal = get_decimal_odds(american_odds)
    ev = (win_prob * decimal) - 1
    return ev

def match_odds_to_prop(row, odds_df):
    """
    Finds the matching odds for a specific prop.
    Matches on: Player Name (Fuzzy), Category, and Exact Line.
    """
    if odds_df.empty:
        return -110

    # 1. Filter by Category
    subset = odds_df[odds_df['Prop Category'] == row['Prop Category']].copy()
    if subset.empty: return -110

    # 2. Match Player Name (Exact first, then Fuzzy)
    target_name = row['Player Name'].lower()
    player_matches = subset[subset['Player Name'].str.lower() == target_name]
    
    if player_matches.empty:
        names = subset['Player Name'].unique()
        match = process.extractOne(row['Player Name'], names, score_cutoff=85)
        if match:
            player_matches = subset[subset['Player Name'] == match[0]]
    
    if player_matches.empty: return -110

    # 3. Match Line (Exact)
    # We only use the odds if the Sportsbook line matches our input line exactly.
    # If input is 24.5 and book is 25.5, the odds are irrelevant.
    line_match = player_matches[player_matches['Prop Line'] == row['Prop Line']]
    
    if line_match.empty:
        return -110
    
    # 4. Get 'Over' Odds
    # Currently, we prioritize finding the price for the Over. 
    # (A more advanced version could return both Over/Under prices)
    over_bet = line_match[line_match['Bet Type'] == 'Over']
    if not over_bet.empty:
        # Return the best (highest) odds found across available books
        return int(over_bet['Odds'].max())
        
    return -110

def print_report(results):
    if not results:
        print("No results to display.")
        return

    df = pd.DataFrame(results)
    
    # Sorting Logic for Display
    tier_order = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
    df['sort_idx'] = df['Tier'].map(tier_order).fillna(99)
    
    # Sort by: 1. Tier (Safety) -> 2. EV (Value)
    df.sort_values(by=['sort_idx', 'EV'], ascending=[True, False], inplace=True)

    print("\n" + "="*135)
    print(f"NBA PROP REPORT | {len(df)} Props Analyzed")
    print("="*135)
    print(f"{'Tier':<6} | {'Player':<20} | {'Prop':<12} | {'Line':>5} | {'Pick':<5} | {'Odds':>5} | {'Prob%':>6} | {'EV%':>6} | {'Proj':>5} | {'Vac%':>5}")
    print("-" * 135)

    for _, row in df.head(50).iterrows():
        prob_str = f"{row['Win_Prob']:.1%}" if pd.notna(row['Win_Prob']) else "N/A"
        ev_str = f"{row['EV']:.1%}"
        
        # Formatting for readability
        ev_display = f"+{ev_str}" if row['EV'] > 0 else ev_str
        odds_display = f"{row['Odds']}" if row['Odds'] > 0 else f"{row['Odds']}"

        print(
            f"{row['Tier']:<6} | "
            f"{str(row['Player Name'])[:20]:<20} | "
            f"{row['Prop Category']:<12} | "
            f"{row['Prop Line']:>5.1f} | "
            f"{row['Best Pick']:<5} | "
            f"{odds_display:>5} | "
            f"{prob_str:>6} | "
            f"{ev_display:>6} | "
            f"{row['Median_Proj']:>5.1f} | "
            f"{row.get('TEAM_MISSING_USG', 0):>5.1f}"
        )
    print("="*135)

# --- MAIN EXECUTION ---

def main():
    common.setup_logging(name="analyzer")
    logging.info(">>> Starting Daily Prop Analyzer (v2.0 - Integrated)")

    # 1. Load Static Data
    player_stats, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    model_cache = registry.load_model_cache()
    
    # Load Phase 2 Data (DvP)
    try:
        dvp_df = pd.read_csv(cfg.DATA_DIR / "master_dvp_stats.csv")
    except:
        dvp_df = None
        logging.warning("DvP stats not found. Skipping DvP features.")
    
    if player_stats is None or model_cache is None:
        logging.critical("Missing required data or models.")
        return

    # 2. Load Input Props
    try:
        props_df = pd.read_csv(cfg.PROPS_FILE)
    except Exception as e:
        logging.critical(f"Could not load props file: {e}")
        return

    # 3. Fetch & Match Live Odds
    live_odds_df = odds_client.get_daily_odds()
    
    if not live_odds_df.empty:
        logging.info(f"Fetched {len(live_odds_df)} live odds lines.")
        # Apply matching logic
        props_df['Odds'] = props_df.apply(lambda x: match_odds_to_prop(x, live_odds_df), axis=1)
    else:
        logging.warning("Could not fetch live odds (or API key missing). Defaulting to -110.")
        if 'Odds' not in props_df.columns:
            props_df['Odds'] = -110

    # 4. Load Box Scores (Optimized by Player ID)
    relevant_ids = []
    for name in props_df['Player Name'].unique():
        p_data = text.fuzzy_match_player(name, player_stats)
        if p_data is not None: relevant_ids.append(p_data['PLAYER_ID'])
    
    box_scores = loader.load_box_scores(relevant_ids)

    results = []
    
    # 5. Main Analysis Loop
    for _, row in props_df.iterrows():
        player_name = row['Player Name']
        prop_cat_raw = row['Prop Category']
        prop_cat = cfg.MASTER_PROP_MAP.get(prop_cat_raw, prop_cat_raw)
        
        if prop_cat not in cfg.SUPPORTED_PROPS: continue

        p_data = text.fuzzy_match_player(player_name, player_stats)
        if p_data is None: continue

        # Build Features (Includes Phase 2 DvP & Vacancy)
        features, _, _ = generator.build_feature_vector(
            p_data, p_data['PLAYER_ID'], prop_cat, row['Prop Line'],
            row['Team'], row['Opponent'], 
            ('vs' in row['Matchup']), row['GAME_DATE'],
            box_scores, team_stats, vs_opp_df, full_roster_df=player_stats,
            dvp_df=dvp_df 
        )

        # Predict
        pred = inference.predict_prop(model_cache, prop_cat, features)
        if pred:
            metrics = inference.determine_tier(
                row['Prop Line'], pred['q20'], pred['q80'], pred['prob_over']
            )
            
            # Calculate EV
            odds = row.get('Odds', -110)
            win_prob = metrics['Win_Prob']
            
            # Note: If model picks Under, we rely on Win_Prob being the Under probability.
            # EV calc assumes 'odds' aligns with the 'Best Pick'.
            # Since we mostly fetch Over odds, this is an approximation for Unders 
            # unless specific Under odds are mapped.
            
            ev = calculate_ev(win_prob, odds)

            res = {
                'Player Name': p_data['PLAYER_NAME'],
                'Prop Category': prop_cat,
                'Prop Line': row['Prop Line'],
                'GAME_DATE': row['GAME_DATE'],
                'Matchup': row['Matchup'],
                'Team': row['Team'],
                'Opponent': row['Opponent'],
                'Odds': odds,
                'EV': ev, 
                **metrics,
                'TEAM_MISSING_USG': features.get('TEAM_MISSING_USG', 0.0)
            }
            results.append(res)

    # 6. Sort and Save
    if results:
        df = pd.DataFrame(results)
        
        # Sort by Tier (S->C) then EV (High->Low)
        tier_order = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
        df['sort_idx'] = df['Tier'].map(tier_order).fillna(99)
        df = df.sort_values(by=['sort_idx', 'EV'], ascending=[True, False])
        df = df.drop(columns=['sort_idx']) 
        
        print_report(df.to_dict('records'))
        
        df.to_csv(cfg.PROCESSED_OUTPUT, index=False)
        logging.info(f"Saved sorted results to {cfg.PROCESSED_OUTPUT}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()