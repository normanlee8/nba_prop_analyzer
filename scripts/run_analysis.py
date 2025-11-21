import sys
import pandas as pd
import logging
from pathlib import Path
from textwrap import dedent

sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import loader
from prop_analyzer.features import generator
from prop_analyzer.models import registry, inference
from prop_analyzer.utils import common, text

def print_report(results):
    if not results:
        print("No results to display.")
        return

    df = pd.DataFrame(results)
    
    # Sorting for display
    tier_order = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
    df['sort_idx'] = df['Tier'].map(tier_order).fillna(99)
    df.sort_values(by=['sort_idx', 'Win_Prob'], ascending=[True, False], inplace=True)

    print("\n" + "="*115)
    print(f"NBA PROP REPORT | {len(df)} Props Analyzed")
    print("="*115)
    print(f"{'Tier':<6} | {'Player':<20} | {'Prop':<12} | {'Line':>5} | {'Pick':<5} | {'Prob%':>6} | {'Edge':>5} | {'Proj':>5} | {'Vac%':>5}")
    print("-" * 115)

    for _, row in df.head(30).iterrows():
        prob_str = f"{row['Win_Prob']:.1%}" if pd.notna(row['Win_Prob']) else "N/A"
        
        print(
            f"{row['Tier']:<6} | "
            f"{str(row['Player Name'])[:20]:<20} | "
            f"{row['Prop Category']:<12} | "
            f"{row['Prop Line']:>5.1f} | "
            f"{row['Best Pick']:<5} | "
            f"{prob_str:>6} | "
            f"{row['Edge']:>5.2f} | "
            f"{row['Median_Proj']:>5.1f} | "
            f"{row.get('TEAM_MISSING_USG', 0):>5.1f}"
        )
    print("="*115)

def main():
    common.setup_logging(name="analyzer")
    logging.info(">>> Starting Daily Prop Analyzer")

    # 1. Load Static Data
    player_stats, team_stats, _ = loader.load_static_data()
    vs_opp_df = loader.load_vs_opponent_data()
    model_cache = registry.load_model_cache()
    
    if player_stats is None or model_cache is None:
        logging.critical("Missing required data or models.")
        return

    # 2. Load Props
    try:
        props_df = pd.read_csv(cfg.PROPS_FILE)
    except Exception as e:
        logging.critical(f"Could not load props file: {e}")
        return

    # 3. Load Box Scores (Optimization)
    relevant_ids = []
    for name in props_df['Player Name'].unique():
        p_data = text.fuzzy_match_player(name, player_stats)
        if p_data is not None: relevant_ids.append(p_data['PLAYER_ID'])
    
    box_scores = loader.load_box_scores(relevant_ids)

    results = []
    
    # 4. Main Loop
    for _, row in props_df.iterrows():
        player_name = row['Player Name']
        prop_cat_raw = row['Prop Category']
        prop_cat = cfg.MASTER_PROP_MAP.get(prop_cat_raw, prop_cat_raw)
        
        if prop_cat not in cfg.SUPPORTED_PROPS: continue

        p_data = text.fuzzy_match_player(player_name, player_stats)
        if p_data is None: continue

        features, _, _ = generator.build_feature_vector(
            p_data, p_data['PLAYER_ID'], prop_cat, row['Prop Line'],
            row['Team'], row['Opponent'], 
            ('vs' in row['Matchup']), row['GAME_DATE'],
            box_scores, team_stats, vs_opp_df, full_roster_df=player_stats
        )

        pred = inference.predict_prop(model_cache, prop_cat, features)
        if pred:
            metrics = inference.determine_tier(
                row['Prop Line'], pred['q20'], pred['q80'], pred['prob_over']
            )
            
            res = {
                'Player Name': p_data['PLAYER_NAME'],
                'Prop Category': prop_cat,
                'Prop Line': row['Prop Line'],
                'GAME_DATE': row['GAME_DATE'],
                'Matchup': row['Matchup'],
                'Team': row['Team'],
                'Opponent': row['Opponent'],
                **metrics,
                'TEAM_MISSING_USG': features.get('TEAM_MISSING_USG', 0.0)
            }
            results.append(res)

    # 5. Sort and Save
    if results:
        df = pd.DataFrame(results)
        
        # --- SORTING LOGIC ---
        tier_order = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
        df['sort_idx'] = df['Tier'].map(tier_order).fillna(99)
        # Sort by Tier (Asc) then Win Probability (Desc)
        df = df.sort_values(by=['sort_idx', 'Win_Prob'], ascending=[True, False])
        df = df.drop(columns=['sort_idx']) # Clean up helper column
        
        # Print Report (Converting back to list of dicts for the existing function, or just pass DF)
        print_report(df.to_dict('records'))
        
        # Save Sorted DF
        df.to_csv(cfg.PROCESSED_OUTPUT, index=False)
        logging.info(f"Saved sorted results to {cfg.PROCESSED_OUTPUT}")
    else:
        print("No results generated.")

if __name__ == "__main__":
    main()