import pandas as pd
import logging
from prop_analyzer import config as cfg
from prop_analyzer.utils import text

def calculate_derived_stats(df):
    # Derived Quarter Stats
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        pts, reb, ast = f'{q}_PTS', f'{q}_REB', f'{q}_AST'
        if pts in df.columns and reb in df.columns and ast in df.columns:
            df[f'{q}_PRA'] = df[pts] + df[reb] + df[ast]
            df[f'{q}_PR'] = df[pts] + df[reb]
            df[f'{q}_PA'] = df[pts] + df[ast]
            df[f'{q}_RA'] = df[reb] + df[ast]

    # Derived Halves
    for stat in ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA']:
        if f'Q1_{stat}' in df.columns and f'Q2_{stat}' in df.columns:
            df[f'1H_{stat}'] = df[f'Q1_{stat}'] + df[f'Q2_{stat}']
    return df

def check_prop_row(row):
    # Logic to compare prediction vs actual
    prop_cat_clean = str(row.get('Prop Category', '')).strip()
    prop_map_lookup = cfg.MASTER_PROP_MAP.get(prop_cat_clean, prop_cat_clean)
    
    try:
        line = float(row['Prop Line'])
        actual = float(row.get(prop_map_lookup, row.get(prop_cat_clean))) # Try mapped then raw
    except:
        return pd.Series([None, 'Error', None])
        
    if pd.isna(actual): return pd.Series([None, 'Missing Data', None])
    
    res = 'Push'
    if actual > line: res = 'Over'
    elif actual < line: res = 'Under'
    
    correctness = 'Incorrect'
    if res == row['Best Pick']: correctness = 'Correct'
    elif res == 'Push': correctness = 'Push'
    
    return pd.Series([actual, res, correctness])

def grade_predictions():
    # 1. Load Data
    try:
        df_props = pd.read_csv(cfg.PROCESSED_OUTPUT)
        df_box = pd.read_csv(cfg.MASTER_BOX_SCORES_FILE)
    except FileNotFoundError:
        logging.error("Missing processed props or master box scores.")
        return

    # 2. Prep
    df_props = df_props.dropna(subset=['Score']).copy()
    df_props['join_player'] = df_props['Player Name'].apply(text.preprocess_name_for_fuzzy_match)
    # Normalize Matchup/Date
    df_props['join_date'] = pd.to_datetime(df_props['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
    df_box['join_player'] = df_box['PLAYER_NAME'].apply(text.preprocess_name_for_fuzzy_match)
    df_box['join_date'] = pd.to_datetime(df_box['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
    df_box = calculate_derived_stats(df_box)

    # 3. Merge
    df_merged = pd.merge(
        df_props, df_box, 
        on=['join_player', 'join_date'], 
        how='left', suffixes=('', '_box')
    )

    # 4. Grade
    cols = ['Actual Value', 'Result (O/U/P)', 'Correctness']
    df_merged[cols] = df_merged.apply(check_prop_row, axis=1)
    
    # --- SORTING LOGIC ---
    tier_order = {'S Tier': 0, 'A Tier': 1, 'B Tier': 2, 'C Tier': 3}
    if 'Tier' in df_merged.columns:
        df_merged['sort_idx'] = df_merged['Tier'].map(tier_order).fillna(99)
        
        sort_cols = ['sort_idx']
        ascending_order = [True]
        
        if 'Win_Prob' in df_merged.columns:
            sort_cols.append('Win_Prob')
            ascending_order.append(False)
            
        df_merged = df_merged.sort_values(by=sort_cols, ascending=ascending_order)
        df_merged = df_merged.drop(columns=['sort_idx'])

    # 5. Save
    out_file = cfg.OUTPUT_DIR / "prop_check_results.csv"
    df_merged.to_csv(out_file, index=False)
    logging.info(f"Graded results saved to {out_file}")
    
    # 6. Summary Report
    # Filter strictly for Correct/Incorrect (Exclude Pushes/Errors)
    graded = df_merged[df_merged['Correctness'].isin(['Correct', 'Incorrect'])]
    
    def print_stat_line(subset, label):
        total = len(subset)
        if total > 0:
            correct = len(subset[subset['Correctness'] == 'Correct'])
            acc = (correct / total) * 100
            logging.info(f"Accuracy on {total} {label}: {acc:.2f}% ({correct}/{total})")
        else:
            logging.info(f"Accuracy on 0 {label}: N/A")

    logging.info("-" * 40)
    logging.info("PERFORMANCE SUMMARY")
    
    # Overall Stats
    print_stat_line(graded, "Total props")
    
    # S-Tier Stats
    s_tier_props = graded[graded['Tier'] == 'S Tier']
    print_stat_line(s_tier_props, "S-Tier props")
    
    logging.info("-" * 40)

    # 2. Prep
    df_props = df_props.dropna(subset=['Score']).copy()
    df_props['join_player'] = df_props['Player Name'].apply(text.preprocess_name_for_fuzzy_match)
    # Normalize Matchup/Date logic as needed... simplified here:
    df_props['join_date'] = pd.to_datetime(df_props['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
    df_box['join_player'] = df_box['PLAYER_NAME'].apply(text.preprocess_name_for_fuzzy_match)
    df_box['join_date'] = pd.to_datetime(df_box['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
    df_box = calculate_derived_stats(df_box)

    # 3. Merge
    # Note: In a real scenario, you might want to merge on Matchup too, 
    # but Player + Date is usually unique enough for NBA
    df_merged = pd.merge(
        df_props, df_box, 
        on=['join_player', 'join_date'], 
        how='left', suffixes=('', '_box')
    )

    # 4. Grade
    cols = ['Actual Value', 'Result (O/U/P)', 'Correctness']
    df_merged[cols] = df_merged.apply(check_prop_row, axis=1)
    
    # 5. Save
    out_file = cfg.OUTPUT_DIR / "prop_check_results.csv"
    df_merged.to_csv(out_file, index=False)
    logging.info(f"Graded results saved to {out_file}")
    
    # 6. Summary
    graded = df_merged[df_merged['Correctness'].isin(['Correct', 'Incorrect'])]
    if len(graded) > 0:
        acc = (len(graded[graded['Correctness']=='Correct']) / len(graded)) * 100
        logging.info(f"Accuracy on {len(graded)} props: {acc:.2f}%")