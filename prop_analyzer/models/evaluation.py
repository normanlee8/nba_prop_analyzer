import pandas as pd
import logging
from prop_analyzer import config as cfg
from prop_analyzer.utils import text

def calculate_derived_stats(df):
    """
    Calculates composite stats (PRA, PA, etc.) from raw box score columns.
    Includes Quarters and Halves.
    """
    # Derived Quarter Stats
    for q in ['Q1', 'Q2', 'Q3', 'Q4']:
        pts, reb, ast = f'{q}_PTS', f'{q}_REB', f'{q}_AST'
        # Only calculate if all components exist (prevent partial sums)
        if pts in df.columns and reb in df.columns and ast in df.columns:
            df[f'{q}_PRA'] = df[pts] + df[reb] + df[ast]
            df[f'{q}_PR'] = df[pts] + df[reb]
            df[f'{q}_PA'] = df[pts] + df[ast]
            df[f'{q}_RA'] = df[reb] + df[ast]

    # Derived Halves (1H = Q1 + Q2)
    base_stats = ['PTS', 'REB', 'AST', 'PRA', 'PR', 'PA', 'RA', 'STL', 'BLK', 'TOV', 'FG3M']
    for stat in base_stats:
        q1_col = f'Q1_{stat}'
        q2_col = f'Q2_{stat}'
        
        # For composite stats like Q1_PRA, they might have just been created above.
        # Ensure they exist before summing.
        if q1_col in df.columns and q2_col in df.columns:
            df[f'1H_{stat}'] = df[q1_col] + df[q2_col]
            
    return df

def check_prop_row(row):
    """
    Compares the prediction against the actual value to determine correctness.
    """
    # Resolve actual column name
    prop_cat_clean = str(row.get('Prop Category', '')).strip()
    # Use the map to find the DB column name (e.g. 'Points' -> 'PTS')
    prop_map_lookup = cfg.MASTER_PROP_MAP.get(prop_cat_clean, prop_cat_clean)
    
    try:
        line = float(row['Prop Line'])
        # Try getting the mapped column first (e.g. 'PTS'), then fallback to raw
        actual = row.get(prop_map_lookup)
        if pd.isna(actual):
            actual = row.get(prop_cat_clean)
            
        if actual is not None:
            actual = float(actual)
    except (ValueError, TypeError):
        return pd.Series([None, 'Error', None])
        
    if pd.isna(actual): 
        return pd.Series([None, 'Missing Data', None])
    
    # Determine Result
    if actual > line: 
        res = 'Over'
    elif actual < line: 
        res = 'Under'
    else:
        res = 'Push'
    
    # Determine Correctness
    # Best Pick: 'Over' or 'Under'
    correctness = 'Incorrect'
    if res == 'Push':
        correctness = 'Push'
    elif res == row['Best Pick']:
        correctness = 'Correct'
    
    return pd.Series([actual, res, correctness])

def grade_predictions():
    logging.info("--- Grading Predictions vs Actuals ---")
    
    # 1. Load Data
    try:
        if not cfg.PROCESSED_OUTPUT.exists():
            logging.warning("No processed props file found to grade.")
            return
            
        df_props = pd.read_csv(cfg.PROCESSED_OUTPUT)
        
        if not cfg.MASTER_BOX_SCORES_FILE.exists():
            logging.warning("No master box scores found. Cannot grade.")
            return
            
        df_box = pd.read_csv(cfg.MASTER_BOX_SCORES_FILE, low_memory=False)
    except Exception as e:
        logging.error(f"Error loading files for grading: {e}")
        return

    if df_props.empty:
        logging.warning("Props file is empty.")
        return

    # 2. Prep & Normalize
    # Filter out rows without scores/predictions if any
    if 'Score' in df_props.columns:
        df_props = df_props.dropna(subset=['Score']).copy()
    
    # Create join keys
    df_props['join_player'] = df_props['Player Name'].apply(text.preprocess_name_for_fuzzy_match)
    # Assume 'GAME_DATE' in props is YYYY-MM-DD
    df_props['join_date'] = pd.to_datetime(df_props['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
    df_box['join_player'] = df_box['PLAYER_NAME'].apply(text.preprocess_name_for_fuzzy_match)
    df_box['join_date'] = pd.to_datetime(df_box['GAME_DATE']).dt.strftime('%Y-%m-%d')
    
    # Calculate derived stats on the box scores so we can grade things like "1H Points"
    df_box = calculate_derived_stats(df_box)

    # 3. Merge
    # Inner join would lose props we can't find box scores for. Left join keeps them so we see "Missing Data".
    df_merged = pd.merge(
        df_props, 
        df_box, 
        on=['join_player', 'join_date'], 
        how='left', 
        suffixes=('', '_box')
    )

    # 4. Grade
    cols = ['Actual Value', 'Result (O/U/P)', 'Correctness']
    df_merged[cols] = df_merged.apply(check_prop_row, axis=1)
    
    # --- SORTING LOGIC ---
    # Sort by Tier (S -> A -> B) then by Win Probability
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

    # 5. Save Results
    out_file = cfg.OUTPUT_DIR / "prop_check_results.csv"
    try:
        df_merged.to_csv(out_file, index=False)
        logging.info(f"Graded results saved to {out_file}")
    except Exception as e:
        logging.error(f"Failed to save grading results: {e}")
    
    # 6. Performance Summary Report
    # Filter strictly for Correct/Incorrect (Exclude Pushes/Missing)
    graded = df_merged[df_merged['Correctness'].isin(['Correct', 'Incorrect'])]
    
    def log_accuracy(subset, label):
        total = len(subset)
        if total > 0:
            correct = len(subset[subset['Correctness'] == 'Correct'])
            acc = (correct / total) * 100
            logging.info(f"Accuracy on {total} {label}: {acc:.2f}% ({correct}/{total})")
        else:
            logging.info(f"Accuracy on 0 {label}: N/A")

    logging.info("-" * 40)
    logging.info("PERFORMANCE SUMMARY")
    
    log_accuracy(graded, "Total Graded Props")
    
    # S-Tier Stats
    if 'Tier' in graded.columns:
        s_tier = graded[graded['Tier'] == 'S Tier']
        log_accuracy(s_tier, "S-Tier Props")
        
        a_tier = graded[graded['Tier'] == 'A Tier']
        log_accuracy(a_tier, "A-Tier Props")
    
    logging.info("-" * 40)