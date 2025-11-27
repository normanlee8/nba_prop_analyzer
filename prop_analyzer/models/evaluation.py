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
    # Resolve Prop Category (Handle 'Prop' or 'Prop Category')
    prop_cat_raw = row.get('Prop', row.get('Prop Category', ''))
    prop_cat_clean = str(prop_cat_raw).strip()
    
    # Use the map to find the DB column name (e.g. 'Points' -> 'PTS')
    prop_map_lookup = cfg.MASTER_PROP_MAP.get(prop_cat_clean, prop_cat_clean)
    
    try:
        # Handle 'Line' or 'Prop Line'
        line = float(row.get('Line', row.get('Prop Line', 0)))
        
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
    # Best Pick: 'Over' or 'Under' (Handle 'Pick' or 'Best Pick')
    pick = row.get('Pick', row.get('Best Pick', ''))
    
    correctness = 'Incorrect'
    if res == 'Push':
        correctness = 'Push'
    elif res == pick:
        correctness = 'Correct'
    
    return pd.Series([actual, res, correctness])

def grade_predictions():
    logging.info("--- Grading Predictions vs Actuals ---")
    
    # 1. Load Data
    try:
        if not cfg.PROCESSED_OUTPUT.exists():
            logging.warning(f"No processed props file found to grade at {cfg.PROCESSED_OUTPUT}")
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
    
    # --- DETECT PLAYER COLUMN ---
    props_player_col = None
    possible_player_cols = ['Player Name', 'PLAYER_NAME', 'Player', 'PLAYER']
    for col in possible_player_cols:
        if col in df_props.columns:
            props_player_col = col
            break
            
    if not props_player_col:
        logging.error(f"Could not find Player column in props file. Available columns: {df_props.columns.tolist()}")
        return

    # --- DETECT OR INFER DATE ---
    props_date_col = None
    possible_date_cols = ['GAME_DATE', 'Date', 'date', 'Game Date']
    for col in possible_date_cols:
        if col in df_props.columns:
            props_date_col = col
            break
            
    if not props_date_col:
        logging.info("No Date column found in props file. Attempting to infer date from Box Scores...")
        
        # Strategy: Find the most recent date in the box scores that matches these players
        prop_players = df_props[props_player_col].unique()
        
        # Filter box scores to only these players to find their latest games
        # Using simple matching first (assuming names are relatively standard)
        relevant_box = df_box[df_box['PLAYER_NAME'].isin(prop_players)]
        
        if not relevant_box.empty:
            # Assume the slate date is the most recent date found for these players
            inferred_date = relevant_box['GAME_DATE'].max()
            logging.info(f"Inferred Slate Date: {inferred_date} (derived from most recent matching box scores)")
        else:
            # Fallback: Just take the absolute newest date in the box score file
            inferred_date = df_box['GAME_DATE'].max()
            logging.warning(f"Could not match specific players. Defaulting to most recent box score date: {inferred_date}")
            
        df_props['inferred_date'] = inferred_date
        props_date_col = 'inferred_date'
    else:
        logging.info(f"Using existing Date column: {props_date_col}")

    # Create join keys
    df_props['join_player'] = df_props[props_player_col].apply(text.preprocess_name_for_fuzzy_match)
    df_props['join_date'] = pd.to_datetime(df_props[props_date_col]).dt.strftime('%Y-%m-%d')
    
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
        elif 'Win%' in df_merged.columns: # Handle 'Win%' column name
            sort_cols.append('Win%')
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