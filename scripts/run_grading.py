import sys
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.utils import common
from prop_analyzer.data import loader

def print_accuracy_report(df, label="Total"):
    """Helper to print formatted percentage stats"""
    total = len(df)
    if total == 0:
        return

    wins = len(df[df[Cols.RESULT] == 'WIN'])
    losses = len(df[df[Cols.RESULT] == 'LOSS'])
    pushes = len(df[df[Cols.RESULT] == 'PUSH'])
    
    decided = wins + losses
    if decided > 0:
        acc = (wins / decided) * 100
        logging.info(f"{label}: {acc:.1f}% ({wins}/{decided}) [Pushes: {pushes}]")
    else:
        logging.info(f"{label}: N/A (Only Pushes)")

def save_user_scorecard(df, date_str):
    """
    Saves a clean, human-readable Excel file for the user.
    Applies conditional formatting: WIN = Green, LOSS = Red.
    """
    user_cols = [
        'Player Name', 'Team', 'Opponent', 'Prop Category', 'Prop Line', 'GAME_DATE', 
        'Model_Pred', 'Model_Conf', 'Edge_Type', 'Tier', 'Diff%', 'Actual Value', 'Result'
    ]
    
    rename_map = {
        'Player Name': 'Player',
        'Prop Category': 'Prop',
        'Prop Line': 'Line',
        'GAME_DATE': 'Date',
        'Model_Pred': 'Proj',
        'Model_Conf': 'Prob',
        'Edge_Type': 'Pick',
        'Actual Value': 'Actual'
    }
    
    available_cols = [c for c in user_cols if c in df.columns]
    
    clean_df = df[available_cols].copy()
    clean_df = clean_df.rename(columns=rename_map)
    
    # Save to specific subfolder: user_scorecards
    scorecard_dir = cfg.GRADED_DIR / "user_scorecards"
    scorecard_dir.mkdir(parents=True, exist_ok=True)
    
    scorecard_path = scorecard_dir / f"{date_str}.xlsx"
    
    try:
        # --- STYLING LOGIC ---
        def color_result(val):
            """Colors the text based on Win/Loss"""
            if val == 'WIN':
                return 'color: #008000; font-weight: bold' # Green
            elif val == 'LOSS':
                return 'color: #FF0000; font-weight: bold' # Red
            elif val == 'PUSH':
                return 'color: #808080; font-weight: bold' # Gray
            return ''

        # Fix for FutureWarning: Use .map instead of .applymap for newer pandas versions
        if 'Result' in clean_df.columns:
            try:
                styler = clean_df.style.map(color_result, subset=['Result'])
            except AttributeError:
                # Fallback for older pandas versions
                styler = clean_df.style.applymap(color_result, subset=['Result'])
        else:
            styler = clean_df.style

        # Save to Excel
        styler.to_excel(scorecard_path, index=False, engine='openpyxl')
        
    except Exception as e:
        logging.error(f"Failed to save user scorecard: {e}")

def grade_predictions():
    # 1. Load Predictions
    preds_path = cfg.PROCESSED_OUTPUT_SYSTEM
    if not preds_path.exists():
        logging.critical(f"No predictions file found at {preds_path}")
        return

    try:
        preds_df = pd.read_parquet(preds_path)
        if preds_df.empty:
            logging.warning("Predictions file is empty.")
            return
    except Exception as e:
        logging.critical(f"Failed to load predictions: {e}")
        return

    # 2. Load Truth Data
    logging.info("Loading historical data for grading...")
    
    full_game_df = loader.load_box_scores()
    q1_game_df = loader.load_master_q1_history()
    h1_game_df = loader.load_master_1h_history()

    # --- ADDED: Calculate Combo Stats for Grading ---
    # Ensure Q1/1H combo stats exist for grading
    if not q1_game_df.empty:
        # Fill missing base stats with 0 just in case
        for col in ['PTS', 'REB', 'AST']:
            if col not in q1_game_df.columns: q1_game_df[col] = 0
            
        q1_game_df['PRA'] = q1_game_df['PTS'] + q1_game_df['REB'] + q1_game_df['AST']
        q1_game_df['PR'] = q1_game_df['PTS'] + q1_game_df['REB']
        q1_game_df['PA'] = q1_game_df['PTS'] + q1_game_df['AST']
        q1_game_df['RA'] = q1_game_df['REB'] + q1_game_df['AST']

    if not h1_game_df.empty:
        for col in ['PTS', 'REB', 'AST']:
            if col not in h1_game_df.columns: h1_game_df[col] = 0
            
        h1_game_df['PRA'] = h1_game_df['PTS'] + h1_game_df['REB'] + h1_game_df['AST']
        h1_game_df['PR'] = h1_game_df['PTS'] + h1_game_df['REB']
        h1_game_df['PA'] = h1_game_df['PTS'] + h1_game_df['AST']
        h1_game_df['RA'] = h1_game_df['REB'] + h1_game_df['AST']
    # ------------------------------------------------

    if full_game_df is None or full_game_df.empty:
        logging.warning("No master box scores found. Full game props cannot be graded.")
        full_game_df = pd.DataFrame()

    logging.info(f"Grading {len(preds_df)} predictions...")

    for df in [full_game_df, q1_game_df, h1_game_df]:
        if not df.empty and Cols.DATE in df.columns:
            df[Cols.DATE] = pd.to_datetime(df[Cols.DATE]).dt.normalize()

    preds_df['Match_Date'] = pd.to_datetime(preds_df[Cols.DATE]).dt.normalize()
    prop_map = cfg.MASTER_PROP_MAP
    
    results = []
    
    for idx, row in preds_df.iterrows():
        prop_type = row[Cols.PROP_TYPE]
        p_date = row['Match_Date']
        
        prop_key = str(prop_map.get(prop_type, prop_type))
        
        is_q1 = '1st Quarter' in prop_type or '1Q' in prop_type or prop_key.startswith('Q1_')
        is_1h = '1st Half' in prop_type or '1H' in prop_type or prop_key.startswith('1H_')
        
        if is_q1:
            truth_df = q1_game_df
            data_col = prop_key.replace('Q1_', '')
        elif is_1h:
            truth_df = h1_game_df
            data_col = prop_key.replace('1H_', '')
        else:
            truth_df = full_game_df
            data_col = prop_key

        if truth_df.empty:
            row[Cols.RESULT] = 'Missing Data Source'
            results.append(row)
            continue

        mask = None
        if Cols.PLAYER_ID in row and pd.notna(row[Cols.PLAYER_ID]):
             p_id = int(row[Cols.PLAYER_ID])
             if Cols.PLAYER_ID in truth_df.columns:
                 mask = (truth_df[Cols.PLAYER_ID] == p_id) & (truth_df[Cols.DATE] == p_date)
        
        if mask is None or mask.sum() == 0:
             p_name = str(row.get(Cols.PLAYER_NAME, '')).lower().strip()
             if 'PLAYER_NAME' in truth_df.columns:
                 mask = (truth_df['PLAYER_NAME'].str.lower().str.strip() == p_name) & (truth_df[Cols.DATE] == p_date)
             
        if mask is not None:
            match = truth_df[mask]
        else:
            match = pd.DataFrame()
        
        if match.empty:
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = 'Pending / Not Found'
            results.append(row)
            continue
            
        if data_col not in match.columns:
             if 'Points' in prop_type: data_col = 'PTS'
             elif 'Rebounds' in prop_type: data_col = 'REB'
             elif 'Assists' in prop_type: data_col = 'AST'

        if data_col not in match.columns:
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = f'Stat {data_col} Missing'
            results.append(row)
            continue

        actual = match.iloc[0].get(data_col)
        
        if pd.isna(actual):
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = 'Stat is NaN'
            results.append(row)
            continue
            
        row[Cols.ACTUAL_VAL] = actual
        
        try:
            line = float(row[Cols.PROP_LINE])
            pick = row[Cols.EDGE_TYPE]
            
            if pick == 'Over':
                if actual > line: res = 'WIN'
                elif actual < line: res = 'LOSS'
                else: res = 'PUSH'
            elif pick == 'Under':
                if actual < line: res = 'WIN'
                elif actual > line: res = 'LOSS'
                else: res = 'PUSH'
            else:
                res = 'ERROR'
        except Exception:
            res = 'Error Grading'
            
        row[Cols.RESULT] = res
        row[Cols.CORRECTNESS] = 1 if res == 'WIN' else 0
        results.append(row)

    graded_df = pd.DataFrame(results)
    
    # 7. Reporting
    logging.info("-" * 40)
    logging.info(">>> GRADING REPORT <<<")
    
    if graded_df.empty:
        logging.warning("No results to grade.")
        return

    finished = graded_df[graded_df[Cols.RESULT].isin(['WIN', 'LOSS', 'PUSH'])]
    
    print_accuracy_report(finished, "Total Props")
    
    if Cols.TIER in finished.columns:
        for tier in ['S Tier', 'A Tier', 'B Tier']:
            tier_df = finished[finished[Cols.TIER] == tier]
            print_accuracy_report(tier_df, f"{tier} Props")

    q1_props = finished[finished[Cols.PROP_TYPE].str.contains('1st Quarter|1Q', na=False)]
    if not q1_props.empty:
        print_accuracy_report(q1_props, "1st Quarter Props")
        
    h1_props = finished[finished[Cols.PROP_TYPE].str.contains('1st Half|1H', na=False)]
    if not h1_props.empty:
        print_accuracy_report(h1_props, "1st Half Props")

    logging.info("-" * 40)
    
    # 8. Save Outputs
    today_str = datetime.now().strftime("%Y-%m-%d")
    
    parquet_path = cfg.GRADED_DIR / f"graded_props_{today_str}.parquet"
    csv_path = cfg.GRADED_DIR / f"graded_{today_str}.csv"
    
    try:
        for col in graded_df.select_dtypes(include=['object']).columns:
            graded_df[col] = graded_df[col].astype(str)
            
        graded_df.to_parquet(parquet_path, index=False)
        graded_df.to_csv(csv_path, index=False)
        
        save_user_scorecard(graded_df, today_str)
        
    except Exception as e:
        logging.error(f"Failed to save output: {e}")

def main():
    common.setup_logging(name="grading")
    grade_predictions()

if __name__ == "__main__":
    main()