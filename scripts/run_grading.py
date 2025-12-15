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
        logging.info(f"{label}: No bets graded.")
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

def grade_predictions():
    # 1. Load Predictions (Parquet)
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
    
    # A. Full Game Box Scores
    full_game_df = loader.load_box_scores()
    if full_game_df is None or full_game_df.empty:
        logging.warning("No master box scores found. Full game props cannot be graded.")
        full_game_df = pd.DataFrame()

    # B. 1st Quarter Box Scores
    q1_game_df = loader.load_master_q1_history()
    if q1_game_df.empty:
        logging.warning("No master Q1 history found. 1st Quarter props cannot be graded.")

    logging.info(f"Grading {len(preds_df)} predictions...")

    # Standardize Dates in Truth Dataframes
    if not full_game_df.empty and Cols.DATE in full_game_df.columns:
        full_game_df[Cols.DATE] = pd.to_datetime(full_game_df[Cols.DATE]).dt.normalize()
        
    if not q1_game_df.empty and Cols.DATE in q1_game_df.columns:
        q1_game_df[Cols.DATE] = pd.to_datetime(q1_game_df[Cols.DATE]).dt.normalize()

    # Prepare for iteration
    preds_df['Match_Date'] = pd.to_datetime(preds_df[Cols.DATE]).dt.normalize()
    prop_map = cfg.MASTER_PROP_MAP
    
    results = []
    
    for idx, row in preds_df.iterrows():
        # Identify Prop Type & Date
        prop_type = row[Cols.PROP_TYPE]
        p_date = row['Match_Date']
        
        # 3. Determine Source (Q1 vs Full Game)
        is_q1_prop = '1st Quarter' in prop_type or '1Q' in prop_type or str(prop_map.get(prop_type, '')).startswith('Q1_')
        
        if is_q1_prop:
            truth_df = q1_game_df
            # Mapping: Internal 'Q1_PTS' -> Raw File 'PTS'
            internal_col = prop_map.get(prop_type, prop_type)
            if internal_col.startswith('Q1_'):
                data_col = internal_col.replace('Q1_', '') # e.g. PTS, REB, AST
            else:
                data_col = internal_col
        else:
            truth_df = full_game_df
            data_col = prop_map.get(prop_type, prop_type)

        if truth_df.empty:
            row[Cols.RESULT] = 'Missing Data Source'
            results.append(row)
            continue

        # 4. Find Player Match
        # Try ID first, then Name
        mask = None
        if Cols.PLAYER_ID in row and pd.notna(row[Cols.PLAYER_ID]):
             # Ensure ID types match (int vs int)
             p_id = int(row[Cols.PLAYER_ID])
             if Cols.PLAYER_ID in truth_df.columns:
                 mask = (truth_df[Cols.PLAYER_ID] == p_id) & (truth_df[Cols.DATE] == p_date)
        
        if mask is None or mask.sum() == 0:
             # Fallback to Name
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
            
        # 5. Extract Actual Value
        # Fallback for simple name mismatches (e.g. 'Points' vs 'PTS') if mapping failed
        if data_col not in match.columns:
             if prop_type == 'Points': data_col = 'PTS'
             elif prop_type == 'Rebounds': data_col = 'REB'
             elif prop_type == 'Assists': data_col = 'AST'

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
        
        # 6. Determine Win/Loss
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

    # Filter only graded rows
    finished = graded_df[graded_df[Cols.RESULT].isin(['WIN', 'LOSS', 'PUSH'])]
    
    print_accuracy_report(finished, "Total Props")
    
    if Cols.TIER in finished.columns:
        for tier in ['S Tier', 'A Tier', 'B Tier']:
            tier_df = finished[finished[Cols.TIER] == tier]
            print_accuracy_report(tier_df, f"{tier} Props")

    # Q1 Specific Report
    q1_props = finished[finished[Cols.PROP_TYPE].str.contains('1st Quarter|1Q', na=False)]
    if not q1_props.empty:
        print_accuracy_report(q1_props, "1st Quarter Props")

    logging.info("-" * 40)
    
    # 8. Save Graded Output
    today_str = datetime.now().strftime("%Y-%m-%d")
    record_path = cfg.GRADED_DIR / f"graded_props_{today_str}.parquet"
    
    try:
        # Convert objects to string for Parquet safety
        for col in graded_df.select_dtypes(include=['object']).columns:
            graded_df[col] = graded_df[col].astype(str)
            
        graded_df.to_parquet(record_path, index=False)
        logging.info(f"Full graded dataset saved to: {record_path}")
        
        # Optional: Save a small CSV for quick viewing
        csv_view = record_path.with_suffix('.csv')
        graded_df.to_csv(csv_view, index=False)
        
    except Exception as e:
        logging.error(f"Failed to save graded output: {e}")

def main():
    common.setup_logging(name="grading")
    grade_predictions()

if __name__ == "__main__":
    main()