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

def load_master_box_scores():
    """
    Loads all available master box score Parquet files to find actual results.
    """
    logging.info("Loading master box scores (Parquet)...")
    files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_BOX_SCORES_PATTERN))
    if not files:
        logging.error("No master box scores found. Cannot grade.")
        return pd.DataFrame()
        
    dfs = []
    for f in files:
        try:
            dfs.append(pd.read_parquet(f))
        except Exception as e:
            logging.warning(f"Failed to read {f}: {e}")
            
    if not dfs: return pd.DataFrame()
    
    # Combine and deduplicate (just in case)
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Standardize Date
    if Cols.DATE in full_df.columns:
        full_df[Cols.DATE] = pd.to_datetime(full_df[Cols.DATE]).dt.normalize()
    elif 'GAME_DATE' in full_df.columns:
        full_df[Cols.DATE] = pd.to_datetime(full_df['GAME_DATE']).dt.normalize()
        
    # Create unique key for lookup: PlayerID + Date
    # Note: We rely on Player Name matching if ID is missing in props file, 
    # but ID + Date is safer.
    return full_df

def grade_predictions():
    """
    Reads the last generated predictions (processed_props.csv) and checks
    if they won or lost based on master box scores.
    """
    # 1. Load Predictions
    preds_path = cfg.PROCESSED_OUTPUT_CSV
    if not preds_path.exists():
        logging.critical(f"No predictions file found at {preds_path}")
        return

    try:
        preds_df = pd.read_csv(preds_path)
        if preds_df.empty:
            logging.warning("Predictions file is empty.")
            return
    except Exception as e:
        logging.critical(f"Failed to load predictions: {e}")
        return

    # 2. Load Truth Data (Box Scores)
    truth_df = load_master_box_scores()
    if truth_df.empty:
        return

    logging.info(f"Grading {len(preds_df)} predictions against {len(truth_df)} historical records...")

    # Standardize Dates in Preds
    preds_df['Match_Date'] = pd.to_datetime(preds_df[Cols.DATE]).dt.normalize()
    
    # Standardize Player Names for Join (if ID not present)
    if Cols.PLAYER_ID not in preds_df.columns:
         preds_df['clean_name'] = preds_df[Cols.PLAYER_NAME].astype(str).str.lower().str.strip()
         truth_df['clean_name'] = truth_df[Cols.PLAYER_NAME].astype(str).str.lower().str.strip()
         join_cols = ['clean_name', Cols.DATE]
         truth_cols = ['clean_name', Cols.DATE]
    else:
         join_cols = [Cols.PLAYER_ID, 'Match_Date']
         truth_cols = [Cols.PLAYER_ID, Cols.DATE]

    # 3. Grading Logic
    results = []
    
    # Map Prop Types to Data Columns (using config map)
    # Reverse the map to find Data Col -> Prop Type? No, we need Prop Type -> Data Col
    # We can use the PROP_KEY_MAP from training.py concept, or just the config map
    prop_map = cfg.MASTER_PROP_MAP
    
    for idx, row in preds_df.iterrows():
        # Find matching game
        # Simple filter is safer than merge for row-by-row grading logic
        p_name = str(row.get(Cols.PLAYER_NAME, '')).lower().strip()
        p_date = row['Match_Date']
        
        # Look up player in truth df
        # Try by ID first if available
        if Cols.PLAYER_ID in row and pd.notna(row[Cols.PLAYER_ID]):
             mask = (truth_df[Cols.PLAYER_ID] == row[Cols.PLAYER_ID]) & (truth_df[Cols.DATE] == p_date)
        else:
             # Fallback to name
             mask = (truth_df['PLAYER_NAME'].str.lower().str.strip() == p_name) & (truth_df[Cols.DATE] == p_date)
             
        match = truth_df[mask]
        
        if match.empty:
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = 'Pending'
            results.append(row)
            continue
            
        # Get Actual Stat
        prop_type = row[Cols.PROP_TYPE]
        data_col = prop_map.get(prop_type, prop_type) # Get 'PTS' from 'Points'
        
        actual = match.iloc[0].get(data_col)
        
        if pd.isna(actual):
            row[Cols.ACTUAL_VAL] = None
            row[Cols.RESULT] = 'Unknown Stat'
            results.append(row)
            continue
            
        row[Cols.ACTUAL_VAL] = actual
        
        # Determine Win/Loss
        line = row[Cols.PROP_LINE]
        pick = row[Cols.EDGE_TYPE] # 'Over' or 'Under'
        
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
            
        row[Cols.RESULT] = res
        row[Cols.CORRECTNESS] = 1 if res == 'WIN' else 0
        results.append(row)

    graded_df = pd.DataFrame(results)
    
    # 4. Save Graded Output
    # Save to records folder with date
    today_str = datetime.now().strftime("%Y-%m-%d")
    record_path = cfg.INPUT_DIR / "records" / f"graded_props_{today_str}.csv"
    
    graded_df.to_csv(record_path, index=False)
    logging.info(f"Graded results saved to: {record_path}")
    
    # Summary
    if 'Result' in graded_df.columns:
        summary = graded_df['Result'].value_counts()
        logging.info(f"Summary: {summary.to_dict()}")

def main():
    common.setup_logging(name="grading")
    grade_predictions()

if __name__ == "__main__":
    main()