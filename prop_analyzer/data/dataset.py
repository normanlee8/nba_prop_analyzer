import pandas as pd
import logging
from prop_analyzer import config as cfg

MASTER_FILE = cfg.DATA_DIR / 'master_training_dataset.csv'
NEW_RESULTS_FILE = cfg.OUTPUT_DIR / 'prop_check_results.csv'
UNIQUE_COLS = ['Player Name', 'GAME_DATE', 'Prop Category']

def update_master_dataset():
    if not NEW_RESULTS_FILE.exists():
        logging.error(f"{NEW_RESULTS_FILE} not found.")
        return

    try:
        df_new = pd.read_csv(NEW_RESULTS_FILE)
        df_new = df_new.dropna(subset=['Actual Value', 'Correctness']).copy()
        df_new = df_new[df_new['Correctness'] != 'Push']
        
        if df_new.empty:
            logging.warning("No new valid graded props found.")
            return

        if MASTER_FILE.exists():
            df_master = pd.read_csv(MASTER_FILE, low_memory=False)
            df_combined = pd.concat([df_master, df_new], ignore_index=True)
        else:
            df_combined = df_new

        # Deduplicate (Keep last update)
        before = len(df_combined)
        df_combined = df_combined.drop_duplicates(subset=UNIQUE_COLS, keep='last')
        
        df_combined.to_csv(MASTER_FILE, index=False)
        logging.info(f"Updated Master Dataset. Rows: {len(df_combined)} (+{len(df_combined)-before} dupes removed)")
        
    except Exception as e:
        logging.error(f"Failed to update dataset: {e}")