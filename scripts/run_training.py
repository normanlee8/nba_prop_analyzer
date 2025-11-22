import sys
import pandas as pd
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.models import training
from prop_analyzer.utils import common

def main():
    common.setup_logging(name="train_models")
    logging.info(">>> STARTING MODEL TRAINING PIPELINE")

    # 1. Load Training Data
    train_file = cfg.DATA_DIR / "master_training_dataset.csv"
    if not train_file.exists():
        logging.critical(f"Training dataset not found at {train_file}")
        logging.critical("Please run 'scripts/run_build_db.py' first.")
        return

    try:
        logging.info(f"Loading dataset: {train_file}")
        df = pd.read_csv(train_file, low_memory=False)
        
        if df.empty:
            logging.critical("Training dataset is empty.")
            return
            
        logging.info(f"Loaded {len(df)} rows of training data.")
        
    except Exception as e:
        logging.critical(f"Failed to load training data: {e}")
        return

    # 2. Filter Props based on Dataset Availability
    # We strictly train only what exists in the columns.
    available_cols = set(df.columns)
    
    # Intersection: Config vs Actual Data
    props_to_train = [p for p in cfg.SUPPORTED_PROPS if p in available_cols]
    skipped_props = [p for p in cfg.SUPPORTED_PROPS if p not in available_cols]

    if skipped_props:
        logging.info(f"Note: {len(skipped_props)} props excluded from training (Historical data not available).")
        logging.info(f"Excluded: {', '.join(skipped_props)}")

    logging.info(f"Proceeding to train models for {len(props_to_train)} props...")

    # 3. Train Models
    successful = 0
    failed = 0
    
    for prop in props_to_train:
        logging.info(f"--- Training Model: {prop} ---")
        
        # Create specific dataframe for this prop
        prop_df = df.dropna(subset=[prop]).copy()
        prop_df['Actual Value'] = prop_df[prop]
        
        # Create Dummy Prop Line for Classifier Context
        # (Simulates "Over/Under" logic using a rolling average as the line)
        if 'SZN_AVG' not in prop_df.columns:
             prop_df['Prop Line'] = prop_df[prop].rolling(window=5, min_periods=1).mean().shift(1)
             # Drop the initial rows where rolling avg is NaN
             prop_df.dropna(subset=['Prop Line'], inplace=True)
        
        if prop_df.empty:
            logging.warning(f"Skipping {prop}: No valid rows after preprocessing.")
            failed += 1
            continue

        try:
            training.train_single_prop(prop_df, prop)
            successful += 1
        except Exception as e:
            logging.error(f"Failed to train {prop}: {e}", exc_info=True)
            failed += 1

    logging.info(f"<<< TRAINING COMPLETE. Success: {successful}, Failed: {failed}")

if __name__ == "__main__":
    main()