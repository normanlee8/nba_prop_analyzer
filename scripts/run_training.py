import sys
import pandas as pd
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.models import training
from prop_analyzer.features import definitions
from prop_analyzer.utils.common import setup_logging

MASTER_TRAIN_FILE = "master_training_dataset.csv"

def load_training_dataset():
    # Try multiple locations
    paths = [
        cfg.DATA_DIR / MASTER_TRAIN_FILE,
        Path(MASTER_TRAIN_FILE)
    ]
    for p in paths:
        if p.exists():
            return pd.read_csv(p, low_memory=False)
    return None

def main():
    setup_logging(name="training", log_file="training.log")
    logging.info(">>> Starting Model Training Pipeline")

    df = load_training_dataset()
    if df is None:
        logging.critical(f"FATAL: {MASTER_TRAIN_FILE} not found.")
        return

    # Clean Data
    df = df.dropna(subset=['Actual Value', 'Prop Line'])
    df['Actual Value'] = pd.to_numeric(df['Actual Value'], errors='coerce')
    df['Prop Line'] = pd.to_numeric(df['Prop Line'], errors='coerce')

    # Filter for minimum games if column exists
    if 'SZN Games' in df.columns:
        df = df[df['SZN Games'] >= 5]

    # Train loop
    props_to_train = definitions.PROP_FEATURE_MAP.keys()
    
    for prop_cat in props_to_train:
        # Filter dataset for this specific prop type
        prop_df = df[df['Prop Category'] == prop_cat].copy()
        
        if len(prop_df) < 100:
            logging.warning(f"Skipping {prop_cat}: Insufficient data ({len(prop_df)} rows)")
            continue
            
        try:
            training.train_single_prop(prop_df, prop_cat)
        except Exception as e:
            logging.error(f"Failed to train {prop_cat}: {e}")

    logging.info("<<< Training Complete >>>")

if __name__ == "__main__":
    main()