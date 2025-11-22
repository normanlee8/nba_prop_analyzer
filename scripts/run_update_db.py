import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import dataset
from prop_analyzer.utils import common

def main():
    common.setup_logging(name="update_db")
    logging.info(">>> STARTING DATASET BUILD (No Scrape/ETL)")
    
    # Build Dataset Only
    try:
        logging.info("Building Training Dataset...")
        dataset.create_training_dataset()
    except Exception as e:
        logging.error(f"Dataset build failed: {e}", exc_info=True)
        
    logging.info("<<< DATASET BUILD COMPLETE")

if __name__ == "__main__":
    main()