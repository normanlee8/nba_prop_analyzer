import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import dataset
from prop_analyzer.utils import common

def main():
    # Setup logging
    common.setup_logging(name="update_db")
    
    try:
        logging.info(">>> STARTING DATASET BUILD (No Scrape/ETL) <<<")
        
        # Build Dataset Only
        # This regenerates the training file with the latest master stats
        logging.info("Building Training Dataset...")
        dataset.create_training_dataset()
        
        logging.info("<<< DATASET BUILD COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Update DB: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()