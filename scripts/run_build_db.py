import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.data import etl, loader
from prop_analyzer.utils import common

def main():
    common.setup_logging(name="build_db")
    logging.info(">>> STARTING ETL PIPELINE (Master Files)")

    # 1. Run ETL (Raw Scrapes -> Master CSVs)
    logging.info("Step 1: Creating Master Files...")
    try:
        # Initialize Player Map (Critical first step)
        player_id_map = etl.create_player_id_map(cfg.DATA_DIR)
        if player_id_map is None:
            logging.critical("Failed to create Player ID Map. Aborting.")
            return

        # Process Stats
        etl.process_master_player_stats(cfg.DATA_DIR, player_id_map, cfg.DATA_DIR)
        etl.process_master_team_stats(cfg.DATA_DIR, player_id_map, cfg.DATA_DIR)
        
        # Process Box Scores (With Vacancy Logic)
        etl.process_master_box_scores(cfg.DATA_DIR, player_id_map, cfg.DATA_DIR)
        
        # Process Aggregates
        etl.process_vs_opponent_stats(cfg.DATA_DIR, cfg.DATA_DIR)
        etl.process_dvp_stats(cfg.DATA_DIR)
        
        logging.info("ETL Process Complete.")
        logging.info("Next Step: Run 'scripts/run_update_db.py' to build the training dataset.")
        
    except Exception as e:
        logging.critical(f"ETL Failed: {e}", exc_info=True)
        return

    logging.info("<<< ETL PIPELINE COMPLETE")

if __name__ == "__main__":
    main()