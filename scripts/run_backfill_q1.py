import sys
import pandas as pd
import time
import logging
import random
from pathlib import Path
from datetime import datetime
from nba_api.stats.endpoints import leaguedashplayerstats

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer import config as cfg
from prop_analyzer.config import Cols
from prop_analyzer.utils import common
from prop_analyzer.data import scraper

def get_season_info(date_obj):
    """
    Returns the correct Season ID and output folder based on the date.
    """
    year = date_obj.year
    month = date_obj.month
    
    # Season start logic (Sept/Oct starts new season)
    if month >= 9:
        start_year = year
    else:
        start_year = year - 1
        
    end_year = start_year + 1
    season_id = f"{start_year}-{str(end_year)[-2:]}"
    season_dir = cfg.DATA_DIR / season_id / "q1_logs"
    
    return season_id, season_dir

def get_valid_game_dates():
    """
    Scans existing master box scores to find ONLY dates where games occurred.
    This skips summer, off-days, and holidays automatically.
    """
    valid_dates = set()
    
    # 1. Check Master Files (Fastest)
    master_files = sorted(cfg.DATA_DIR.glob(cfg.MASTER_BOX_SCORES_PATTERN))
    
    # 2. Check Raw Season Files (Fallback if master not built yet)
    if not master_files:
        raw_files = sorted(cfg.DATA_DIR.glob("*/NBA Player Box Scores.parquet"))
        files_to_scan = raw_files
    else:
        files_to_scan = master_files

    if not files_to_scan:
        logging.error("No box score files found! Cannot optimize dates.")
        return []

    logging.info(f"Scanning {len(files_to_scan)} files to find valid game dates...")

    for f in files_to_scan:
        try:
            # Just read the Date column
            # Try 'GAME_DATE' (raw) or Cols.DATE (master)
            df = pd.read_parquet(f)
            
            date_col = None
            if Cols.DATE in df.columns: date_col = Cols.DATE
            elif 'GAME_DATE' in df.columns: date_col = 'GAME_DATE'
            
            if date_col:
                # Convert to strings 'YYYY-MM-DD'
                dates = pd.to_datetime(df[date_col]).dt.strftime('%Y-%m-%d').unique()
                valid_dates.update(dates)
                
        except Exception as e:
            logging.warning(f"Skipping file {f.name}: {e}")

    # Sort dates chronologically
    sorted_dates = sorted(list(valid_dates))
    
    # Filter out dates in the future (just in case)
    today = datetime.now().strftime('%Y-%m-%d')
    sorted_dates = [d for d in sorted_dates if d < today]
    
    return sorted_dates

def main():
    common.setup_logging(name="backfill_q1")
    logging.info(">>> STARTING SMART Q1 BACKFILL (Optimized) <<<")
    
    # 1. Get Exact Game Dates
    target_dates = get_valid_game_dates()
    
    if not target_dates:
        logging.critical("No game dates found in your data. Run 'run_scrape.py' first to populate box scores.")
        return

    logging.info(f"Found {len(target_dates)} active game days. Skipping summer/off-days.")
    
    # 2. Iterate
    for i, date_str in enumerate(target_dates):
        # Convert to objects
        dt_obj = datetime.strptime(date_str, "%Y-%m-%d")
        api_date_str = dt_obj.strftime("%m/%d/%Y") # API needs MM/DD/YYYY
        
        # Get Folder
        season_id, output_dir = get_season_info(dt_obj)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"daily_q1_stats_{date_str}"
        file_path = output_dir / f"{filename}.parquet"
        
        # SKIP if already exists
        if file_path.exists():
            print(f"[{i+1}/{len(target_dates)}] Skipping {date_str} (Already exists)")
            continue
            
        logging.info(f"[{i+1}/{len(target_dates)}] Fetching Q1 for {date_str}...")
        
        try:
            # API Call
            q1_stats = leaguedashplayerstats.LeagueDashPlayerStats(
                period=1,
                date_from_nullable=api_date_str,
                date_to_nullable=api_date_str,
                season=season_id,
                season_type_all_star='Regular Season',
                timeout=20
            )
            df = q1_stats.get_data_frames()[0]
            
            if not df.empty:
                df['GAME_DATE'] = date_str
                scraper.save_clean_parquet(df, filename, output_dir)
            else:
                logging.warning(f"No Q1 stats returned for {date_str} (Odd, since it was in box scores).")
                
            # Sleep to be polite
            time.sleep(0.6 + random.random())
            
        except Exception as e:
            logging.error(f"Failed to fetch {date_str}: {e}")
            time.sleep(2)

    logging.info("<<< BACKFILL COMPLETE. NOW RUN 'run_build_db.py' >>>")

if __name__ == "__main__":
    main()