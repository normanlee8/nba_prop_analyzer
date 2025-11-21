import logging
import sys
from datetime import datetime

def setup_logging(name="prop_analyzer", log_file=None, level=logging.INFO):
    """
    Sets up a standardized logger for the application.
    """
    logger = logging.getLogger()
    
    # Clear existing handlers to avoid duplicates if called multiple times
    if logger.hasHandlers():
        logger.handlers.clear()
        
    logger.setLevel(level)
    
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    
    # Console Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File Handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
    return logger

def get_nba_season_id(date_obj):
    """
    Determines the NBA Season ID (e.g., 22025) for a given date.
    NBA Seasons typically start in October.
    """
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.strptime(date_obj, "%Y-%m-%d")
        except:
            return 22025 # Fallback
            
    year = date_obj.year
    month = date_obj.month
    
    # If month is Oct (10), Nov (11), Dec (12), the season started in this year.
    # NBA API Format: 2 + Year (e.g., 2025-26 Season -> 22025)
    if month >= 9: # September buffers preseason
        season_year = year
    else:
        season_year = year - 1
        
    return int(f"2{season_year}")