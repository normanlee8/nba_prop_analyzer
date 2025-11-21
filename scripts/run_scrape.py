import sys
import logging
from pathlib import Path

# Add project root to path so we can import prop_analyzer
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.data import scraper  # <--- Correct import
from prop_analyzer.utils.common import setup_logging

def main():
    setup_logging(name="scraper", log_file="scraper.log")
    logging.info(">>> Starting Data Scraper Script")
    
    # Run the scraper logic
    scraper.main()

if __name__ == "__main__":
    main()