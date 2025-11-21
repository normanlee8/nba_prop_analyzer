import sys
import logging
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.data import etl  # <--- Correct import
from prop_analyzer.utils.common import setup_logging

def main():
    setup_logging(name="etl", log_file="build_db.log")
    logging.info(">>> Starting Master DB Builder")
    
    # Run the ETL logic
    etl.main()

if __name__ == "__main__":
    main()