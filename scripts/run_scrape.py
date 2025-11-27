import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.data import scraper
from prop_analyzer.utils import common

if __name__ == "__main__":
    # This line was missing:
    common.setup_logging(name="scraper")
    scraper.main()