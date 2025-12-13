import sys
import logging
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.models import evaluation
from prop_analyzer.utils import common

def main():
    # Setup logging
    common.setup_logging(name="grading")
    
    try:
        logging.info(">>> STARTING GRADING PIPELINE <<<")
        
        # Execute grading logic
        # This now relies on the standardized Cols schema in evaluation.py
        evaluation.grade_predictions()
        
        logging.info("<<< GRADING COMPLETE >>>")
        
    except Exception as e:
        logging.critical(f"FATAL ERROR in Grading Pipeline: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()