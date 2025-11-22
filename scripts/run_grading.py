import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.models import evaluation
from prop_analyzer.utils import common

if __name__ == "__main__":
    common.setup_logging(name="grading")
    evaluation.grade_predictions()