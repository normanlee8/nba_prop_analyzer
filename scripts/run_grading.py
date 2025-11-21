import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.models import evaluation
from prop_analyzer.utils.common import setup_logging

if __name__ == "__main__":
    setup_logging(name="grading")
    evaluation.grade_predictions()