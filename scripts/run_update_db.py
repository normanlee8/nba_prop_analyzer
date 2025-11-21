import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from prop_analyzer.data import dataset
from prop_analyzer.utils.common import setup_logging

if __name__ == "__main__":
    setup_logging(name="dataset_update")
    dataset.update_master_dataset()