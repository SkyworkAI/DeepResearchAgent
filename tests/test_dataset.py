import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import os
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.dataset.huggingface import HLEDataset
from src.logger import logger

if __name__ == "__main__":
    logger.init_logger("tmp.log")

    dataset = HLEDataset(
        path=os.path.join(root, "data", "hle"), split="test", name="hle"
    )
    print(len(dataset))
