import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import asyncio
import os
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.models import model_manager
from src.tools.deep_analyzer import DeepAnalyzerTool

if __name__ == "__main__":
    model_manager.init_models(use_local_proxy=True)

    deep_analyzer = DeepAnalyzerTool()

    task = """
    If this whole pint is made up of ice cream, how many percent above or below the US federal standards for butterfat content is it when using the standards as reported by Wikipedia in 2020? Answer as + or - a number rounded to one decimal place.
    """

    response = asyncio.run(
        deep_analyzer.forward(
            task=task,
            source=os.path.join(
                root,
                "data/GAIA/2023/validation/b2c257e0-3ad7-4f05-b8e3-d9da973be36e.jpg",
            ),
        )
    )

    print(response)
