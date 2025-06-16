import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import asyncio
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(verbose=True)

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.models import model_manager
from src.tools.auto_browser import AutoBrowserUseTool

if __name__ == "__main__":
    model_manager.init_models(use_local_proxy=True)

    tool = AutoBrowserUseTool()

    loop = asyncio.get_event_loop()

    task1 = "Find the minimum perigee value (closest approach distance) between the Earth and the Moon on the Wikipedia page for the Moon."
    res = loop.run_until_complete(tool.forward(task=task1))
    print(res)

    task2 = "Eliud Kipchoge's marathon world record time and pace"
    res = loop.run_until_complete(tool.forward(task=task2))
    print(res)
