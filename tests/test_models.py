import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import asyncio
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[1])
sys.path.append(root)

from src.models import model_manager

if __name__ == "__main__":
    model_manager.init_models(use_local_proxy=True)

    messages = [{"role": "user", "content": "What is the capital of France?"}]

    response = asyncio.run(
        model_manager.registed_models["gpt-4.1"](
            messages=messages,
        )
    )
    print(response)

    # test langchain models
    model = model_manager.registed_models["langchain-gpt-4.1"]
    response = asyncio.run(model.ainvoke("What is the capital of France?"))
    print(response)
