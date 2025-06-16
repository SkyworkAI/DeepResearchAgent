import warnings

warnings.simplefilter("ignore", DeprecationWarning)

import asyncio
import sys
from pathlib import Path

root = str(Path(__file__).resolve().parents[0])
sys.path.append(root)

from src.agent import create_agent
from src.config import config
from src.logger import logger
from src.models import model_manager
from src.utils import assemble_project_path


async def main():
    # Init config and logger
    config.init_config(config_path=assemble_project_path("./configs/config_qwen.toml"))
    logger.init_logger(config.log_path)
    logger.info(f"Initializing logger: {config.log_path}")
    logger.info(f"Load config: {config}")

    # Registed models
    model_manager.init_models(use_local_proxy=config.use_local_proxy)
    logger.info("Registed models: %s", ", ".join(model_manager.registed_models.keys()))

    # Create agent
    agent = create_agent()

    # Default task
    default_task = "Use deep_researcher_agent to search the latest papers on the topic of 'AI Agent' and then summarize it."

    # Interactive loop
    print("Enter your task (press Enter to use default, type 'exit' to quit):")
    while True:
        task = input(">>> ") or default_task
        if task.lower() in ["exit", "quit"]:
            print("Exiting.")
            break
        res = await agent.run(task)
        logger.info(f"Result: {res}")
        print(f"Result:\n{res}\n")


if __name__ == "__main__":
    asyncio.run(main())
