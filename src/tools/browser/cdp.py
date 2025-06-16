import os
import subprocess

import requests
from dotenv import load_dotenv

load_dotenv(verbose=True)

from src.logger import logger


class CDP:
    def __init__(
        self,
        chrome_path: str = None,
        user_data_dir: str = None,
    ):
        self.chrome_path = (
            chrome_path if chrome_path else os.getenv("CHROME_DRIVER_PATH")
        )
        self.user_data_dir = (
            user_data_dir if user_data_dir else os.getenv("CHROME_USER_DATA_PATH")
        )

        self.process = None
        self.process_id = None

    def start(self):
        logger.info(f"Starting Chrome with user data dir: {self.user_data_dir}")

        # start chrome with remote debugging port and get the process id
        # --disable-gpu --no-sandbox --disable-dev-shm-usage
        self.process = subprocess.Popen(
            [
                self.chrome_path,
                "--remote-debugging-port=9222",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                f"--user-data-dir={self.user_data_dir}",
            ],
        )

        self.process_id = self.process.pid
        logger.info(f"Chrome started with pid: {self.process_id}")

    def check(self):
        try:
            response = requests.get("http://localhost:9222/json")
            if response.status_code == 200:
                logger.info("Chrome is running")
                return True
            else:
                logger.error("Chrome is not running")
                return False
        except requests.exceptions.RequestException as e:
            logger.error(f"Error checking Chrome status: {e}")
            return False

    def stop(self):
        logger.info(f"Stopping Chrome with pid: {self.process_id}")
        # stop the chrome process
        self.process.terminate()
        logger.info("Chrome stopped")
