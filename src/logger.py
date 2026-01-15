import logging
import os
from datetime import datetime



# Log file path setup
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
LOG_FILE_PATH = os.path.join(os.getcwd(), "logs", LOG_FILE)
os.makedirs(os.path.dirname(LOG_FILE_PATH), exist_ok=True)

logging.basicConfig(
    filename=LOG_FILE_PATH, 
    format='[%(asctime)s] [Line:%(lineno)d] %(name)s - %(levelname)s - %(message)s',  # Correct format syntax
    level=logging.INFO 
)

if __name__ == "__main__":
    logging.info("Logging has started")
