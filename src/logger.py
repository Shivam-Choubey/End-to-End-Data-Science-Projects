import logging
import os
from datetime import datetime

# 1. Create a unique log file name based on the current timestamp (Month_Day_Year_Hour_Minute_Second)
LOG_FILE = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"

# 2. Define the path for the 'logs' directory within the current working directory
# It creates a subfolder for each log session based on the filename above
logs_path = os.path.join(os.getcwd(), "logs", LOG_FILE)

# 3. Create the directory if it doesn't exist. 
# exist_ok=True prevents the code from crashing if the folder is already there.
os.makedirs(logs_path, exist_ok=True)

# 4. Define the full path for the log file (inside the new directory)
LOG_FILE_PATH = os.path.join(logs_path, LOG_FILE)

# 5. Configure the logging settings
logging.basicConfig(
    filename=LOG_FILE_PATH,
    # Format: [Time] Line_Number Name - Level - Message
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    # Set the minimum logging level to INFO (ignores DEBUG messages)
    level=logging.INFO,
)

if __name__ == "__main__":
    # Test message to ensure the logger is working properly
    logging.info("Logging has started")