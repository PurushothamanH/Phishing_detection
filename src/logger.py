import logging
### logger file is used for storing or log all executions, exceptions in a some particular file for tracking or future resolving process..
import os
from  datetime import datetime

LOG_FILE =f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log" ### .log used to convert txt file to log file
logs_path = os.path.join(os.getcwd(),"logs",LOG_FILE) ### 'logs" comes for bcz that file is log format nd line for storing all log files in log directory..
os.makedirs(logs_path,exist_ok=True) ## used to create directories nd store all log files

LOG_FILE_PATH = os.path.join(logs_path,LOG_FILE)

logging.basicConfig(

    filename=LOG_FILE_PATH,
    format="[ %(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO
)
