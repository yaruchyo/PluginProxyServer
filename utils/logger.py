import loguru
import sys

# Configure Loguru
logger = loguru.logger
logger.remove() # Remove default handlers
logger.add(
    sys.stderr,
    level="INFO", # Set your desired default level
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    colorize=True,
)

# You can add file logging or other configurations here if needed
# logger.add("file_{time}.log", rotation="1 week")
