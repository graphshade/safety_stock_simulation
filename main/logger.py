import os
import logging
from logging.handlers import RotatingFileHandler




#logger = logging.getLogger(__name__)
logger = logging.getLogger(name="safetystock")



fileHandler = RotatingFileHandler(os.path.abspath("./logs/logs.log"), backupCount=50, maxBytes=5000000)


fmt = logging.Formatter(
    "%(name)s: %(asctime)s | %(levelname)s | %(filename)s%(lineno)s | %(process)d | >>> %(message)s"
)

fileHandler.setFormatter(fmt)

logger.addHandler(fileHandler)

logger.setLevel(logging.INFO)
