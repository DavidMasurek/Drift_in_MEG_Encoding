import sys
import os
import logging
from datetime import datetime
from pathlib import Path

# Add parent folder of src to path and change cwd
__location__ = Path(__file__).parent.parent
sys.path.append(str(__location__))
os.chdir(__location__)

# Add custom loggers for uncluttered debugging without info and debugging from imported packages
def setup_logger(logger_level):
    CUSTOM_DEBUG_INFO_LEVEL = 23 # between info and warning
    def custom_debug_info(self, message, *args, **kwargs):
        if self.isEnabledFor(CUSTOM_DEBUG_INFO_LEVEL):
            self._log(CUSTOM_DEBUG_INFO_LEVEL, message, args, **kwargs)

    logging.addLevelName(CUSTOM_DEBUG_INFO_LEVEL, "Debug_c")
    logging.Logger.custom_debug = custom_debug_info

    CUSTOM_INFO_LEVEL = 27  # between info and warning
    def custom_info(self, message, *args, **kwargs):
        if self.isEnabledFor(CUSTOM_INFO_LEVEL):
            self._log(CUSTOM_INFO_LEVEL, message, args, **kwargs)

    logging.addLevelName(CUSTOM_INFO_LEVEL, "Info_c")
    logging.Logger.custom_info = custom_info

    logger = logging.getLogger(__name__)
    logging.root.handlers = []
    filename = (
            "logs/pipeline_" + datetime.now().strftime("%d-%m-%Y_%H-%M-%S") + ".log"
        )
    handlers = [logging.StreamHandler()]  # logging.FileHandler(filename=filename, encoding="utf-8", mode="w"),
    logging.basicConfig(
        format="[%(asctime)s] [%(name)s] [%(levelname)s] [%(funcName)s] %(message)s",
        datefmt="%d/%m/%Y %H:%M:%S",
        level=logger_level,
        handlers=handlers,
    )
