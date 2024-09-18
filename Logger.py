import logging
import os

class Logger:
    def __init__(self, filename) -> None:
        self.logger = None
        self.filename = filename
    
    def initialize(self):
        if os.path.exists(self.filename):
            os.remove(self.filename)

        logging.basicConfig(
            filename=self.filename,
            level=logging.DEBUG,
            format='%(asctime)s %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )

        logger = logging.getLogger(__name__)
        self.logger = logger

    def log(self, message, level='info'):
        if level == 'debug':
            self.logger.debug(f"[DEBUG] {message}")
        elif level == 'info':
            self.logger.info(f"[INFO] {message}")
        elif level == 'warning':
            self.logger.warning(f"[WARNING] {message}")
        elif level == 'error':
            self.logger.error(f"[ERROR] {message}")
        elif level == 'critical':
            self.logger.critical(f"[CRITICAL] {message}")