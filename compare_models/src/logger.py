import inspect
import logging
import os
from datetime import datetime


class Logger:
    _instance = None  # Singleton instance

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        log_filename = datetime.now().strftime("logs_%Y-%m-%d.log")

        if not os.path.exists("logs"):
            os.makedirs("logs")

        log_file_path = os.path.join("logs", log_filename)

        self.logger = logging.getLogger("Logger")

        if not self.logger.hasHandlers():
            self.logger.setLevel(logging.DEBUG)

            log_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

            console_handler = logging.StreamHandler()
            console_handler.setLevel(logging.DEBUG)
            console_handler.setFormatter(log_format)

            file_handler = logging.FileHandler(log_file_path)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(log_format)

            self.logger.addHandler(console_handler)
            self.logger.addHandler(file_handler)

    def log(self, level, message):
        frame = inspect.currentframe().f_back.f_back
        filename = os.path.basename(frame.f_code.co_filename)
        lineno = frame.f_lineno
        function_name = frame.f_code.co_name

        extra_info = f"{filename}:{lineno} ({function_name})"
        log_message = f"[{extra_info}] - {message}"

        if level.lower() == "debug":
            self.logger.debug(log_message)
        elif level.lower() == "info":
            self.logger.info(log_message)
        elif level.lower() == "warning":
            self.logger.warning(log_message)
        elif level.lower() == "error":
            self.logger.error(log_message)
        elif level.lower() == "critical":
            self.logger.critical(log_message)
        else:
            self.logger.info(log_message)

    def debug(self, message):
        self.log("debug", message)

    def info(self, message):
        self.log("info", message)

    def warning(self, message):
        self.log("warning", message)

    def error(self, message):
        self.log("error", message)

    def critical(self, message):
        self.log("critical", message)
