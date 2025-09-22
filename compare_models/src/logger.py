import inspect
import logging
import os
from datetime import datetime
from typing import Literal


class Logger:
    """
    Singleton Logger class to log messages to both console and file with detailed context.
    Uses Python's built-in logging module.
    Ensures only one instance of the logger exists throughout the application.
    """
    _instance = None  # Singleton instance

    def __new__(cls):
        """
        Create or return the singleton instance of the Logger class.
        Returns:
            Logger: Singleton instance of the Logger class.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self) -> None:
        """
        Initialize the logger by setting up console and file handlers with appropriate formatting.
        Creates a 'logs' directory if it doesn't exist and names log files based on the current date.

        Returns:
            None
        """
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

    def log(self, level: Literal["debug", "info", "warning", "error", "critical"], message: str) -> None:
        """
        Log a message with the specified level, including context about the caller's file, line number, and function name.

        Args:
            level (Literal["debug", "info", "warning", "error", "critical"]): The log level.
            message (str): The message to log.
        Returns:
            None
        """
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

    def debug(self, message: str) -> None:
        """
        Log a debug message.

        Args:
            message (str): The debug message to log.
        Returns:
            None
        """
        self.log("debug", message)

    def info(self, message: str) -> None:
        """
        Log an info message.

        Args:
            message (str): The info message to log.
        Returns:
            None
        """
        self.log("info", message)

    def warning(self, message: str) -> None:
        """
        Log a warning message.

        Args:
            message (str): The warning message to log.
        Returns:
            None
        """
        self.log("warning", message)

    def error(self, message: str) -> None:
        """
        Log an error message.

        Args:
            message (str): The error message to log.
        Returns:
            None
        """
        self.log("error", message)

    def critical(self, message: str) -> None:
        """
        Log a critical message.

        Args:
            message (str): The critical message to log.
        Returns:
            None
        """
        self.log("critical", message)
