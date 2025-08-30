import os

import yaml
from box import Box


class ConfigManager:
    """
    Class to manage configuration settings from a YAML file.
    """
    def __init__(self, config_file: str):
        """
        Initialize the ConfigManager with a configuration file.

        Args:
            config_file (str): Path to the YAML configuration file.
        """
        self.config = self.read_config(config_file)
        self._w = 1

    def read_config(self, config_file: str) -> Box:
        """
        Read and parse the YAML configuration file.

        Args:
            config_file (str): Path to the YAML configuration file.
        Returns:
            Box: Parsed configuration settings.
        Raises:
            FileNotFoundError: If the configuration file does not exist.
        """
        if not os.path.exists(config_file):
            raise FileNotFoundError("No config file provided")
        with open(config_file, "r") as file:
            return Box(yaml.safe_load(file))
