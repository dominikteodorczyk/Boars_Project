import os

import yaml
from box import Box


class ConfigManager:
    def __init__(self, config_file: str):
        self.config = self.read_config(config_file)

    def read_config(self, config_file: str):
        if not os.path.exists(config_file):
            raise FileNotFoundError("No config file provided")
        with open(config_file, "r") as file:
            return Box(yaml.safe_load(file))
