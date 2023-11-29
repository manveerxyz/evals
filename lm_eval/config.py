import os
import configparser
from dataclasses import dataclass

APP_NAME = "scholar-evals"


@dataclass
class Config:
    api_key: str


def get_config_path():
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, f".{APP_NAME}")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return os.path.join(config_dir, "config.ini")


def get_config() -> Config:
    config_path = get_config_path()
    config = configparser.ConfigParser()

    if os.path.exists(config_path):
        config.read(config_path)
        key = config.get("DEFAULT", "api_key", fallback=None)
        if key:
            return Config(api_key=key)

    # key not found in config, prompt the user
    api_key = input(
        "\nAPI Key not found in config.\nEnter your Scholar API Key (or press enter to skip): "
    ).strip()
    if not api_key or api_key == "":
        return None

    set_api_key(api_key)

    c = Config(api_key=api_key)
    return c


def set_api_key(api_key):
    config_path = get_config_path()
    config = configparser.ConfigParser()
    config["DEFAULT"] = {"api_key": api_key}
    with open(config_path, "w") as configfile:
        config.write(configfile)
