import os
import configparser
from dataclasses import dataclass
from print_color import print

APP_SHORT_NAME = "sevals"
APP_NAME = "scholar-eval"
APP_VERSION = "0.0.2"


@dataclass
class Config:
    api_key: str


def get_config_path():
    home_dir = os.path.expanduser("~")
    config_dir = os.path.join(home_dir, f".{APP_NAME}")
    if not os.path.exists(config_dir):
        os.makedirs(config_dir)
    return os.path.join(config_dir, "config.ini")


def get_config(
    require_api_key: bool = False,
) -> Config:
    config_path = get_config_path()
    config = configparser.ConfigParser()

    if os.path.exists(config_path):
        config.read(config_path)
        key = config.get("DEFAULT", "api_key", fallback=None)
        if key:
            return Config(api_key=key)

    # key not found in config, prompt the user
    print("Scholar API Key not found in config.", color="yellow")
    input_str = "Enter your Scholar API Key (or press enter to skip): "
    if require_api_key:
        input_str = "Enter your Scholar API Key: "
    api_key = input(input_str).strip()
    if not api_key or api_key == "":
        if require_api_key:
            print("\nAPI Key is required.", color="red")
            print(
                "You can get an API Key from: https://usescholar.org/api-keys",
                color="red",
            )
            exit(1)
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
