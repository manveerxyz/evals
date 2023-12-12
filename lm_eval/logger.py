import logging


class VerboseLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.verbose = False  # Default value for verbose

    def set_verbose(self, verbose):
        self.verbose = verbose

    def debug(self, msg, *args, **kwargs):
        if self.verbose:
            super().debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.verbose:
            super().info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.verbose:
            super().warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.verbose:
            super().error(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self.verbose:
            super().critical(msg, *args, **kwargs)


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "\033[94m",  # Blue
        "INFO": "\033[90m",  # Grey
        "WARNING": "\033[93m",  # Yellow
        "ERROR": "\033[91m",  # Red
        "CRITICAL": "\033[41m\033[97m",  # Red background, white text
    }

    RESET = "\033[0m"

    def format(self, record):
        log_level = record.levelname
        color = self.COLORS.get(log_level, "")
        formatted_message = super().format(record)
        if color:
            return f"{color}{formatted_message}{self.RESET}"
        return formatted_message


# Create the custom logger
eval_logger = VerboseLogger("scholar-eval")

# Create a handler and set the formatter
handler = logging.StreamHandler()
formatter = ColoredFormatter(
    fmt="%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
)
handler.setFormatter(formatter)

# Add the handler to the logger
eval_logger.addHandler(handler)
eval_logger.setLevel(logging.DEBUG)
