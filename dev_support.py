import builtins
import logging
from functools import wraps
import pandas as pd

# Logger configuration: define names and default levels
_LOGGER_CONFIG = {
    "fn_logger": logging.DEBUG,
    "loader_logger": logging.INFO,
    "metrics_logger": logging.DEBUG,
    "visualizer_logger": logging.INFO,
    "stress_test_logger": logging.INFO,
    "top_logger": logging.DEBUG,
}


# Custom handler for dev_support.py
class SimpleFormatter(logging.Formatter):
    def format(self, record):
        # Only output the log message, ignoring other metadata
        return f"{record.msg}"


# Add a specific handler for dev_support.py
def get_dev_support_logger(logger_name):
    """
    Returns a logger with a simple format specifically for dev_support.py.
    
    Args:
        logger_name (str): The name of the logger.
    """
    logger = logging.getLogger(logger_name)
    # Avoid duplicate handlers
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        handler = logging.StreamHandler()
        handler.setFormatter(SimpleFormatter())
        logger.addHandler(handler)
        logger.propagate = False  # Prevent duplicate logs from propagating to the root logger
    return logger


def configure_logging(level=logging.INFO):
    """
    Configure logging for the application, including suppressing noisy third-party loggers
    and setting default levels for custom loggers.

    Parameters:
        level (int): Default logging level for the root logger.
    """
    # Set up root logger with a minimal format
    logging.basicConfig(
        level=level,
        format="%(message)s"  # Simple format for the root logger
    )

    # Suppress noisy third-party loggers
    noisy_loggers = ["matplotlib", "urllib3", "yfinance", "matplotlib.font_manager"]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.WARNING)

    # Apply custom configurations to each logger in _LOGGER_CONFIG
    for logger_name, logger_level in _LOGGER_CONFIG.items():
        logger = logging.getLogger(logger_name)
        logger.setLevel(logger_level)

        # Clear existing handlers to avoid duplicates
        if logger.hasHandlers():
            logger.handlers.clear()

        # Add a specific formatter for `metrics_logger` and `top_logger`
        if logger_name in ["metrics_logger", "top_logger"]:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(levelname).1s:%(filename)s:%(funcName)s:%(lineno)d:%(message)s"
            ))
            logger.addHandler(handler)
        else:
            # Use a default handler for other loggers
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(handler)

        # Expose loggers globally
        globals()[logger_name] = logger
        setattr(builtins, logger_name, logger)


def set_aspect_logging(aspect, level):
    """
    Dynamically adjust logging level for a specific aspect.
    
    Args:
        aspect (str): Aspect name (e.g., 'loader_logger', 'metrics_logger').
        level (int): Logging level (e.g., logging.DEBUG, logging.ERROR).
    """
    if aspect in _LOGGER_CONFIG:
        logger = globals().get(aspect)  # Get the logger from the global namespace
        if logger:
            logger.setLevel(level)
        else:
            print(f"Logger for aspect '{aspect}' is not initialized.")
    else:
        print(f"Unknown aspect: '{aspect}'")


def log_function_entry(logger_name):
    """
    Decorator to log function entry with arguments.
    Args:
        logger_name (str): Name of the logger to use for logging.
    """
    logger = get_dev_support_logger(logger_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            arg_list = ", ".join(
                [repr(a) for a in args] +
                [f"{k}={v!r}" for k, v in kwargs.items()]
            )
            logger.debug(f"\n***** Entering {func.__name__}({arg_list})")
            return func(*args, **kwargs)
        return wrapper
    return decorator


def log_function_details(logger_name="default"):
    """
    A decorator to log argument and return details for functions.

    Parameters:
        logger_name (str): The name of the logger to use for logging.
    """
    logger = get_dev_support_logger(logger_name)

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Log function entry
            logger.debug(f"\n***** Entering {func.__name__}")
            # Log argument details
            for idx, arg in enumerate(args):
                if isinstance(arg, pd.DataFrame):
                    logger.debug(f"Arg {idx} (DataFrame): shape={arg.shape}")  #, head=\n{arg.head()}")
                elif isinstance(arg, pd.Series):
                    logger.debug(f"Arg {idx} (Series): shape={arg.shape}")  #, head=\n{arg.head()}")
                elif isinstance(arg, (str, int, float)):
                    logger.debug(f"Arg {idx} ({type(arg).__name__}): value={arg}")
                else:
                    logger.debug(f"Arg {idx} (type={type(arg).__name__}): value={arg}")

            for key, value in kwargs.items():
                if isinstance(value, pd.DataFrame):
                    logger.debug(f"Kwarg {key} (DataFrame): shape={value.shape}, head=\n{value.head()}")
                elif isinstance(value, pd.Series):
                    logger.debug(f"Kwarg {key} (Series): shape={value.shape}, head=\n{value.head()}")
                elif isinstance(value, (str, int, float)):
                    logger.debug(f"Kwarg {key} ({type(value).__name__}): value={value}")
                else:
                    logger.debug(f"Kwarg {key} (type={type(value).__name__}): value={value}")

            # Execute the function
            result = func(*args, **kwargs)

            # Log return value details
            if isinstance(result, pd.DataFrame):
                logger.debug(f"Return (DataFrame): shape={result.shape}, head=\n{result.head()}")
            elif isinstance(result, pd.Series):
                logger.debug(f"Return (Series): shape={result.shape}, head=\n{result.head()}")
            elif isinstance(result, (str, int, float)):
                logger.debug(f"Return ({type(result).__name__}): value={result}")
            else:
                logger.debug(f"Return (type={type(result).__name__}): value={result}")

            # Log function exit
            logger.debug(f"Exiting {func.__name__}")
            return result

        return wrapper
    return decorator

# Add log_function_details to builtins for global access
setattr(builtins, "log_function_details", log_function_details)

class ContextualLogger(logging.LoggerAdapter):
    """
    A logger adapter for adding contextual information to logs.
    """
    def process(self, msg, kwargs):
        context = " ".join([f"{key}={value}" for key, value in self.extra.items()])
        return f"{context} {msg}", kwargs


def get_contextual_logger(logger_name, **context):
    """
    Create a contextual logger with additional context.
    Args:
        logger_name (str): Name of the logger to adapt.
        context (dict): Contextual information to include in logs.
    """
    logger = loggers.get(logger_name, logging.getLogger(logger_name))
    return ContextualLogger(logger, context)


def __getattr__(name):
    """
    Dynamically retrieve loggers as attributes of this module.

    Args:
        name (str): The name of the logger.

    Returns:
        logging.Logger: The requested logger.

    Raises:
        AttributeError: If the requested attribute is not found.
    """
    if name in _LOGGER_CONFIG:
        # Ensure loggers are configured and return the requested logger
        return globals().get(name)
    raise AttributeError(f"Module 'dev_support' has no attribute '{name}'")


def set_aspect_logging(aspect, level):
    """
    Dynamically adjust logging level for a specific aspect.
    
    Args:
        aspect (str): Aspect name (e.g., 'loader_logger', 'metrics_logger').
        level (int): Logging level (e.g., logging.DEBUG, logging.ERROR).
    """
    if aspect in _LOGGER_CONFIG:
        logger = globals().get(aspect)  # Get the logger from the global namespace
        if logger:
            logger.setLevel(level)
        else:
            print(f"Logger for aspect '{aspect}' is not initialized.")
    else:
        print(f"Unknown aspect: '{aspect}'")


def type_shape_length(variable, variable_name, logger):
    logger.debug(f"Type of {variable_name}: {type(variable)}")
    logger.debug(f"isinstance(variable, pd.DataFrame): {isinstance(variable, pd.DataFrame)}")
    if isinstance(variable, pd.DataFrame):
        logger.debug(f"Shape of DataFrame {variable_name}: {variable.shape}")
    elif isinstance(variable, pd.Series):
        logger.debug(f"Length of Series {variable_name}: {len(variable)}")


def head_description(variable, variable_name, logger):
    logger.debug(f"First few rows of {variable_name}:\n{variable.head()}")
    logger.debug(f"Summary statistics of {variable_name}:\n{variable.describe()}")


# Automatically expose all dynamically created loggers as part of the module
__all__ = (
    list(_LOGGER_CONFIG.keys())
    + ["configure_logging", "set_aspect_logging", "log_function_details", "type_shape_length", "head_description"]
)
