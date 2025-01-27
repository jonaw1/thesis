import logging
from logging.handlers import RotatingFileHandler


def setup_logger():
    """
    Configure and return a logger.
    """
    # Create a logger
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)  # Set the base logging level

    # Create a formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # Create a file handler for rotating logs
    file_handler = RotatingFileHandler(
        "app.log",
        maxBytes=500 * 1024,
        backupCount=3,  # Rotate after 500KB, keep 3 backups
    )
    file_handler.setFormatter(formatter)

    # Create a console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Add handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger


logger = setup_logger()
