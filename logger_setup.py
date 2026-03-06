# logger_setup.py
# =====================================================================
# Centralised Logger Configuration
# =====================================================================
# Provides a reusable _setup_logger() function that creates a logger
# writing to both a log file and the console.  Extracted from
# data_loader.py so that any module in the project can import and
# use the same logging configuration.
# =====================================================================

import logging
import os


def _setup_logger(log_dir='output_data', log_filename='data_loading.log'):
    """
    Creates and returns a logger that writes to BOTH:
      1. A log file  (output_data/data_loading.log)  -- for the future app
      2. The console  (stdout)                        -- for interactive use

    Each run overwrites the log file so the front-end always reads
    the latest data-loading result.
    """
    logger = logging.getLogger('data_loader')

    # Avoid adding duplicate handlers if this module is re-imported
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    # Formatter shared by both handlers
    fmt = logging.Formatter(
        fmt='%(asctime)s | %(levelname)-7s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # --- File handler (overwrite mode) ---
    os.makedirs(log_dir, exist_ok=True)
    fh = logging.FileHandler(
        os.path.join(log_dir, log_filename), mode='w', encoding='utf-8'
    )
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # --- Console handler ---
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    return logger
