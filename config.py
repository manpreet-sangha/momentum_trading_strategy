# config.py
# =====================================================================
# Central Configuration — All Tuneable Parameters
# =====================================================================
#
# This file is the SINGLE SOURCE OF TRUTH for every numerical
# parameter used across the project.  Every other module imports
# from here rather than defining its own constants.
#
# To change a parameter, edit it HERE — the change propagates
# automatically to all modules that use it.
#
# =====================================================================

# ─────────────────────────────────────────────────────────────────────
# MOMENTUM SIGNAL  (compute_momentum_signal.py)
# ─────────────────────────────────────────────────────────────────────
LOOKBACK = 48   # weeks of compounded returns (≈ 11 months)
SKIP     = 4    # most recent weeks to exclude (short-term reversal)
TOTAL    = LOOKBACK + SKIP   # = 52 weeks of history before first score

# ─────────────────────────────────────────────────────────────────────
# COMOMENTUM  (compute_comomentum.py)
# ─────────────────────────────────────────────────────────────────────
CORR_WINDOW   = 52     # weeks used for rolling FF3 residual estimation
                        # Also the minimum valid return weeks per stock
                        # (a stock must have all 52 weeks in the window).

MIN_RESID_OBS = None   # RESERVED — currently ignored.
                        # CORR_WINDOW is used as the hard threshold.
                        # Set to an integer (e.g. 26) to override.

MIN_STOCKS    = None   # RESERVED — currently ignored.
                        # Whatever number of stocks land in a decile is used.
                        # Set to an integer (e.g. 5) to impose a minimum.

DECILE_PCT_LO = 10     # bottom 10 % = extreme loser decile
DECILE_PCT_HI = 90     # top 90 % threshold → extreme winner decile

# ─────────────────────────────────────────────────────────────────────
# ADJUSTED MOMENTUM  (compute_adjusted_momentum.py)
# ─────────────────────────────────────────────────────────────────────
MIN_PAST_VALUES = 10   # minimum past comomentum values needed to
                        # compute a percentile rank (expanding window)

# ─────────────────────────────────────────────────────────────────────
# PERFORMANCE  (performance.py)
# ─────────────────────────────────────────────────────────────────────
WEEKS_PER_YEAR = 52    # used to annualise weekly statistics
