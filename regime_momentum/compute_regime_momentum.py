# compute_regime_momentum.py
# =====================================================================
# Regime-Conditional Momentum — Comomentum-Based Timing
# =====================================================================
#
# PAPER MOTIVATION:
#   Lou, D. & Polk, C. (2021). "Comomentum: Inferring Arbitrage
#   Activity from Return Correlations."  Review of Financial Studies.
#
# RATIONALE:
#   High comomentum signals crowded momentum trading, which predicts
#   subsequent momentum reversals and crashes.  Rather than smoothly
#   scaling the factor return (as in Step 5), this approach applies
#   a hard regime switch: when comomentum indicates heavy crowding,
#   we EXIT momentum entirely; otherwise we keep full exposure.
#
#   The logic is a direct implication of the paper's finding: if
#   high-comomentum periods earn lower (or negative) future momentum
#   returns, the simplest exploit is to avoid those periods altogether.
#
# PROCEDURE:
#   Step A — Expanding-window tercile rank of comomentum
#       At each week t, compute the percentile rank of comomentum[t]
#       relative to all values from the start up to t-1 (no lookahead).
#       Classify:
#         • rank > REGIME_THRESHOLD  →  "crowded" (top tercile)
#         • rank ≤ REGIME_THRESHOLD  →  "uncrowded"
#
#   Step B — Modify momentum exposures
#       For weeks classified as "crowded" (using the LAGGED regime,
#       i.e. t-1's classification applied at t):
#         - Set all momentum z-scores to zero → no positions
#       For "uncrowded" weeks:
#         - Keep standard momentum z-scores unchanged
#
#   Step C — Re-run Fama-MacBeth on the modified exposures
#       The modified exposure matrix feeds into famaMacBeth().
#       "Crowded" weeks will have all-zero exposures → gamma ≈ 0
#       or NaN (no variation to estimate).
#       Performance is measured across all weeks.
#
# IMPORTANT DIFFERENCE FROM STEP 5:
#   Step 5 scales the FACTOR RETURN series post-hoc.
#   This step modifies the EXPOSURE MATRIX and re-runs Fama-MacBeth,
#   as required by coursework point (6).
#
# INPUT:
#   momentum_std : TxN np.ndarray — standardised momentum z-scores
#   comomentum   : T-length array — comomentum time series
#   returns_clean: TxN np.ndarray — weekly stock returns
#   live         : TxN np.ndarray — live indicator
#   dates        : T-length DatetimeIndex
#
# OUTPUT:
#   gamma_regime : T-length np.ndarray — factor returns from FM regression
#   tstat_regime : float — t-statistic on mean gamma
#   regime       : T-length np.ndarray — regime indicator (1=active, 0=exit)
#
# Standalone:  python -m regime_momentum.compute_regime_momentum
# =====================================================================

import sys, os
if __name__ == '__main__':
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from logger_setup import _setup_logger
from config import MIN_PAST_VALUES, REGIME_THRESHOLD

log = _setup_logger()


def compute_regime_momentum(momentum_std, comomentum, returns_clean,
                             live, dates, save_path=None):
    """
    Compute regime-conditional momentum factor returns by zeroing out
    momentum exposures during high-comomentum (crowded) weeks, then
    re-running Fama-MacBeth regressions.

    Parameters
    ----------
    momentum_std  : TxN np.ndarray — standardised momentum z-scores
    comomentum    : T-length array — comomentum time series
    returns_clean : TxN np.ndarray — weekly stock returns
    live          : TxN np.ndarray — live indicator (1=live)
    dates         : T-length DatetimeIndex
    save_path     : str or None — optional xlsx output path

    Returns
    -------
    gamma_regime : T-length np.ndarray — weekly factor returns
    tstat_regime : float — t-statistic
    regime       : T-length np.ndarray — 1 = active, 0 = exit
    """
    from fama_macbeth.fama_macbeth import famaMacBeth

    T, N = momentum_std.shape

    log.info("=" * 60)
    log.info("STEP 5b: Regime-Conditional Momentum")
    log.info("=" * 60)
    log.info(f"  Threshold: top {(1 - REGIME_THRESHOLD)*100:.0f}% "
             f"of comomentum → EXIT momentum")
    log.info(f"  Min past values for percentile rank: {MIN_PAST_VALUES}")

    # ── Step A: Expanding-window percentile rank ─────────────────────
    log.info("  Step A: Computing expanding-window percentile ranks ...")
    comom_pctile = np.full(T, np.nan)

    for t in range(1, T):
        past = comomentum[:t]
        valid_past = past[np.isfinite(past)]
        if len(valid_past) < MIN_PAST_VALUES or not np.isfinite(comomentum[t]):
            continue
        comom_pctile[t] = np.mean(valid_past <= comomentum[t])

    n_ranked = int(np.sum(np.isfinite(comom_pctile)))
    log.info(f"    Percentile ranks computed: {n_ranked}")

    # ── Step B: Classify regime using LAGGED percentile rank ─────────
    log.info("  Step B: Classifying regimes (lagged percentile rank) ...")
    regime = np.ones(T, dtype=float)  # default: active (1)

    n_exit = 0
    for t in range(1, T):
        if np.isfinite(comom_pctile[t - 1]):
            if comom_pctile[t - 1] > REGIME_THRESHOLD:
                regime[t] = 0.0  # crowded → exit
                n_exit += 1

    n_active = int(np.sum(regime == 1.0)) - 1  # minus week 0
    log.info(f"    Active weeks (uncrowded): {n_active}")
    log.info(f"    Exit weeks (crowded):     {n_exit}")

    # ── Step C: Modify exposures and re-run Fama-MacBeth ─────────────
    log.info("  Step C: Zeroing exposures in crowded weeks & "
             "re-running Fama-MacBeth ...")

    # Copy momentum z-scores; set crowded weeks to zero
    momentum_regime = momentum_std.copy()
    for t in range(T):
        if regime[t] == 0.0:
            momentum_regime[t, :] = 0.0

    gamma_regime, tstat_regime = famaMacBeth(
        momentum_regime, returns_clean, live,
        dates=dates, save_path=save_path
    )

    # ── Summary ──────────────────────────────────────────────────────
    n_finite = int(np.sum(np.isfinite(gamma_regime)))
    log.info("-" * 60)
    log.info("REGIME-CONDITIONAL MOMENTUM SUMMARY")
    log.info("-" * 60)
    log.info(f"  Regime threshold:  {REGIME_THRESHOLD}")
    log.info(f"  Active weeks:      {n_active}")
    log.info(f"  Exit weeks:        {n_exit}")
    log.info(f"  Valid gamma weeks: {n_finite}")
    log.info(f"  t-statistic:       {tstat_regime:.4f}")

    if n_finite > 0:
        vr = gamma_regime[np.isfinite(gamma_regime)]
        log.info(f"  Mean weekly γ:     {np.mean(vr)*100:.4f}%")
        log.info(f"  Std weekly γ:      {np.std(vr)*100:.4f}%")

    log.info("-" * 60)
    log.info("STEP 5b COMPLETE")
    log.info("-" * 60)

    return gamma_regime, tstat_regime, regime


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data.data_loader import load_all_data
    from compute_momentum.compute_momentum_signal import compute_momentum_signal
    from fama_macbeth.fama_macbeth import famaMacBeth
    from comomentum.compute_comomentum import compute_comomentum

    data = load_all_data('input_data/')
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )
    gamma_std, tstat_std = famaMacBeth(
        momentum_std, data['returns_clean'], data['live']
    )
    comomentum_arr, _, _ = compute_comomentum(
        data['returns_clean'], momentum_std,
        data['live'], data['ff_factors'], data['dates']
    )

    gamma_regime, tstat_regime, regime = compute_regime_momentum(
        momentum_std, comomentum_arr,
        data['returns_clean'], data['live'], data['dates']
    )

    print(f"\nStandard  t-stat: {tstat_std:.4f}")
    print(f"Regime    t-stat: {tstat_regime:.4f}")
