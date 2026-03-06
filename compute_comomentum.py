# compute_comomentum.py
# =====================================================================
# Comomentum Measure — Lou & Polk (2021)
# =====================================================================
#
# PAPER REFERENCE:
#   Lou, D. & Polk, C. (2021). "Comomentum: Inferring Arbitrage
#   Activity from Return Correlations."  Review of Financial Studies,
#   35(7), 3272–3302.
#
# WHAT IS COMOMENTUM?
#   Comomentum measures the excess return correlation (over and above
#   common risk factors) among stocks that belong to the extreme
#   momentum portfolios (winners and losers).  High comomentum
#   implies crowded arbitrage activity; low comomentum implies less
#   crowding.
#
# PROCEDURE (at each week t):
#   1. SORT into deciles: sort all live stocks with a valid momentum
#      score into deciles (10 equal-sized groups) based on their
#      standardised momentum.  The EXTREME LOSER decile is the bottom
#      10 %; the EXTREME WINNER decile is the top 10 %.
#
#   2. FF3 RESIDUALS:  For every stock in the loser or winner decile,
#      regress its last 52 weeks of returns on the Fama-French three
#      factors (Mkt-RF, SMB, HML) using OLS.  Collect the residuals
#      (= abnormal returns not explained by common risk factors).
#      Following Lewellen & Nagel (2006), betas are allowed to vary
#      over time because we use a rolling 52-week estimation window.
#      A stock must have all 52 weekly returns available in the
#      regression window to be included (no partial windows).
#
#   3. LOSER COMOMENTUM (CoMOM_L):
#        CoMOM_L = avg over all unique pairs (i,j) of Corr(resid_i, resid_j)
#      where resid_i and resid_j are FF3 residuals of two distinct
#      loser-decile stocks.  There are K*(K-1)/2 unique pairs.
#      This is the "average pairwise abnormal correlation" described
#      in Lou & Polk (2021).
#
#   4. WINNER COMOMENTUM (CoMOM_W):
#        CoMOM_W = avg over all unique pairs (i,j) of Corr(resid_i, resid_j)
#      Same pairwise procedure applied to the winner decile.
#
#   5. COMOMENTUM:
#        CoMOM = 0.5 * (CoMOM_W + CoMOM_L)
#      Simple average of winner and loser comomentum.
#
# TIMING:
#   Momentum scores are first available at week index 51 (19921225).
#   Comomentum requires a 52-week return window for the FF3 regression,
#   so its first computable date is max(51, 51) = week 51.  In practice,
#   because the first few momentum weeks have thin cross-sections, the
#   decile sort may be unreliable until slightly later.  No minimum
#   stock count is enforced — whatever lands in each decile is used.
#
# INPUT:
#   returns_clean : TxN np.ndarray  — weekly returns (not-listed = NaN)
#   momentum_std  : TxN np.ndarray  — standardised momentum (z-scored)
#   live          : TxN np.ndarray  — live/dead indicator (1/0)
#   ff_factors    : Tx3 np.ndarray  — [Mkt-RF, SMB, HML] weekly
#   dates         : DatetimeIndex   — T weekly dates
#
# OUTPUT:
#   comomentum    : T-length np.ndarray — comomentum time series
#   comom_winner  : T-length np.ndarray — winner-decile comomentum
#   comom_loser   : T-length np.ndarray — loser-decile  comomentum
#
# Standalone:  python compute_comomentum.py
# =====================================================================

import numpy as np
import pandas as pd
from logger_setup import _setup_logger
from config import CORR_WINDOW, MIN_RESID_OBS, MIN_STOCKS, DECILE_PCT_LO, DECILE_PCT_HI

log = _setup_logger()


# =====================================================================
# Helper: compute pairwise comomentum for one decile
# =====================================================================
def _decile_comomentum(residuals):
    """
    Given a (W x K) matrix of FF3 residuals for K stocks in one decile
    over W weeks, compute the average pairwise abnormal correlation
    as described in Lou & Polk (2021).

    For every unique pair (i, j) where i < j:
        rho_{i,j} = Corr(resid_i, resid_j)
    CoMOM_decile = mean of all K*(K-1)/2 pairwise correlations.

    A stock must have at least CORR_WINDOW (52) valid residual weeks
    to be eligible.  Only pairs where BOTH stocks are eligible
    contribute a correlation.

    Returns:
        (comom_value, K_total, K_contributed)
        comom_value    : float or NaN — decile comomentum
        K_total        : int — total stocks in the decile
        K_contributed  : int — stocks that were eligible (had enough
                         valid residual observations)
    """
    W, K = residuals.shape

    # If MIN_STOCKS is set (not None), enforce it; otherwise accept any K ≥ 2
    min_k = MIN_STOCKS if MIN_STOCKS is not None else 2
    if K < min_k:
        return np.nan, K, 0

    # Threshold for valid weeks: use MIN_RESID_OBS if set, else CORR_WINDOW
    resid_threshold = MIN_RESID_OBS if MIN_RESID_OBS is not None else CORR_WINDOW

    # ── Identify eligible stocks (enough valid residual weeks) ───────
    eligible = np.zeros(K, dtype=bool)
    for i in range(K):
        n_valid_i = np.sum(np.isfinite(residuals[:, i]))
        eligible[i] = (n_valid_i >= resid_threshold)

    eligible_idx = np.where(eligible)[0]
    K_eligible = len(eligible_idx)

    min_contrib = MIN_STOCKS if MIN_STOCKS is not None else 2
    if K_eligible < min_contrib:
        return np.nan, K, K_eligible

    # ── Compute full correlation matrix via np.corrcoef ──────────────
    # Extract only the eligible columns
    resid_eligible = residuals[:, eligible_idx]          # (W, K_eligible)

    # np.corrcoef expects each ROW to be a variable → transpose
    corr_matrix = np.corrcoef(resid_eligible, rowvar=False)  # (K_eligible, K_eligible)

    # ── Extract upper triangle (all unique pairs i < j) ──────────────
    upper_idx = np.triu_indices(K_eligible, k=1)
    pairwise_corrs = corr_matrix[upper_idx]

    # Remove any NaN pairs (can occur if a stock has zero variance)
    valid_corrs = pairwise_corrs[np.isfinite(pairwise_corrs)]

    if len(valid_corrs) == 0:
        return np.nan, K, K_eligible

    return np.mean(valid_corrs), K, K_eligible


# =====================================================================
# Helper: compute FF3 residuals for a subset of stocks
# =====================================================================
def _compute_ff3_residuals(returns_window, ff_window):
    """
    Regresses each stock's returns on FF3 factors and returns the
    residual matrix.

    A stock must have at least CORR_WINDOW (52) valid return weeks
    to be included (or MIN_RESID_OBS if set to a non-None value).
    Stocks with fewer valid weeks get all-NaN residuals.

    INPUT:
        returns_window : (W, K) — weekly returns for K stocks
        ff_window      : (W, 3) — [Mkt-RF, SMB, HML]

    OUTPUT:
        residuals      : (W, K) — OLS residuals (NaN where return was NaN)
    """
    W, K = returns_window.shape
    residuals = np.full((W, K), np.nan)

    # Threshold: use MIN_RESID_OBS if set, else CORR_WINDOW (52)
    resid_threshold = MIN_RESID_OBS if MIN_RESID_OBS is not None else CORR_WINDOW

    for j in range(K):
        y_j = returns_window[:, j]
        valid_j = np.isfinite(y_j)

        if np.sum(valid_j) < resid_threshold:
            continue

        # Design matrix: intercept + 3 FF factors
        X_j = np.column_stack([
            np.ones(np.sum(valid_j)),
            ff_window[valid_j, :]
        ])
        Y_j = y_j[valid_j]

        # OLS:  Y = X @ beta  →  residual = Y - X @ beta_hat
        coefs, _, _, _ = np.linalg.lstsq(X_j, Y_j, rcond=None)
        residuals[valid_j, j] = Y_j - X_j @ coefs

    return residuals


# =====================================================================
# Main function
# =====================================================================
def compute_comomentum(returns_clean, momentum_std, live, ff_factors, dates):
    """
    Computes the comomentum time series following Lou & Polk (2021).

    See module docstring for full methodology.

    RETURNS:
        comomentum   : T-length array — average of winner & loser comom
        comom_winner : T-length array — winner decile comomentum
        comom_loser  : T-length array — loser  decile comomentum
    """

    T, N = returns_clean.shape

    # ── Determine the first computable week ──────────────────────────
    # Need: (a) valid momentum scores (first at index 51)
    #        (b) 52 weeks of return history for FF3 regressions
    first_mom = 51   # first week with momentum scores
    comom_start = max(first_mom, CORR_WINDOW - 1)

    def _d(idx):
        return pd.Timestamp(dates[idx]).strftime('%Y-%m-%d')

    log.info("=" * 60)
    log.info("STEP 4: Computing comomentum (Lou & Polk, 2021)")
    log.info("=" * 60)
    log.info(f"  Method        : pairwise abnormal correlation (Lou & Polk 2021)")
    log.info(f"  Deciles       : bottom {DECILE_PCT_LO}% (losers) + "
             f"top {100 - DECILE_PCT_HI}% (winners)")
    log.info(f"  Corr window   : {CORR_WINDOW} weeks (rolling FF3 regressions)")
    log.info(f"  Min resid obs : {MIN_RESID_OBS if MIN_RESID_OBS is not None else f'None → using CORR_WINDOW={CORR_WINDOW}'}")
    log.info(f"  Min stocks    : {MIN_STOCKS if MIN_STOCKS is not None else 'None → using whatever lands in each decile'}")
    log.info(f"  First week    : index {comom_start} ({_d(comom_start)})")
    log.info(f"  Last week     : index {T-1} ({_d(T-1)})")

    # ── Pre-allocate ─────────────────────────────────────────────────
    comom_winner = np.full(T, np.nan)
    comom_loser  = np.full(T, np.nan)
    comomentum   = np.full(T, np.nan)

    n_computed        = 0
    n_skip_thin_xs    = 0
    n_skip_no_winners = 0
    n_skip_no_losers  = 0

    # ── Main loop ────────────────────────────────────────────────────
    for t in range(comom_start, T):

        # ── 1. Identify stocks with valid momentum + live ────────────
        mom_t = momentum_std[t, :]
        lv_t  = live[t, :]
        valid_mask = np.isfinite(mom_t) & (lv_t == 1)
        n_valid = np.sum(valid_mask)

        # Need at least 20 stocks to form a meaningful decile sort
        # (10 % of 20 = 2 stocks per decile — the minimum for a
        # leave-one-out correlation to be computable)
        if n_valid < 20:
            n_skip_thin_xs += 1
            log.info(f"    Week {t+1:>5} ({_d(t)}): SKIP — only {n_valid} "
                     f"valid stocks (need ≥ 20 for ≥ 2 per decile)")
            continue

        # ── 2. Decile breakpoints ────────────────────────────────────
        mom_valid = mom_t[valid_mask]
        q_lo = np.percentile(mom_valid, DECILE_PCT_LO)
        q_hi = np.percentile(mom_valid, DECILE_PCT_HI)

        # Column indices of extreme loser / winner decile stocks
        loser_mask  = valid_mask & (mom_t <= q_lo)
        winner_mask = valid_mask & (mom_t >= q_hi)

        loser_idx   = np.where(loser_mask)[0]
        winner_idx  = np.where(winner_mask)[0]

        # ── 3. Return & FF windows for regression ────────────────────
        w_start = t - CORR_WINDOW + 1
        w_end   = t + 1  # exclusive
        ff_window = ff_factors[w_start:w_end, :]   # (52, 3)

        # ── 4. Loser decile comomentum ───────────────────────────────
        n_loser_contrib = 0
        if len(loser_idx) >= 2:
            ret_losers = returns_clean[w_start:w_end, :][:, loser_idx]
            resid_losers = _compute_ff3_residuals(ret_losers, ff_window)
            comom_loser[t], _, n_loser_contrib = _decile_comomentum(resid_losers)
        else:
            n_skip_no_losers += 1

        # ── 5. Winner decile comomentum ──────────────────────────────
        n_winner_contrib = 0
        if len(winner_idx) >= 2:
            ret_winners = returns_clean[w_start:w_end, :][:, winner_idx]
            resid_winners = _compute_ff3_residuals(ret_winners, ff_window)
            comom_winner[t], _, n_winner_contrib = _decile_comomentum(resid_winners)
        else:
            n_skip_no_winners += 1

        # ── 6. Aggregate: CoMOM = 0.5 * (CoMOM_W + CoMOM_L) ────────
        cw = comom_winner[t]
        cl = comom_loser[t]
        if np.isfinite(cw) and np.isfinite(cl):
            comomentum[t] = 0.5 * (cw + cl)
            n_computed += 1
        elif np.isfinite(cw):
            comomentum[t] = cw            # fallback: only winner
            n_computed += 1
        elif np.isfinite(cl):
            comomentum[t] = cl            # fallback: only loser
            n_computed += 1

        # ── Report stock counts every week ───────────────────────────
        cl_str = f"{comom_loser[t]:.4f}" if np.isfinite(comom_loser[t]) else "NaN"
        cw_str = f"{comom_winner[t]:.4f}" if np.isfinite(comom_winner[t]) else "NaN"
        cm_str = f"{comomentum[t]:.4f}" if np.isfinite(comomentum[t]) else "NaN"
        log.info(f"    Week {t+1:>5} ({_d(t)}): "
                 f"N_valid={n_valid:>5}, "
                 f"losers={len(loser_idx):>4} (used={n_loser_contrib:>4}), "
                 f"winners={len(winner_idx):>4} (used={n_winner_contrib:>4}), "
                 f"CoMOM_L={cl_str}, CoMOM_W={cw_str}, CoMOM={cm_str}")

    # ── Summary ──────────────────────────────────────────────────────
    n_valid_comom = int(np.sum(np.isfinite(comomentum)))
    n_valid_w     = int(np.sum(np.isfinite(comom_winner)))
    n_valid_l     = int(np.sum(np.isfinite(comom_loser)))
    total_weeks   = T - comom_start

    log.info("-" * 60)
    log.info("COMOMENTUM SUMMARY")
    log.info("-" * 60)
    log.info(f"  Total weeks in loop       : {total_weeks}")
    log.info(f"  Successfully computed      : {n_computed}")
    log.info(f"  Skipped (thin cross-section): {n_skip_thin_xs}")
    log.info(f"  Skipped (few winners)       : {n_skip_no_winners}")
    log.info(f"  Skipped (few losers)        : {n_skip_no_losers}")
    log.info(f"  Valid CoMOM values          : {n_valid_comom} / {T}")
    log.info(f"  Valid CoMOM_W values        : {n_valid_w} / {T}")
    log.info(f"  Valid CoMOM_L values        : {n_valid_l} / {T}")

    if n_valid_comom > 0:
        vals = comomentum[np.isfinite(comomentum)]
        log.info(f"  CoMOM  mean={np.mean(vals):.6f}, "
                 f"std={np.std(vals):.6f}, "
                 f"min={np.min(vals):.6f}, max={np.max(vals):.6f}")
    if n_valid_w > 0:
        wv = comom_winner[np.isfinite(comom_winner)]
        log.info(f"  CoMOM_W mean={np.mean(wv):.6f}, "
                 f"std={np.std(wv):.6f}")
    if n_valid_l > 0:
        lv = comom_loser[np.isfinite(comom_loser)]
        log.info(f"  CoMOM_L mean={np.mean(lv):.6f}, "
                 f"std={np.std(lv):.6f}")

    log.info("-" * 60)
    log.info("STEP 4 COMPLETE — comomentum computed.")
    log.info("-" * 60)

    return comomentum, comom_winner, comom_loser


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    from data_loader import load_all_data
    from compute_momentum_signal import compute_momentum_signal

    data = load_all_data('input_data/')
    momentum, momentum_std = compute_momentum_signal(
        data['returns_clean'], data['dates']
    )
    comomentum, comom_w, comom_l = compute_comomentum(
        data['returns_clean'], momentum_std,
        data['live'], data['ff_factors'], data['dates']
    )
    n_valid = np.sum(np.isfinite(comomentum))
    print(f"\nComomentum: {n_valid} valid values out of {len(comomentum)} weeks.")
    print(f"Mean = {np.nanmean(comomentum):.6f}")
