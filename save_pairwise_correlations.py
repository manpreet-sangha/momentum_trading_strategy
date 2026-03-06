# save_pairwise_correlations.py
# =====================================================================
# Save Pairwise Correlation Matrices — Step 4 Diagnostic Output
# =====================================================================
#
# After computing FF3 residuals for the loser and winner deciles,
# the comomentum procedure computes pairwise abnormal correlations
# (Lou & Polk, 2021):
#     Corr(resid_i, resid_j)  for all unique pairs (i, j) in each decile
#
# This module saves the full K×K correlation matrices for both
# deciles to an Excel workbook so the computations can be inspected.
#
# WHAT IS SAVED:
#   An .xlsx file with five sheets:
#     1. 'Documentation'       — explains methodology + summary stats
#     2. 'Loser_CorrMatrix'    — K_L × K_L pairwise correlation matrix
#     3. 'Winner_CorrMatrix'   — K_W × K_W pairwise correlation matrix
#     4. 'Loser_Pairwise'      — long-format table of all unique pairs
#     5. 'Winner_Pairwise'     — long-format table of all unique pairs
#
#   Like the residual file, this is a SNAPSHOT for one chosen week.
#
# Standalone:  python save_pairwise_correlations.py
# =====================================================================

import numpy as np
import pandas as pd
import os

from config import (CORR_WINDOW, DECILE_PCT_LO, DECILE_PCT_HI,
                     MIN_RESID_OBS, MIN_STOCKS)


def _compute_ff3_residuals(returns_window, ff_window):
    """Regress each stock on FF3 factors; return (W, K) residual matrix."""
    W, K = returns_window.shape
    residuals = np.full((W, K), np.nan)
    resid_threshold = MIN_RESID_OBS if MIN_RESID_OBS is not None else CORR_WINDOW

    for j in range(K):
        y_j = returns_window[:, j]
        valid_j = np.isfinite(y_j)
        if np.sum(valid_j) < resid_threshold:
            continue
        X_j = np.column_stack([np.ones(np.sum(valid_j)),
                                ff_window[valid_j, :]])
        Y_j = y_j[valid_j]
        coefs, _, _, _ = np.linalg.lstsq(X_j, Y_j, rcond=None)
        residuals[valid_j, j] = Y_j - X_j @ coefs
    return residuals


def _build_corr_outputs(residuals, stock_names):
    """
    Given a (W, K) residual matrix and stock names, compute:
      - The K×K pairwise correlation matrix
      - A long-format DataFrame of all unique pairs with their correlations
      - Summary statistics

    Returns:
        (corr_df, pairs_df, avg_corr, n_pairs, K_eligible)
    """
    W, K = residuals.shape
    resid_threshold = MIN_RESID_OBS if MIN_RESID_OBS is not None else CORR_WINDOW

    # Identify eligible stocks
    eligible = np.zeros(K, dtype=bool)
    for i in range(K):
        eligible[i] = np.sum(np.isfinite(residuals[:, i])) >= resid_threshold
    eligible_idx = np.where(eligible)[0]
    K_eligible = len(eligible_idx)

    if K_eligible < 2:
        return None, None, np.nan, 0, K_eligible

    eligible_names = [stock_names[i] for i in eligible_idx]
    resid_eligible = residuals[:, eligible_idx]

    # Full correlation matrix
    corr_matrix = np.corrcoef(resid_eligible, rowvar=False)
    corr_df = pd.DataFrame(corr_matrix,
                            index=eligible_names,
                            columns=eligible_names)

    # Long-format: all unique pairs (upper triangle)
    pairs_rows = []
    upper_i, upper_j = np.triu_indices(K_eligible, k=1)
    for idx in range(len(upper_i)):
        ii, jj = upper_i[idx], upper_j[idx]
        rho = corr_matrix[ii, jj]
        pairs_rows.append({
            'stock_i': eligible_names[ii],
            'stock_j': eligible_names[jj],
            'correlation': rho
        })

    pairs_df = pd.DataFrame(pairs_rows)

    # Summary
    valid_corrs = corr_matrix[upper_i, upper_j]
    valid_corrs = valid_corrs[np.isfinite(valid_corrs)]
    n_pairs = len(valid_corrs)
    avg_corr = np.mean(valid_corrs) if n_pairs > 0 else np.nan

    return corr_df, pairs_df, avg_corr, n_pairs, K_eligible


def save_pairwise_correlations(returns_clean, momentum_std, live,
                                ff_factors, dates, names,
                                snapshot_week=None,
                                save_path='output_data/pairwise_correlations.xlsx'):
    """
    Compute and save pairwise correlation matrices for loser and
    winner deciles at one snapshot week.

    INPUTS:
        returns_clean : TxN np.ndarray
        momentum_std  : TxN np.ndarray
        live          : TxN np.ndarray
        ff_factors    : Tx3 np.ndarray
        dates         : length-T array-like
        names         : length-N array-like  (stock names / tickers)
        snapshot_week : int index (0-based). Defaults to last week.
        save_path     : output .xlsx path

    OUTPUTS:
        Writes an Excel file.  Returns a dict with keys:
            corr_losers, corr_winners, pairs_losers, pairs_winners,
            avg_corr_losers, avg_corr_winners, comomentum
    """
    T, N = returns_clean.shape

    if snapshot_week is None:
        snapshot_week = T - 1

    t = snapshot_week
    date_str = pd.Timestamp(dates[t]).strftime('%Y-%m-%d')

    # ── Decile sort ──────────────────────────────────────────────────
    mom_t = momentum_std[t, :]
    lv_t  = live[t, :]
    valid_mask = np.isfinite(mom_t) & (lv_t == 1)
    mom_valid = mom_t[valid_mask]
    q_lo = np.percentile(mom_valid, DECILE_PCT_LO)
    q_hi = np.percentile(mom_valid, DECILE_PCT_HI)

    loser_mask  = valid_mask & (mom_t <= q_lo)
    winner_mask = valid_mask & (mom_t >= q_hi)
    loser_idx   = np.where(loser_mask)[0]
    winner_idx  = np.where(winner_mask)[0]

    loser_names  = [str(names[i]) for i in loser_idx]
    winner_names = [str(names[i]) for i in winner_idx]

    # ── Rolling window ───────────────────────────────────────────────
    w_start = t - CORR_WINDOW + 1
    w_end   = t + 1
    window_start_str = pd.Timestamp(dates[w_start]).strftime('%Y-%m-%d')
    window_end_str   = date_str
    ff_window = ff_factors[w_start:w_end, :]

    # ── Compute residuals ────────────────────────────────────────────
    ret_losers  = returns_clean[w_start:w_end, :][:, loser_idx]
    ret_winners = returns_clean[w_start:w_end, :][:, winner_idx]
    resid_losers  = _compute_ff3_residuals(ret_losers, ff_window)
    resid_winners = _compute_ff3_residuals(ret_winners, ff_window)

    # ── Pairwise correlations ────────────────────────────────────────
    corr_l, pairs_l, avg_l, n_pairs_l, k_elig_l = _build_corr_outputs(
        resid_losers, loser_names)
    corr_w, pairs_w, avg_w, n_pairs_w, k_elig_w = _build_corr_outputs(
        resid_winners, winner_names)

    comom = 0.5 * (avg_w + avg_l) if (np.isfinite(avg_w) and
                                       np.isfinite(avg_l)) else np.nan

    # ── Documentation sheet ──────────────────────────────────────────
    doc_rows = [
        ["PAIRWISE ABNORMAL CORRELATION MATRICES — STEP 4 SNAPSHOT", ""],
        ["", ""],
        ["Snapshot week index", str(t)],
        ["Snapshot date", date_str],
        ["Rolling window",
         f"{window_start_str} to {window_end_str} ({CORR_WINDOW} weeks)"],
        ["", ""],
        ["METHODOLOGY (Lou & Polk, 2021):", ""],
        ["  1. Sort all live stocks with valid momentum into deciles.", ""],
        ["  2. For each loser/winner stock, regress its 52-week returns "
         "on FF3 factors (Mkt-RF, SMB, HML) → collect residuals.", ""],
        ["  3. Compute Corr(resid_i, resid_j) for every unique pair "
         "(i, j) within each decile.", ""],
        ["  4. CoMOM_decile = mean of all K*(K-1)/2 pairwise correlations.", ""],
        ["  5. CoMOM = 0.5 * (CoMOM_winners + CoMOM_losers).", ""],
        ["", ""],
        ["LOSER DECILE", ""],
        ["  Criterion", f"Bottom {DECILE_PCT_LO}% of momentum scores"],
        ["  N stocks in decile", str(len(loser_idx))],
        ["  N eligible (≥{} valid weeks)".format(CORR_WINDOW), str(k_elig_l)],
        ["  N unique pairs", str(n_pairs_l)],
        ["  Mean pairwise correlation (= CoMOM_L)", f"{avg_l:.6f}"
         if np.isfinite(avg_l) else "NaN"],
        ["", ""],
        ["WINNER DECILE", ""],
        ["  Criterion", f"Top {100 - DECILE_PCT_HI}% of momentum scores"],
        ["  N stocks in decile", str(len(winner_idx))],
        ["  N eligible (≥{} valid weeks)".format(CORR_WINDOW), str(k_elig_w)],
        ["  N unique pairs", str(n_pairs_w)],
        ["  Mean pairwise correlation (= CoMOM_W)", f"{avg_w:.6f}"
         if np.isfinite(avg_w) else "NaN"],
        ["", ""],
        ["COMOMENTUM", f"{comom:.6f}" if np.isfinite(comom) else "NaN"],
        ["  = 0.5 * (CoMOM_W + CoMOM_L)", ""],
        ["", ""],
        ["SHEET: Loser_CorrMatrix", ""],
        ["  Shape", f"{k_elig_l} x {k_elig_l}"],
        ["  Values", "Pearson correlation of FF3 residuals between each "
                     "pair of loser-decile stocks. Diagonal = 1.0."],
        ["", ""],
        ["SHEET: Winner_CorrMatrix", ""],
        ["  Shape", f"{k_elig_w} x {k_elig_w}"],
        ["  Values", "Same, for winner-decile stocks."],
        ["", ""],
        ["SHEET: Loser_Pairwise", ""],
        ["  Columns", "stock_i, stock_j, correlation"],
        ["  Rows", f"All {n_pairs_l} unique pairs (i < j) in the "
                   "loser decile."],
        ["", ""],
        ["SHEET: Winner_Pairwise", ""],
        ["  Columns", "stock_i, stock_j, correlation"],
        ["  Rows", f"All {n_pairs_w} unique pairs in the winner decile."],
    ]
    df_doc = pd.DataFrame(doc_rows, columns=["Item", "Details"])

    # ── Write Excel ──────────────────────────────────────────────────
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path)
                else '.', exist_ok=True)
    base, _ = os.path.splitext(save_path)
    xlsx_path = base + '.xlsx'

    with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
        df_doc.to_excel(writer, sheet_name='Documentation', index=False)
        if corr_l is not None:
            corr_l.to_excel(writer, sheet_name='Loser_CorrMatrix')
            pairs_l.to_excel(writer, sheet_name='Loser_Pairwise',
                             index=False)
        if corr_w is not None:
            corr_w.to_excel(writer, sheet_name='Winner_CorrMatrix')
            pairs_w.to_excel(writer, sheet_name='Winner_Pairwise',
                             index=False)

    print(f"  Saved pairwise correlations: {xlsx_path}")

    return {
        'corr_losers':      corr_l,
        'corr_winners':     corr_w,
        'pairs_losers':     pairs_l,
        'pairs_winners':    pairs_w,
        'avg_corr_losers':  avg_l,
        'avg_corr_winners': avg_w,
        'comomentum':       comom,
    }


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

    result = save_pairwise_correlations(
        data['returns_clean'], momentum_std,
        data['live'], data['ff_factors'],
        data['dates'], data['names'],
        snapshot_week=None,  # last week
        save_path='output_data/pairwise_correlations.xlsx'
    )

    print(f"\n  CoMOM_L = {result['avg_corr_losers']:.6f}")
    print(f"  CoMOM_W = {result['avg_corr_winners']:.6f}")
    print(f"  CoMOM   = {result['comomentum']:.6f}")
    if result['pairs_losers'] is not None:
        print(f"  Loser pairs  : {len(result['pairs_losers'])}")
    if result['pairs_winners'] is not None:
        print(f"  Winner pairs : {len(result['pairs_winners'])}")
