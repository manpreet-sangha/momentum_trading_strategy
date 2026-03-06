# momentum_schedule.py
# =====================================================================
# Momentum Factor Schedule — Lookback, Skip & Compute Date Log
# =====================================================================
# PURPOSE:
#   Reads the weekly dates from US_Dates.xlsx and outputs a CSV that
#   documents every momentum factor computation, showing:
#     - Which 48 weeks form the lookback (compounding) window
#     - Which 4 weeks are skipped (most recent month)
#     - On which date the momentum factor is assigned
#
# LOGIC:
#   The academic momentum factor avoids short-term reversal by skipping
#   the most recent month.  With weekly data:
#
#     Lookback = 48 weeks    (≈ 11 months of compounded returns)
#     Skip     =  4 weeks    (≈ 1 month, most recent)
#     Total    = 52 weeks    needed before the first score
#
#   At each computation date t (from week 52 onwards):
#     - Skip the 4 most recent weeks:  t, t-1, t-2, t-3
#     - Compound returns over the 48 weeks before that:
#       from week (t - 51) to week (t - 4)
#
#   Example (first computation, t = week 52 = 19921225):
#     Lookback :  week 1  to week 48  →  19920103 to 19921127
#     Skipped  :  week 49 to week 52  →  19921204 to 19921225
#     Computed :  week 52             →  19921225
#
#   Example (second computation, t = week 53 = 19930101):
#     Lookback :  week 2  to week 49  →  19920110 to 19921204
#     Skipped  :  week 50 to week 53  →  19921211 to 19930101
#     Computed :  week 53             →  19930101
#
#   The window rolls forward by one week each time.
#
# OUTPUT:
#   output_data/momentum_schedule.csv  with columns:
#     Momentum_Factor_Number   (1, 2, 3, ...)
#     Lookback_Period          (start_date - end_date)
#     Skipped_4_Weeks          (start_date - end_date)
#     Momentum_Compute_Date    (the date the score is assigned)
#
# Standalone:   python momentum_schedule.py
# =====================================================================

import os
import pandas as pd
from logger_setup import _setup_logger

log = _setup_logger()

# ── Parameters ───────────────────────────────────────────────────────
LOOKBACK = 48   # weeks of compounded returns
SKIP     = 4    # most recent weeks to exclude
TOTAL    = LOOKBACK + SKIP   # = 52 weeks needed before first score


def generate_momentum_schedule(datadir='input_data/',
                               output_dir='output_data'):
    """
    Reads US_Dates.xlsx and produces a CSV documenting every momentum
    factor computation: lookback window, skipped weeks, compute date.

    INPUT:
        datadir    : str - folder containing US_Dates.xlsx
        output_dir : str - folder for the output CSV

    OUTPUT:
        schedule_df : pd.DataFrame - the full schedule (also saved as CSV)
    """

    # ── Load dates ───────────────────────────────────────────────────
    dates_df = pd.read_excel(os.path.join(datadir, 'US_Dates.xlsx'),
                             header=None)
    dates_raw = dates_df.iloc[:, 0].values
    dates = pd.to_datetime(dates_raw.astype(str), format='%Y%m%d')
    T = len(dates)

    log.info(f"  Loaded {T} weekly dates: {dates[0].strftime('%Y%m%d')} "
             f"to {dates[-1].strftime('%Y%m%d')}")
    log.info(f"  Lookback = {LOOKBACK} weeks, Skip = {SKIP} weeks, "
             f"Total window = {TOTAL} weeks")
    log.info(f"  First momentum score at week {TOTAL} "
             f"({dates[TOTAL - 1].strftime('%Y%m%d')})")
    log.info(f"  Last  momentum score at week {T} "
             f"({dates[T - 1].strftime('%Y%m%d')})")

    # ── Build schedule ───────────────────────────────────────────────
    fmt = '%Y%m%d'
    rows = []
    factor_num = 0

    for t in range(TOTAL - 1, T):
        # t is the index of the compute date (0-based)
        # Lookback: indices (t - TOTAL + 1) to (t - SKIP)
        # Skipped:  indices (t - SKIP + 1) to t

        lb_start = t - TOTAL + 1       # first week of lookback
        lb_end   = t - SKIP            # last week of lookback
        sk_start = t - SKIP + 1        # first skipped week
        sk_end   = t                   # last skipped week (= compute date)

        factor_num += 1

        rows.append({
            'Momentum_Factor_Number': factor_num,
            'Lookback_Period': (f"{dates[lb_start].strftime(fmt)} - "
                                f"{dates[lb_end].strftime(fmt)}"),
            'Skipped_4_Weeks': (f"{dates[sk_start].strftime(fmt)} - "
                                f"{dates[sk_end].strftime(fmt)}"),
            'Momentum_Compute_Date': dates[t].strftime(fmt),
        })

    schedule_df = pd.DataFrame(rows)

    # ── Save CSV ─────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, 'momentum_schedule.csv')
    schedule_df.to_csv(save_path, index=False)

    log.info(f"  Saved {save_path}  ({len(schedule_df)} momentum factors)")

    # ── Print first and last few rows for quick verification ─────────
    log.info("-" * 70)
    log.info("  FIRST 5 MOMENTUM FACTORS:")
    log.info("-" * 70)
    for _, row in schedule_df.head(5).iterrows():
        log.info(f"    #{row['Momentum_Factor_Number']:>4d}  |  "
                 f"Lookback: {row['Lookback_Period']}  |  "
                 f"Skipped: {row['Skipped_4_Weeks']}  |  "
                 f"Computed: {row['Momentum_Compute_Date']}")

    log.info("-" * 70)
    log.info("  LAST 5 MOMENTUM FACTORS:")
    log.info("-" * 70)
    for _, row in schedule_df.tail(5).iterrows():
        log.info(f"    #{row['Momentum_Factor_Number']:>4d}  |  "
                 f"Lookback: {row['Lookback_Period']}  |  "
                 f"Skipped: {row['Skipped_4_Weeks']}  |  "
                 f"Computed: {row['Momentum_Compute_Date']}")

    log.info("-" * 70)

    return schedule_df


# =====================================================================
# Standalone execution
# =====================================================================
if __name__ == '__main__':
    schedule = generate_momentum_schedule()
    print(f"\nTotal momentum factors: {len(schedule)}")
    print(f"\nFirst 5 rows:")
    print(schedule.head().to_string(index=False))
    print(f"\nLast 5 rows:")
    print(schedule.tail().to_string(index=False))
