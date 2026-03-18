# performance_table_latex.py
# =====================================================================
# Generates a LaTeX comparison table for Standard vs. Adjusted Momentum
# factor return statistics.
# =====================================================================

import os


def generate_performance_table_latex(stats_std, stats_adj, stats_regime=None,
                                      save_path='latex_report/table_performance.tex'):
    """
    Generates a LaTeX table comparing standard, adjusted, and optionally
    regime-conditional momentum factor return statistics.

    INPUTS:
        stats_std    : dict - output of compute_stats() for standard momentum
        stats_adj    : dict - output of compute_stats() for adjusted momentum
        stats_regime : dict or None - output of compute_stats() for regime momentum
        save_path    : str  - path to write the .tex file
    """

    strategies = [stats_std, stats_adj]
    col_headers = ['Standard Momentum', 'Adjusted Momentum']
    if stats_regime is not None:
        strategies.append(stats_regime)
        col_headers.append('Regime Momentum')

    n_cols = len(strategies)

    rows = [
        ('Valid weeks',            'n',        'd',  0),
        ('Ann.\\ mean return (\\%)', 'mean_ann', 'pct', 2),
        ('Ann.\\ std.\\ dev.\\ (\\%)', 'std_ann', 'pct', 2),
        ('Ann.\\ Sharpe ratio',    'sharpe',   'f',  3),
        ('$t$-statistic',          'tstat',    'f',  3),
        ('Skewness',               'skew',     'f',  3),
        ('Excess kurtosis',        'kurt',     'f',  3),
        ('Max drawdown (\\%)',     'max_dd',   'pct', 2),
    ]

    tex = []
    tex.append(r'\begin{table}[htbp]')
    tex.append(r'\centering')
    tex.append(r'\caption{Momentum Factor Returns --- '
               r'annualised performance comparison over the full sample period.}')
    tex.append(r'\label{tab:performance}')
    tex.append(r'\small')
    col_spec = 'l ' + ' '.join(['c'] * n_cols)
    tex.append(r'\begin{tabular}{' + col_spec + '}')
    tex.append(r'\toprule')
    header = '    & ' + ' & '.join(col_headers) + r' \\'
    tex.append(header)
    tex.append(r'\midrule')

    for label, key, fmt, dec in rows:
        cells = [label]
        for s in strategies:
            v = s[key]
            if fmt == 'd':
                cells.append(f'{int(v)}')
            elif fmt == 'pct':
                cells.append(f'{v * 100:.{dec}f}')
            else:
                cells.append(f'{v:.{dec}f}')
        tex.append('    ' + ' & '.join(cells) + r' \\')

    tex.append(r'\bottomrule')
    tex.append(r'\end{tabular}')
    tex.append(r'\end{table}')

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    with open(save_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tex) + '\n')
    print(f"  Saved LaTeX table: {save_path}")
