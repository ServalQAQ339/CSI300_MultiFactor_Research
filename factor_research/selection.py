"""Train-only factor redundancy screening and cross-sectional conditional Rank IC."""
import numpy as np
import pandas as pd


def conditional_rank_ic(
    panel: pd.DataFrame, candidate: str, controls: list[str], min_n: int = 30
) -> pd.DataFrame:
    """Description: Calculate daily rank correlation of candidate residuals with forward returns after projecting out control ranks.

    Parameters:
        panel (pd.DataFrame): Aligned panel containing signal_date, research_label_5d, candidate and controls.
        candidate (str): Candidate factor column name.
        controls (list[str]): Previously selected factor columns.
        min_n (int): Minimum complete-case cross-sectional observations.
    Returns:
        pd.DataFrame: Daily conditional Rank IC and common-sample observation counts.
    Notes:
        This is a descriptive partial rank correlation of candidate with raw label ranks after controlling factor ranks only; it is not a partial correlation controlling the outcome, and it is not incremental model performance. Uses same-day factor values, no future data in residualization.
    """
    rows = []

    for date, day in panel.groupby('signal_date', sort=True):
        cols = [candidate] + controls + ['research_label_5d']
        z = day[cols].replace([np.inf, -np.inf], np.nan).dropna()
        if len(z) < min_n or z[candidate].nunique() < 2 or z.research_label_5d.nunique() < 2:
            rows.append({'signal_date': date, 'n': len(z), 'conditional_rank_ic': np.nan}); continue
        ranks = z.rank(method='average').astype(float)
        x = ranks[candidate].to_numpy(); y = ranks['research_label_5d'].to_numpy()

        if controls:
            a = np.column_stack([np.ones(len(z))] + [ranks[c].to_numpy() for c in controls])
            x = x - a @ np.linalg.lstsq(a, x, rcond=None)[0]

        ic = float(pd.Series(x).corr(pd.Series(y))) if np.std(x) > 1e-9 else np.nan
        rows.append({'signal_date': date, 'n': len(z), 'conditional_rank_ic': ic})

    return pd.DataFrame(rows)


def train_redundancy_groups(corr: pd.DataFrame, threshold: float = 0.8) -> pd.DataFrame:
    """Description: Identify connected components in a thresholded absolute Spearman correlation graph.

    Parameters:
        corr (pd.DataFrame): Square train-period factor correlation matrix.
        threshold (float): Absolute pairwise correlation threshold for linking factors.
    Returns:
        pd.DataFrame: Factor and deterministic connected-component group assignment.
    Notes:
        Connected components are not hierarchical clusters; indirect links can connect weakly correlated factors.
    """
    names = list(corr.index)
    if set(names) != set(corr.columns) or not 0 < threshold <= 1:
        raise ValueError('Invalid matrix or threshold')
    seen = set(); groups = []

    for name in names:
        if name in seen: continue
        todo = [name]; members = []

        while todo:
            node = todo.pop()
            if node in seen: continue
            seen.add(node); members.append(node)
            todo.extend(
                [
                    other
                    for other in names
                    if other not in seen
                    and other != node
                    and pd.notna(corr.loc[node, other])
                    and abs(corr.loc[node, other]) >= threshold
                ]
            )

        group_id = len({x['redundancy_group'] for x in groups}) + 1
        groups.extend(
            {'factor': member, 'redundancy_group': group_id} for member in sorted(members)
        )

    return pd.DataFrame(groups)
