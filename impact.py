"""
"Did it work?" — evaluate a decision or pilot after it was implemented.

Three designs, chosen automatically from what the user provides:
  * Difference-in-differences: date + change date + test group vs control group (strongest)
  * Before vs after:           date + change date (no control group)
  * Test vs control:           test group vs control group (no dates)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import analysis as A


def _verdict(effect, p, desired):
    if p is None or not np.isfinite(p):
        return "Could not evaluate", "grey"
    if p >= 0.05:
        return "No clear evidence of an effect", "amber"
    good = np.sign(effect) == np.sign(desired or 1)
    return ("It worked — significant improvement", "green") if good else ("It backfired — significant worsening", "red")


def _welch(a, b):
    t, p = stats.ttest_ind(a, b, equal_var=False)
    va, vb = a.var(ddof=1) / len(a), b.var(ddof=1) / len(b)
    se = np.sqrt(va + vb)
    dof = (va + vb) ** 2 / (va ** 2 / (len(a) - 1) + vb ** 2 / (len(b) - 1))
    tc = stats.t.ppf(0.975, dof)
    d = a.mean() - b.mean()
    return d, d - tc * se, d + tc * se, p


def evaluate(df, target, date_col=None, change_date=None, group_col=None, test_value=None,
             desired=1, equal_windows=True) -> dict:
    import statsmodels.formula.api as smf

    d = df.copy()
    cav = []
    has_time = bool(date_col and change_date is not None)
    has_group = bool(group_col and test_value is not None)
    if not has_time and not has_group:
        return {"ok": False, "message": "Choose a change date, a test group, or both."}
    cols = [target] + ([date_col] if has_time else []) + ([group_col] if has_group else [])
    d = d[cols].dropna()
    if has_time:
        change_date = pd.Timestamp(change_date)
        d["post"] = (d[date_col] >= change_date).astype(int)
        if equal_windows:
            pre_days = (change_date - d[date_col].min()).days
            post_days = (d[date_col].max() - change_date).days
            w = min(pre_days, post_days)
            d = d[(d[date_col] >= change_date - pd.Timedelta(days=w)) & (d[date_col] <= change_date + pd.Timedelta(days=w))]
            cav.append(f"Compared equal windows of {w} days before and after {change_date:%d %b %Y}.")
    if has_group:
        d["treated"] = (d[group_col].astype(str) == str(test_value)).astype(int)
    d = d.rename(columns={target: "y"})

    # ---------------- Difference-in-differences
    if has_time and has_group:
        design = "Difference-in-differences (test vs control, before vs after)"
        cells = d.groupby(["treated", "post"])["y"].agg(["mean", "count"])
        if len(cells) < 4 or cells["count"].min() < 5:
            return {"ok": False, "message": "Each of the four groups (test/control × before/after) needs at least 5 rows."}
        m = smf.ols("y ~ treated + post + treated:post", data=d).fit(cov_type="HC3")
        eff, p = float(m.params["treated:post"]), float(m.pvalues["treated:post"])
        lo, hi = (float(v) for v in m.conf_int().loc["treated:post"])
        base = cells.loc[(1, 0), "mean"]
        table = pd.DataFrame({
            "Group": [f"Test ({test_value})", "Control (others)"],
            "Before": [cells.loc[(1, 0), "mean"], cells.loc[(0, 0), "mean"]],
            "After": [cells.loc[(1, 1), "mean"], cells.loc[(0, 1), "mean"]]})
        table["Change"] = table["After"] - table["Before"]
        table["Change %"] = table["Change"] / table["Before"] * 100
        naive = table.loc[0, "Change"]
        # parallel-trends check on the pre-period
        pre = d[d.post == 0].copy()
        pre["t"] = (pre[date_col] - pre[date_col].min()).dt.days
        try:
            pt = smf.ols("y ~ t * treated", data=pre).fit()
            pt_p = float(pt.pvalues.get("t:treated", np.nan))
            if pt_p < 0.05:
                cav.append(f"Before the change, the test and control groups were already trending differently (p = {pt_p:.3f}); "
                           "the estimate may be biased.")
        except Exception:  # noqa: BLE001
            pt_p = np.nan
        explain = (f"The test group changed by {A._fmt(naive)} per row, but the control group also changed by "
                   f"{A._fmt(table.loc[1, 'Change'])}. Removing the change that happened anyway leaves an effect of "
                   f"{A._fmt(eff)} ({eff / base * 100 if base else np.nan:+.1f}%), p = {A._p(p)}.")
        if abs(naive - eff) > 0.3 * abs(eff) + 1e-9:
            cav.append("A simple before/after comparison would have given a misleading answer here — the control group matters.")
        fig = _time_fig(d, date_col, change_date, test_value, target, by_group=True)
    elif has_time:
        design = "Before vs after (no control group)"
        a, b = d[d.post == 1]["y"], d[d.post == 0]["y"]
        if len(a) < 5 or len(b) < 5:
            return {"ok": False, "message": "Need at least 5 rows before and after the change date."}
        eff, lo, hi, p = _welch(a, b)
        base = b.mean()
        table = pd.DataFrame({"Period": ["Before", "After"], "Average": [b.mean(), a.mean()], "Rows": [len(b), len(a)]})
        explain = f"Average {target} moved from {A._fmt(b.mean())} to {A._fmt(a.mean())} ({eff / base * 100 if base else np.nan:+.1f}%), p = {A._p(p)}."
        cav.append("Without a control group, seasonality or market changes in the same period could explain the difference. "
                   "Add a test group (e.g. pilot region) for a stronger answer.")
        fig = _time_fig(d, date_col, change_date, None, target, by_group=False)
    else:
        design = "Test group vs control group (no dates)"
        a, b = d[d.treated == 1]["y"], d[d.treated == 0]["y"]
        if len(a) < 5 or len(b) < 5:
            return {"ok": False, "message": "Need at least 5 rows in both the test and control groups."}
        eff, lo, hi, p = _welch(a, b)
        base = b.mean()
        table = pd.DataFrame({"Group": [f"Test ({test_value})", "Control (others)"], "Average": [a.mean(), b.mean()],
                              "Rows": [len(a), len(b)]})
        explain = f"The test group averages {A._fmt(a.mean())} vs {A._fmt(b.mean())} for control ({eff / base * 100 if base else np.nan:+.1f}%), p = {A._p(p)}."
        cav.append("Groups were not compared before the change — they may have been different to begin with.")
        fig = A._style(go.Figure(go.Bar(x=table["Group"], y=table["Average"], marker_color=[A.PALETTE[0], "#94a3b8"],
                                        text=[A._fmt(v) for v in table["Average"]], textposition="outside")), 340)
        fig.update_layout(title=f"Average {target}: test vs control")

    verdict, tone = _verdict(eff, p, desired)
    return {"ok": True, "target": target, "design": design, "effect": eff, "ci": (lo, hi), "p": p,
            "pct": eff / base * 100 if base else np.nan, "verdict": verdict, "tone": tone, "explain": explain,
            "table": table, "caveats": cav, "fig": fig, "n": len(d)}


def _time_fig(d, date_col, change_date, test_value, target, by_group):
    span = (d[date_col].max() - d[date_col].min()).days
    freq = "W-MON" if span <= 400 else "MS"
    fig = go.Figure()
    if by_group:
        for flag, name, color in [(1, f"Test ({test_value})", A.PALETTE[0]), (0, "Control (others)", "#94a3b8")]:
            s = d[d.treated == flag].set_index(date_col)["y"].resample(freq).mean().dropna()
            fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=name, line=dict(color=color, width=2.5)))
    else:
        s = d.set_index(date_col)["y"].resample(freq).mean().dropna()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=target, line=dict(color=A.PALETTE[0], width=2.5)))
    fig.add_shape(type="line", x0=change_date, x1=change_date, y0=0, y1=1, yref="paper", line=dict(color="#ef4444", dash="dash"))
    fig.add_annotation(x=change_date, y=1, yref="paper", text="Change made", showarrow=False, yshift=10, font=dict(color="#ef4444"))
    fig.update_layout(title=f"Average {target} over time", yaxis_title=target)
    return A._style(fig, 380)
