"""
What-if estimator: "If we change the lever by X, roughly how much could the outcome change?"

Uses the regression already chosen by the agent (lever effect while controlling for other
factors) and reports a range, not a single promise. Clearly labelled as an estimate based on
historical association, not a guaranteed causal effect.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy import stats

import analysis as A
from planner import DecisionFrame, _predictors


def _fmt(x):
    return A._fmt(x)


def lever_range(df, fr: DecisionFrame):
    s = A.apply_focus(df, fr.focus)[fr.lever].dropna()
    if s.empty:
        s = df[fr.lever].dropna()
    return float(s.min()), float(s.mean()), float(s.max()), float(s.std() or 1)


def _rows_per_month(df, fr):
    if not fr.date_col:
        return None
    d = A.apply_focus(df, fr.focus)[fr.date_col].dropna()
    months = max(1, (d.max() - d.min()).days / 30.44)
    return len(d) / months


def numeric_whatif(df, sch, fr: DecisionFrame, change: float) -> dict:
    lo_obs, cur, hi_obs, _ = lever_range(df, fr)
    new_val = cur + change
    preds, cats = _predictors(df, sch, fr)
    if fr.lever not in preds:
        preds = [fr.lever] + preds
    outcomes = [fr.target] + [o for o in fr.secondary_outcomes if o != fr.target][:2]
    rows, notes = [], []
    seg = A.apply_focus(df, fr.focus)
    rpm = _rows_per_month(df, fr)
    for y in outcomes:
        r = A.regression(df, y, [p for p in preds if p != y], categorical=cats)
        if not r["ok"] or "model" not in r or fr.lever not in r["model"].params.index:
            notes.append(f"Could not estimate the effect on {y}: {r['finding']}")
            continue
        m = r["model"]
        coef = float(m.params[fr.lever])
        ci_lo, ci_hi = (float(v) for v in m.conf_int().loc[fr.lever])
        p = float(m.pvalues[fr.lever])
        eff, a, b = coef * change, ci_lo * change, ci_hi * change
        a, b = min(a, b), max(a, b)
        base = float(seg[y].mean())
        rows.append({"Outcome": y, "Current average (per row)": base, "Estimated change (per row)": eff,
                     "Likely range low": a, "Likely range high": b,
                     "% change": eff / base * 100 if base else np.nan,
                     "Monthly impact": eff * rpm if rpm else None,
                     "Reliable?": "Yes (p < 0.05)" if p < 0.05 else "No — range includes zero",
                     "p": p, "R2": float(m.rsquared)})
    if new_val > hi_obs or new_val < lo_obs:
        notes.append(f"⚠️ {fr.lever} = {new_val:,.2f} is outside the range seen in your data ({lo_obs:,.2f}–{hi_obs:,.2f}). "
                     "Estimates outside past experience are guesses.")
    notes.append("Based on a straight-line relationship in past data, controlling for "
                 + (", ".join([p for p in preds if p != fr.lever][:5]) or "no other factors")
                 + ". It shows association, not guaranteed cause and effect.")
    if fr.focus:
        notes.append(f"Effect is estimated across all data (with {', '.join(fr.focus)} as a control) and applied to {A.focus_label(fr.focus)}.")
    return {"kind": "numeric", "lever": fr.lever, "current": cur, "new": new_val, "change": change,
            "rows": rows, "notes": notes, "fig": _fig(rows), "rows_per_month": rpm}


def categorical_whatif(df, fr: DecisionFrame, share_pct: float) -> dict:
    d = df[[fr.target, fr.lever]].dropna()
    a = d[d[fr.lever].astype(str) == str(fr.lever_value)][fr.target]
    b = d[d[fr.lever].astype(str) != str(fr.lever_value)][fr.target]
    notes = []
    if len(a) < 3 or len(b) < 3:
        return {"kind": "categorical", "rows": [], "notes": ["Not enough data for this option."], "fig": None}
    diff = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    dof = se ** 4 / ((a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    tcrit = stats.t.ppf(0.975, dof)
    p = stats.ttest_ind(a, b, equal_var=False).pvalue
    s = share_pct / 100
    base = d[fr.target].mean()
    eff, lo, hi = diff * s, (diff - tcrit * se) * s, (diff + tcrit * se) * s
    sign = fr.action_direction or 1
    if sign < 0:  # moving volume AWAY from the option
        eff, lo, hi = -eff, -hi, -lo
    rows = [{"Outcome": fr.target, "Current average (per row)": base, "Estimated change (per row)": eff,
             "Likely range low": lo, "Likely range high": hi, "% change": eff / base * 100 if base else np.nan,
             "Monthly impact": None, "Reliable?": "Yes (p < 0.05)" if p < 0.05 else "No — range includes zero", "p": p}]
    notes.append(f"Assumes {share_pct:.0f}% of activity moves {'to' if sign > 0 else 'away from'} {fr.lever_value} and performs like "
                 f"{fr.lever_value} has historically ({_fmt(a.mean())} vs {_fmt(b.mean())} for other options).")
    notes.append("Differences between options may reflect who/what they serve, not the option itself.")
    return {"kind": "categorical", "rows": rows, "notes": notes, "fig": _fig(rows)}


def _fig(rows):
    if not rows:
        return None
    df = pd.DataFrame(rows)
    pct = df["% change"].fillna(0)
    lo = df["Likely range low"] / df["Current average (per row)"].replace(0, np.nan) * 100
    hi = df["Likely range high"] / df["Current average (per row)"].replace(0, np.nan) * 100
    colors = ["#10b981" if v >= 0 else "#ef4444" for v in pct]
    fig = go.Figure(go.Bar(x=df["Outcome"], y=pct, marker_color=colors, text=[f"{v:+.1f}%" for v in pct], textposition="outside",
                           error_y=dict(type="data", symmetric=False, array=(hi - pct).fillna(0), arrayminus=(pct - lo).fillna(0))))
    fig.add_hline(y=0, line_color="#94a3b8")
    fig.update_layout(title="Estimated % change (bars) with likely range (whiskers)", yaxis_title="% change vs current average")
    return A._style(fig, 360)
