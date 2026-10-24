"""
Deterministic statistical "tools" used by the DecisionCheck agent.

Every function returns a plain dict (an AnalysisResult) containing:
    tool, title, method, n, stats, p_value, effect, significant,
    finding (a factual sentence computed from numbers - never from the AI),
    direction (+1 / -1 / 0), caveats (list), table (optional DataFrame),
    fig (Plotly figure)
The language model only ever *reads* these results; it never calculates them.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

warnings.filterwarnings("ignore")

ALPHA = 0.05
PALETTE = ["#2563EB", "#F97316", "#10B981", "#A855F7", "#EF4444", "#0EA5E9", "#EAB308", "#64748B"]
TEMPLATE = "plotly_white"


def _fmt(x, d=2):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    if abs(x) >= 1e6:
        return f"{x/1e6:,.{d}f}M"
    if abs(x) >= 1e4:
        return f"{x:,.0f}"
    return f"{x:,.{d}f}"


def _p(p):
    if p is None or not np.isfinite(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _base(tool, title, method, n):
    return {"tool": tool, "title": title, "method": method, "n": int(n), "stats": {},
            "p_value": None, "effect": None, "effect_label": "", "significant": None,
            "finding": "", "direction": 0, "caveats": [], "table": None, "fig": None, "ok": True}


def _fail(tool, title, reason):
    r = _base(tool, title, "—", 0)
    r.update(ok=False, finding=reason)
    return r


def _style(fig, height=380):
    fig.update_layout(template=TEMPLATE, height=height, margin=dict(l=10, r=10, t=50, b=10),
                      font=dict(family="Inter, Segoe UI, sans-serif", size=13),
                      title_font=dict(size=15), legend=dict(orientation="h", y=-0.2))
    return fig


def apply_focus(df: pd.DataFrame, focus: dict | None) -> pd.DataFrame:
    """Filter to the segment the decision is about, e.g. {'Product': 'Product A'}."""
    if not focus:
        return df
    out = df
    for col, val in focus.items():
        if col in out.columns:
            out = out[out[col].astype(str) == str(val)]
    return out


def focus_label(focus: dict | None) -> str:
    return " & ".join(f"{k} = {v}" for k, v in (focus or {}).items()) or "all data"


# --------------------------------------------------------------------------- #
# 1. Descriptive statistics
# --------------------------------------------------------------------------- #
def descriptive(df, target, group=None, focus=None):
    title = f"Descriptive statistics — {target}" + (f" by {group}" if group else "")
    d = apply_focus(df, focus)
    s = d[target].dropna()
    if len(s) == 0:
        return _fail("descriptive", title, f"No data for '{target}' in {focus_label(focus)}.")
    r = _base("descriptive", title, "Mean, median, spread, skewness", len(s))
    desc = {"Count": len(s), "Mean": s.mean(), "Median": s.median(), "Std dev": s.std(),
            "Min": s.min(), "Max": s.max(), "Total": s.sum(),
            "CV %": s.std() / s.mean() * 100 if s.mean() else np.nan, "Skewness": stats.skew(s) if len(s) > 2 else np.nan}
    r["stats"] = {k: float(v) for k, v in desc.items()}
    skew_txt = "roughly symmetric" if abs(desc["Skewness"]) < 0.5 else ("right-skewed (a few large values)" if desc["Skewness"] > 0 else "left-skewed")
    r["finding"] = (f"For {focus_label(focus)}, {target} averages {_fmt(desc['Mean'])} (median {_fmt(desc['Median'])}), "
                    f"ranging {_fmt(desc['Min'])}–{_fmt(desc['Max'])}; the distribution is {skew_txt}.")
    if group and group in d.columns:
        t = d.groupby(group)[target].agg(["count", "mean", "median", "std", "sum"]).round(2).sort_values("mean", ascending=False)
        r["table"] = t.reset_index()
        fig = px.bar(t.reset_index(), x=group, y="mean", color_discrete_sequence=PALETTE, text_auto=".2s",
                     title=f"Average {target} by {group}")
    else:
        r["table"] = pd.DataFrame([{k: round(v, 3) for k, v in desc.items()}])
        fig = px.histogram(d, x=target, nbins=30, color_discrete_sequence=PALETTE, title=f"Distribution of {target}")
    r["fig"] = _style(fig)
    if len(s) < 30:
        r["caveats"].append("Fewer than 30 observations — summary values may be unstable.")
    return r


# --------------------------------------------------------------------------- #
# 2. Trend analysis
# --------------------------------------------------------------------------- #
def _auto_freq(dates: pd.Series):
    span = (dates.max() - dates.min()).days
    if span > 365 * 3:
        return "QS", "quarter"
    if span > 120:
        return "MS", "month"
    if span > 21:
        return "W-MON", "week"
    return "D", "day"


def trend(df, target, date_col, focus=None, agg="sum"):
    title = f"Trend of {target} over time" + (f" ({focus_label(focus)})" if focus else "")
    d = apply_focus(df, focus)[[date_col, target]].dropna()
    if len(d) < 4:
        return _fail("trend", title, "Not enough dated observations to test a trend.")
    freq, unit = _auto_freq(d[date_col])
    series = d.set_index(date_col)[target].resample(freq).agg(agg)
    series = series[series.index >= d[date_col].min() - pd.Timedelta(days=31)]
    if agg == "sum":
        series = series.replace(0, np.nan).dropna() if (series == 0).mean() < 0.5 else series
    # drop an incomplete trailing period if it is much smaller (partial month)
    if len(series) > 4 and series.iloc[-1] < 0.5 * series.iloc[:-1].median():
        series = series.iloc[:-1]
        partial = True
    else:
        partial = False
    if len(series) < 4:
        return _fail("trend", title, f"Only {len(series)} {unit}s of data — need at least 4 periods for a trend test.")
    y = series.values.astype(float)
    x = np.arange(len(y))
    lr = stats.linregress(x, y)
    rho, p_mk = stats.spearmanr(x, y)
    mean_y = y.mean() if y.mean() else np.nan
    pct_per_period = lr.slope / mean_y * 100 if mean_y else np.nan
    h = len(y) // 2
    first, last = y[:h].mean(), y[-h:].mean()
    change = (last - first) / first * 100 if first else np.nan

    r = _base("trend", title, f"Linear trend + Spearman monotonic test on {len(y)} {unit}ly totals", len(d))
    r["p_value"] = float(lr.pvalue)
    r["effect"] = float(pct_per_period)
    r["effect_label"] = f"% change per {unit}"
    r["significant"] = bool(lr.pvalue < ALPHA)
    r["direction"] = int(np.sign(lr.slope)) if lr.pvalue < ALPHA else 0
    r["stats"] = {"slope": lr.slope, "r_squared": lr.rvalue ** 2, "p_value": lr.pvalue,
                  "spearman_rho": rho, "spearman_p": p_mk, "periods": len(y),
                  "first_half_avg": first, "second_half_avg": last, "half_change_pct": change}
    word = "declining" if lr.slope < 0 else "increasing"
    if lr.pvalue < ALPHA:
        r["finding"] = (f"{target} for {focus_label(focus)} shows a statistically significant {word} trend: "
                        f"about {pct_per_period:+.1f}% of its average per {unit} (p = {_p(lr.pvalue)}, R² = {lr.rvalue**2:.2f}). "
                        f"Second-half average is {change:+.1f}% vs first half.")
    else:
        r["finding"] = (f"No statistically significant trend in {target} for {focus_label(focus)} "
                        f"(slope {pct_per_period:+.1f}% per {unit}, p = {_p(lr.pvalue)}). "
                        f"The {change:+.1f}% half-on-half change may be ordinary fluctuation.")
    if partial:
        r["caveats"].append(f"The last {unit} looked incomplete and was excluded.")
    if len(y) < 8:
        r["caveats"].append(f"Only {len(y)} periods — trend tests have low power.")
    # seasonality hint: lag-1 autocorrelation of residuals
    resid = y - (lr.intercept + lr.slope * x)
    if len(y) >= 8:
        ac = pd.Series(resid).autocorr(1)
        if ac is not None and abs(ac) > 0.5:
            r["caveats"].append("Residuals are autocorrelated — seasonality or cycles may be influencing the trend.")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=series.index, y=y, mode="lines+markers", name=target, line=dict(color=PALETTE[0], width=2.5)))
    fig.add_trace(go.Scatter(x=series.index, y=lr.intercept + lr.slope * x, mode="lines", name="Trend line",
                             line=dict(color=PALETTE[1], dash="dash")))
    fig.update_layout(title=f"{agg.title()} of {target} per {unit} — {focus_label(focus)}", yaxis_title=target)
    r["fig"] = _style(fig)
    r["table"] = pd.DataFrame({"Period": series.index.strftime("%Y-%m-%d"), target: np.round(y, 2)})
    return r


# --------------------------------------------------------------------------- #
# 3. Correlation
# --------------------------------------------------------------------------- #
def _strength(r):
    a = abs(r)
    return "very weak" if a < 0.1 else "weak" if a < 0.3 else "moderate" if a < 0.5 else "strong" if a < 0.7 else "very strong"


def correlation(df, x, y, focus=None):
    title = f"Correlation: {x} vs {y}" + (f" ({focus_label(focus)})" if focus else "")
    d = apply_focus(df, focus)[[x, y]].dropna()
    if len(d) < 5 or d[x].nunique() < 2 or d[y].nunique() < 2:
        return _fail("correlation", title, "Not enough varied data points to compute a correlation.")
    pr, pp = stats.pearsonr(d[x], d[y])
    sr, sp = stats.spearmanr(d[x], d[y])
    r = _base("correlation", title, "Pearson (linear) and Spearman (rank) correlation", len(d))
    # Fisher CI
    if len(d) > 3 and abs(pr) < 1:
        z = np.arctanh(pr); se = 1 / np.sqrt(len(d) - 3)
        lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    else:
        lo = hi = np.nan
    r.update(p_value=float(pp), effect=float(pr), effect_label="Pearson r", significant=bool(pp < ALPHA),
             direction=int(np.sign(pr)) if pp < ALPHA else 0)
    r["stats"] = {"pearson_r": pr, "pearson_p": pp, "spearman_rho": sr, "spearman_p": sp, "ci_low": lo, "ci_high": hi}
    rel = "positive" if pr > 0 else "negative"
    if pp < ALPHA:
        r["finding"] = (f"{x} and {y} have a {_strength(pr)} {rel} relationship (r = {pr:.2f}, 95% CI {lo:.2f} to {hi:.2f}, "
                        f"p = {_p(pp)}). Higher {x} tends to go with {'higher' if pr > 0 else 'lower'} {y}.")
    else:
        r["finding"] = f"No statistically significant linear relationship between {x} and {y} (r = {pr:.2f}, p = {_p(pp)})."
    if abs(pr - sr) > 0.2:
        r["caveats"].append("Pearson and Spearman differ noticeably — relationship may be non-linear or driven by outliers.")
    r["caveats"].append("Correlation does not prove causation; other factors may drive both variables.")
    samp = d.sample(min(len(d), 3000), random_state=1)
    fig = px.scatter(samp, x=x, y=y, trendline="ols", opacity=0.55, color_discrete_sequence=PALETTE,
                     title=f"{y} vs {x} (r = {pr:.2f})")
    r["fig"] = _style(fig)
    return r


def correlation_matrix(df, cols, focus=None):
    title = "Correlation heatmap of numeric measures"
    d = apply_focus(df, focus)[cols].dropna()
    if len(cols) < 2 or len(d) < 5:
        return _fail("correlation_matrix", title, "Need at least two numeric columns.")
    m = d.corr(method="pearson")
    r = _base("correlation_matrix", title, "Pairwise Pearson correlations", len(d))
    pairs = []
    for i, a in enumerate(cols):
        for b in cols[i + 1:]:
            pairs.append((a, b, m.loc[a, b]))
    pairs.sort(key=lambda t: -abs(t[2]))
    top = pairs[:3]
    r["finding"] = "Strongest relationships: " + "; ".join(f"{a} ↔ {b} (r = {v:.2f})" for a, b, v in top) + "."
    r["table"] = m.round(2).reset_index().rename(columns={"index": "Variable"})
    fig = px.imshow(m.round(2), text_auto=True, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
                    title=title)
    r["fig"] = _style(fig, 420)
    return r


# --------------------------------------------------------------------------- #
# 4. Regression
# --------------------------------------------------------------------------- #
def regression(df, target, predictors, focus=None, categorical=None):
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    categorical = [c for c in (categorical or []) if c in predictors]
    title = f"Regression: what drives {target}?" + (f" ({focus_label(focus)})" if focus else "")
    preds = [p for p in predictors if p != target and p in df.columns]
    d = apply_focus(df, focus)[[target] + preds].dropna()
    # drop categoricals that became constant after focus
    preds = [p for p in preds if d[p].nunique() > 1]
    categorical = [c for c in categorical if c in preds]
    # drop categorical columns fully determined by another one (e.g. Category is implied by Product)
    for a in list(categorical):
        for b in list(categorical):
            if a != b and a in categorical and b in categorical and d.groupby(a)[b].nunique().max() == 1:
                categorical.remove(b)
                preds.remove(b)
    # drop numeric controls (not the lever, which is listed first) that are almost fully explained
    # by a categorical control, e.g. Unit_Price is mostly determined by Product
    dropped = []
    for p in [p for p in preds[1:] if p not in categorical]:
        for c in categorical:
            grp = d.groupby(c)[p].transform("mean")
            ss_t = ((d[p] - d[p].mean()) ** 2).sum()
            if ss_t and 1 - ((d[p] - grp) ** 2).sum() / ss_t > 0.9:
                preds.remove(p); dropped.append(f"{p} (mostly determined by {c})")
                break
    if not preds:
        return _fail("regression", title, "No usable predictors with variation.")
    X = pd.get_dummies(d[preds], columns=categorical, drop_first=True, dtype=float)
    X = X.loc[:, X.std() > 0]
    if len(d) < X.shape[1] + 10:
        return _fail("regression", title, f"Only {len(d)} rows for {X.shape[1]} predictors — too few for a reliable regression.")
    Xc = sm.add_constant(X.astype(float))
    model = sm.OLS(d[target].astype(float), Xc).fit(cov_type="HC3")
    r = _base("regression", title, f"OLS multiple regression (robust HC3 standard errors), {X.shape[1]} predictor terms", len(d))
    coefs = pd.DataFrame({"Term": model.params.index, "Coefficient": model.params.values,
                          "Std error": model.bse.values, "p-value": model.pvalues.values,
                          "CI low": model.conf_int()[0].values, "CI high": model.conf_int()[1].values})
    coefs = coefs[coefs.Term != "const"].copy()
    coefs["Significant"] = coefs["p-value"] < ALPHA
    r["table"] = coefs.round(4)
    r.update(p_value=float(model.f_pvalue), effect=float(model.rsquared), effect_label="R²",
             significant=bool(model.f_pvalue < ALPHA))
    r["stats"] = {"r_squared": model.rsquared, "adj_r_squared": model.rsquared_adj, "f_pvalue": model.f_pvalue,
                  "n": int(model.nobs), "coefficients": {t: {"coef": c, "p": p} for t, c, p in
                                                        zip(coefs.Term, coefs.Coefficient, coefs["p-value"])}}
    sig = coefs[coefs.Significant].sort_values("p-value")
    if len(sig):
        parts = [f"{t} ({'+' if c > 0 else ''}{_fmt(c)} per unit, p = {_p(p)})" for t, c, p in
                 zip(sig.Term.head(4), sig.Coefficient.head(4), sig["p-value"].head(4))]
        r["finding"] = (f"The model explains {model.rsquared:.0%} of the variation in {target} (adj. R² = {model.rsquared_adj:.2f}). "
                        f"Significant drivers, holding others constant: " + "; ".join(parts) + ".")
    else:
        r["finding"] = f"The model explains {model.rsquared:.0%} of the variation in {target}; no predictor is individually significant."
    # diagnostics
    if X.shape[1] > 1:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                vifs = {X.columns[i]: variance_inflation_factor(Xc.values, i + 1) for i in range(X.shape[1])}
            high = [k for k, v in vifs.items() if v > 5]
            if high:
                r["caveats"].append(f"Multicollinearity (VIF > 5) for: {', '.join(high[:4])} — individual effects may be unreliable.")
        except Exception:  # noqa: BLE001
            pass
    if len(model.resid) >= 8:
        _, pnorm = stats.normaltest(model.resid) if len(model.resid) >= 20 else stats.shapiro(model.resid)
        if pnorm < 0.01:
            r["caveats"].append("Residuals are not normally distributed; robust standard errors were used to compensate.")
    if dropped:
        r["caveats"].append("Left out to avoid duplicated information: " + ", ".join(dropped) + ".")
    if model.rsquared < 0.1:
        r["caveats"].append("Low R² — most variation is explained by factors not in this dataset.")
    r["caveats"].append("Regression shows association after adjusting for included variables, not proven causation.")
    plot = coefs.sort_values("Coefficient")
    fig = go.Figure(go.Bar(x=plot.Coefficient, y=plot.Term, orientation="h",
                           marker_color=[PALETTE[2] if s else "#CBD5E1" for s in plot.Significant],
                           error_x=dict(type="data", symmetric=False, array=plot["CI high"] - plot.Coefficient,
                                        arrayminus=plot.Coefficient - plot["CI low"])))
    fig.update_layout(title=f"Effect on {target} (green = significant)", xaxis_title="Coefficient")
    r["fig"] = _style(fig, max(320, 40 * len(plot) + 120))
    r["model"] = model
    return r


# --------------------------------------------------------------------------- #
# 5. Group comparison: t-test / ANOVA
# --------------------------------------------------------------------------- #
def group_compare(df, target, group, focus=None, max_groups=12):
    title = f"Does {target} differ by {group}?" + (f" ({focus_label(focus)})" if focus else "")
    d = apply_focus(df, focus)[[target, group]].dropna()
    counts = d[group].value_counts()
    keep = counts[counts >= 3].index[:max_groups]
    d = d[d[group].isin(keep)]
    groups = [g[target].values for _, g in d.groupby(group)]
    names = [str(k) for k, _ in d.groupby(group)]
    if len(groups) < 2:
        return _fail("group_compare", title, f"'{group}' needs at least 2 groups with 3+ observations each.")
    means = d.groupby(group)[target].agg(["count", "mean", "median", "std"]).sort_values("mean", ascending=False)
    lev_p = stats.levene(*groups).pvalue if all(len(g) > 1 for g in groups) else np.nan
    if len(groups) == 2:
        t, p = stats.ttest_ind(groups[0], groups[1], equal_var=False)
        u, pu = stats.mannwhitneyu(groups[0], groups[1])
        pooled = np.sqrt((np.var(groups[0], ddof=1) + np.var(groups[1], ddof=1)) / 2)
        dval = (groups[0].mean() - groups[1].mean()) / pooled if pooled else 0
        r = _base("group_compare", title, "Welch's t-test (+ Mann-Whitney U check)", len(d))
        r.update(p_value=float(p), effect=float(abs(dval)), effect_label="Cohen's d")
        r["stats"] = {"t": t, "p_value": p, "mannwhitney_p": pu, "cohens_d": dval, "levene_p": lev_p}
        size = "negligible" if abs(dval) < 0.2 else "small" if abs(dval) < 0.5 else "medium" if abs(dval) < 0.8 else "large"
        hi, lo = means.index[0], means.index[-1]
        if p < ALPHA:
            r["finding"] = (f"{target} differs significantly between {group} groups: {hi} averages {_fmt(means.loc[hi, 'mean'])} "
                            f"vs {lo} at {_fmt(means.loc[lo, 'mean'])} (p = {_p(p)}, {size} effect, d = {abs(dval):.2f}).")
        else:
            r["finding"] = (f"No significant difference in {target} between {names[0]} and {names[1]} "
                            f"(p = {_p(p)}, d = {abs(dval):.2f}).")
        if (p < ALPHA) != (pu < ALPHA):
            r["caveats"].append("The t-test and the non-parametric Mann-Whitney test disagree — result is sensitive to outliers/skew.")
    else:
        f, p = stats.f_oneway(*groups)
        h, pk = stats.kruskal(*groups)
        grand = d[target].mean()
        ss_b = sum(len(g) * (g.mean() - grand) ** 2 for g in groups)
        ss_t = ((d[target] - grand) ** 2).sum()
        eta = ss_b / ss_t if ss_t else 0
        r = _base("group_compare", title, f"One-way ANOVA across {len(groups)} groups (+ Kruskal-Wallis check)", len(d))
        r.update(p_value=float(p), effect=float(eta), effect_label="eta²")
        r["stats"] = {"F": f, "p_value": p, "kruskal_p": pk, "eta_squared": eta, "levene_p": lev_p}
        size = "negligible" if eta < 0.01 else "small" if eta < 0.06 else "medium" if eta < 0.14 else "large"
        hi, lo = means.index[0], means.index[-1]
        if p < ALPHA:
            r["finding"] = (f"Average {target} differs significantly across {group} (ANOVA p = {_p(p)}, {size} effect, η² = {eta:.2f}). "
                            f"Highest: {hi} ({_fmt(means.loc[hi, 'mean'])}); lowest: {lo} ({_fmt(means.loc[lo, 'mean'])}).")
            try:
                from statsmodels.stats.multicomp import pairwise_tukeyhsd
                tk = pairwise_tukeyhsd(d[target], d[group].astype(str))
                tdf = pd.DataFrame(tk.summary().data[1:], columns=tk.summary().data[0])
                r["posthoc"] = tdf
                nsig = int(tdf["reject"].sum())
                r["finding"] += f" Tukey post-hoc: {nsig} of {len(tdf)} group pairs differ significantly."
            except Exception:  # noqa: BLE001
                pass
        else:
            r["finding"] = f"No significant difference in average {target} across {group} groups (ANOVA p = {_p(p)}, η² = {eta:.2f})."
        if (p < ALPHA) != (pk < ALPHA):
            r["caveats"].append("ANOVA and the non-parametric Kruskal-Wallis test disagree — interpret with caution.")
    r["significant"] = bool(r["p_value"] < ALPHA)
    if np.isfinite(lev_p) and lev_p < ALPHA:
        r["caveats"].append("Group variances are unequal (Levene test); Welch/non-parametric checks were used alongside.")
    if counts.min() < 10:
        r["caveats"].append("Some groups have fewer than 10 observations.")
    r["table"] = means.round(2).reset_index()
    order = means.index.astype(str).tolist()
    d2 = d.copy(); d2[group] = d2[group].astype(str)
    fig = px.box(d2, x=group, y=target, color=group, category_orders={group: order}, points="outliers",
                 color_discrete_sequence=PALETTE, title=f"{target} by {group}")
    fig.update_layout(showlegend=False)
    r["fig"] = _style(fig)
    return r


# --------------------------------------------------------------------------- #
# 6. Focus segment vs rest
# --------------------------------------------------------------------------- #
def segment_vs_rest(df, target, col, value):
    title = f"{col} = {value} vs all other {col} values — {target}"
    d = df[[target, col]].dropna()
    a = d[d[col].astype(str) == str(value)][target]
    b = d[d[col].astype(str) != str(value)][target]
    if len(a) < 3 or len(b) < 3:
        return _fail("segment_vs_rest", title, "Not enough observations in the segment or comparison group.")
    t, p = stats.ttest_ind(a, b, equal_var=False)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    dval = (a.mean() - b.mean()) / pooled if pooled else 0
    r = _base("segment_vs_rest", title, "Welch's t-test: focus segment vs the rest", len(d))
    r.update(p_value=float(p), effect=float(dval), effect_label="Cohen's d", significant=bool(p < ALPHA),
             direction=int(np.sign(a.mean() - b.mean())) if p < ALPHA else 0)
    r["stats"] = {"segment_mean": a.mean(), "rest_mean": b.mean(), "segment_n": len(a), "rest_n": len(b), "p_value": p}
    diff = (a.mean() - b.mean()) / b.mean() * 100 if b.mean() else np.nan
    if p < ALPHA:
        r["finding"] = (f"{value} averages {_fmt(a.mean())} {target}, {diff:+.1f}% vs the rest ({_fmt(b.mean())}); "
                        f"the difference is significant (p = {_p(p)}).")
    else:
        r["finding"] = f"{value}'s average {target} ({_fmt(a.mean())}) is not significantly different from the rest ({_fmt(b.mean())}, p = {_p(p)})."
    plot = pd.DataFrame({"Group": [str(value), "All others"], target: [a.mean(), b.mean()],
                         "se": [a.std() / np.sqrt(len(a)), b.std() / np.sqrt(len(b))]})
    fig = px.bar(plot, x="Group", y=target, error_y=plot["se"] * 1.96, color="Group",
                 color_discrete_sequence=PALETTE, title=f"Average {target}: {value} vs others (95% CI)")
    fig.update_layout(showlegend=False)
    r["fig"] = _style(fig, 340)
    return r


# --------------------------------------------------------------------------- #
# 7. Outlier detection
# --------------------------------------------------------------------------- #
def outliers(df, target, focus=None, id_cols=None):
    title = f"Unusual observations in {target}" + (f" ({focus_label(focus)})" if focus else "")
    d = apply_focus(df, focus)
    s = d[target].dropna()
    if len(s) < 8:
        return _fail("outliers", title, "Too few values to detect outliers reliably.")
    q1, q3 = s.quantile([0.25, 0.75]); iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = (d[target] < lo) | (d[target] > hi)
    z = (s - s.mean()) / s.std() if s.std() else s * 0
    extreme = int((z.abs() > 3).sum())
    n_out = int(mask.sum())
    r = _base("outliers", title, "IQR rule (1.5×IQR) and z-score > 3", len(s))
    r.update(effect=n_out / len(s), effect_label="share of rows flagged")
    mean_with, mean_without = s.mean(), s[(s >= lo) & (s <= hi)].mean()
    shift = (mean_with - mean_without) / mean_without * 100 if mean_without else 0
    r["stats"] = {"lower_fence": lo, "upper_fence": hi, "n_outliers": n_out, "pct_outliers": n_out / len(s) * 100,
                  "extreme_z3": extreme, "mean_with": mean_with, "mean_without": mean_without, "mean_shift_pct": shift}
    r["finding"] = (f"{n_out} of {len(s)} rows ({n_out/len(s):.1%}) are outliers in {target} (outside {_fmt(lo)}–{_fmt(hi)}); "
                    f"{extreme} are extreme (|z| > 3). Removing them changes the average by {shift:+.1f}%.")
    if abs(shift) > 10:
        r["caveats"].append("Outliers materially shift the average — conclusions based on means may be distorted.")
    r["table"] = d[mask].sort_values(target, ascending=False).head(15)
    fig = px.box(d, y=target, points="outliers", color_discrete_sequence=PALETTE, title=f"Outliers in {target}")
    r["fig"] = _style(fig, 340)
    return r


# --------------------------------------------------------------------------- #
# 8. Chi-square association (two categorical variables)
# --------------------------------------------------------------------------- #
def chi_square(df, a, b, focus=None):
    title = f"Association between {a} and {b}"
    d = apply_focus(df, focus)[[a, b]].dropna()
    ct = pd.crosstab(d[a], d[b])
    if ct.shape[0] < 2 or ct.shape[1] < 2:
        return _fail("chi_square", title, "Need at least 2 categories in each variable.")
    chi2, p, dof, exp = stats.chi2_contingency(ct)
    v = np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))
    r = _base("chi_square", title, "Chi-square test of independence (Cramér's V effect size)", len(d))
    r.update(p_value=float(p), effect=float(v), effect_label="Cramér's V", significant=bool(p < ALPHA))
    r["stats"] = {"chi2": chi2, "dof": dof, "p_value": p, "cramers_v": v}
    r["finding"] = (f"{a} and {b} are {'significantly associated' if p < ALPHA else 'not significantly associated'} "
                    f"(χ² = {chi2:.1f}, p = {_p(p)}, Cramér's V = {v:.2f}).")
    if (exp < 5).mean() > 0.2:
        r["caveats"].append("Over 20% of expected cell counts are below 5 — chi-square may be unreliable.")
    r["table"] = ct.reset_index()
    fig = px.imshow(ct, text_auto=True, color_continuous_scale="Blues", aspect="auto", title=f"{a} × {b} counts")
    r["fig"] = _style(fig)
    return r


TOOLS = {
    "descriptive": descriptive, "trend": trend, "correlation": correlation,
    "correlation_matrix": correlation_matrix, "regression": regression,
    "group_compare": group_compare, "segment_vs_rest": segment_vs_rest,
    "outliers": outliers, "chi_square": chi_square,
}

TOOL_DESCRIPTIONS = {
    "descriptive": "Summary statistics of a measure (optionally by group). args: target, group?",
    "trend": "Is a measure rising or falling over time? args: target, date_col",
    "correlation": "Linear relationship between two numeric columns. args: x, y",
    "correlation_matrix": "Heatmap of all numeric relationships. args: cols",
    "regression": "Which variables drive the target, holding others constant. args: target, predictors, categorical",
    "group_compare": "t-test (2 groups) or ANOVA (3+) of a measure across a category. args: target, group",
    "segment_vs_rest": "Compare the focus segment (e.g. Product A) with all others. args: target, col, value",
    "outliers": "Unusual values that may distort averages. args: target",
    "chi_square": "Association between two categorical columns. args: a, b",
}
