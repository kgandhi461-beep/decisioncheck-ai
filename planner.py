"""
The DecisionCheck agent's "brain": understand the decision, plan analyses,
and adapt the plan after observing results (follow-up steps).

Two interpreters exist:
  * AI interpreter  - asks the LLM to map the decision onto the dataset's columns
  * Rule interpreter - keyword & fuzzy matching (works with no API key)
Whatever the AI proposes is VALIDATED against the real columns, so a
hallucinated column name can never reach the analytics layer.
"""
from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from data_quality import Schema

# words that hint at outcomes (things managers want to move) vs levers (things they control)
OUTCOME_WORDS = ["sales", "revenue", "profit", "margin", "income", "turnover", "gmv", "amount", "orders",
                 "conversion", "retention", "churn", "satisfaction", "rating", "nps", "units", "quantity",
                 "qty", "volume", "demand", "footfall", "visits", "leads", "return", "roi", "score", "cost", "expense"]
LEVER_WORDS = ["discount", "price", "promo", "promotion", "markdown", "offer", "coupon", "marketing", "ad", "ads",
               "advert", "advertising", "spend", "budget", "campaign", "staff", "headcount", "training",
               "inventory", "stock", "shipping", "delivery", "commission", "incentive", "rate", "fee", "hours"]
SYNONYMS = {
    "sales": ["sales", "revenue", "turnover", "amount", "gmv", "income"],
    "revenue": ["revenue", "sales", "turnover", "amount"],
    "profit": ["profit", "margin", "earnings", "net"],
    "discount": ["discount", "markdown", "promo", "offer", "coupon"],
    "price": ["price", "mrp", "rate", "unit_price", "cost per"],
    "quantity": ["quantity", "qty", "units", "volume", "items"],
    "marketing": ["marketing", "ad", "ads", "advertising", "spend", "campaign"],
    "customers": ["customer", "customers", "clients", "buyers"],
    "satisfaction": ["satisfaction", "csat", "rating", "nps", "score"],
}
UP = r"\b(increase|raise|boost|grow|expand|add|more|higher|extend|launch|introduce|invest|double|scale up|hike|step up)\b"
DOWN = r"\b(decrease|reduce|cut|lower|less|drop|stop|close|remove|discontinue|shut|scale down|withdraw|trim|eliminate|end)\b"
DECLINE = r"\b(declin\w*|falling|fall\w*|drop\w*|decreas\w*|down|shrink\w*|slow\w*|weak\w*|poor\w*|low\w*|underperform\w*|losing)\b"
RISE = r"\b(ris\w*|grow\w*|increas\w*|up|strong\w*|high\w*|outperform\w*|improv\w*)\b"


@dataclass
class DecisionFrame:
    decision: str
    action_direction: int = 0              # +1 increase lever, -1 decrease, 0 unclear
    lever: str | None = None               # column the manager will change
    lever_kind: str | None = None          # numeric / categorical
    lever_value: str | None = None         # for categorical levers (e.g. Channel=Online)
    target: str | None = None              # main outcome measure
    secondary_outcomes: list = field(default_factory=list)
    focus: dict = field(default_factory=dict)  # {"Product": "Product A"}
    premise_metric: str | None = None
    premise_direction: int = 0             # -1 "declining", +1 "rising"
    date_col: str | None = None
    group_cols: list = field(default_factory=list)
    interpreter: str = "rules"
    notes: list = field(default_factory=list)
    unanswerable: list = field(default_factory=list)

    def to_dict(self):
        return {k: v for k, v in self.__dict__.items()}


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(s).lower()).strip()


def _col_score(col: str, text: str) -> float:
    """How strongly does the decision text refer to this column?"""
    c = _norm(col)
    t = f" {_norm(text)} "
    if not c:
        return 0
    if f" {c} " in t or (len(c) > 3 and f" {c}s " in t) or (c.endswith("s") and f" {c[:-1]} " in t):
        return 1.0
    words = [w for w in c.split() if len(w) > 2]
    best = 0.0
    for key, syns in SYNONYMS.items():
        if any(s in c for s in syns) and re.search(rf"\b{key}\w*\b", t):
            best = max(best, 0.8)
        if re.search(rf"\b{key}\w*\b", c) and any(re.search(rf"\b{re.escape(s)}\w*\b", t) for s in syns):
            best = max(best, 0.75)
    for w in words:
        if re.search(rf"\b{re.escape(w)}\w*\b", t):
            best = max(best, 0.7)
        else:
            for tok in t.split():
                if len(tok) > 3 and difflib.SequenceMatcher(None, w, tok).ratio() > 0.85:
                    best = max(best, 0.6)
    return best


def _is_kind(col: str, words: list[str]) -> bool:
    c = _norm(col)
    return any(re.search(rf"\b{w}", c) for w in words)


def _find_values(df: pd.DataFrame, sch: Schema, text: str) -> dict:
    """Find categorical values (e.g. 'Product A', 'North') mentioned in the text."""
    t = f" {_norm(text)} "
    found = {}
    for col in sch.categorical:
        vals = df[col].dropna().astype(str).unique()
        if len(vals) > 500:
            continue
        best, best_len = None, 0
        for v in vals:
            nv = _norm(v)
            if len(nv) < 2 or nv in ("yes", "no", "true", "false", "0", "1"):
                continue
            if f" {nv} " in t and len(nv) > best_len:
                best, best_len = v, len(nv)
        if best is not None:
            found[col] = best
    # keep the most specific matches; avoid a value matching inside a longer matched value
    return found


# --------------------------------------------------------------------------- #
# Rule-based interpreter
# --------------------------------------------------------------------------- #
def interpret_rules(decision: str, df: pd.DataFrame, sch: Schema) -> DecisionFrame:
    fr = DecisionFrame(decision=decision, interpreter="rules")
    text = decision.lower()
    nums = sch.numeric
    fr.date_col = sch.datetime[0] if sch.datetime else None

    # split "<action part> because <reason part>"
    m = re.split(r"\b(because|since|as|due to|given that|considering)\b", text, maxsplit=1)
    action_txt, reason_txt = m[0], (m[2] if len(m) > 2 else "")
    # split off a goal phrase: "... to increase sales" -> goal names the outcome, not the lever
    g = re.search(r"\b(?:to|so that|in order to)\s+(?:increase|grow|improve|boost|raise|lift|maximi[sz]e|reduce|lower|cut|"
                  r"minimi[sz]e|decrease|get more|drive)\s+(.*)$", action_txt)
    goal_txt = g.group(1) if g else ""
    if g and g.start() > 0:
        action_txt = action_txt[:g.start()]

    # action direction
    up, down = re.search(UP, action_txt), re.search(DOWN, action_txt)
    if up and (not down or up.start() < down.start()):
        fr.action_direction = 1
    elif down:
        fr.action_direction = -1

    # numeric column mentions
    scores = {c: max(_col_score(c, action_txt), 0.9 * _col_score(c, text)) for c in nums}
    mentioned = [c for c, s in sorted(scores.items(), key=lambda kv: -kv[1]) if s >= 0.6]

    # lever = mentioned numeric column in action part that looks like a lever (or isn't an outcome)
    act_scores = {c: _col_score(c, action_txt) for c in nums}
    lever_cands = [c for c in sorted(act_scores, key=lambda c: -act_scores[c]) if act_scores[c] >= 0.6]
    lever_cands.sort(key=lambda c: (not _is_kind(c, LEVER_WORDS), -act_scores[c]))
    if lever_cands and (_is_kind(lever_cands[0], LEVER_WORDS) or len(lever_cands) > 1 or not reason_txt):
        fr.lever, fr.lever_kind = lever_cands[0], "numeric"

    # target = mentioned outcome column (prefer reason part), else best outcome-like column
    def _best(part):
        sc = {c: _col_score(c, part) for c in nums if c != fr.lever} if part else {}
        return [c for c in sorted(sc, key=lambda c: (-sc[c], not _is_kind(c, OUTCOME_WORDS))) if sc[c] >= 0.6]

    reason_tgt = _best(reason_txt)
    tgt = _best(goal_txt) or reason_tgt
    if not tgt:
        tgt = [c for c in mentioned if c != fr.lever and _is_kind(c, OUTCOME_WORDS)] or \
              [c for c in mentioned if c != fr.lever]
    if not tgt:
        outcome_like = [c for c in nums if c != fr.lever and _is_kind(c, OUTCOME_WORDS)]
        pref = sorted(outcome_like, key=lambda c: (not _is_kind(c, ["sales", "revenue"]), not _is_kind(c, ["profit"])))
        tgt = pref or [c for c in nums if c != fr.lever]
        if tgt:
            fr.notes.append(f"No outcome measure named in the question; using '{tgt[0]}' as the main outcome.")
    fr.target = tgt[0] if tgt else None

    # categorical values mentioned -> focus segment or categorical lever
    vals = _find_values(df, sch, decision)
    for col, val in vals.items():
        in_action = _norm(val) in _norm(action_txt)
        if fr.lever is None and in_action and re.search(r"\b(switch|move|shift|expand|open|launch|enter|focus|invest|close|exit|stop|drop|discontinue)\b", action_txt):
            fr.lever, fr.lever_kind, fr.lever_value = col, "categorical", val
            if not fr.action_direction:
                fr.action_direction = -1 if re.search(r"\b(close|exit|stop|drop|discontinue)\b", action_txt) else 1
        else:
            fr.focus[col] = val
    # categorical column named as lever ("change the channel")
    if fr.lever is None:
        cat_scores = {c: _col_score(c, action_txt) for c in sch.categorical}
        best = max(cat_scores, key=cat_scores.get) if cat_scores else None
        if best and cat_scores[best] >= 0.7 and best not in fr.focus:
            fr.lever, fr.lever_kind = best, "categorical"

    # premise ("because sales are declining")
    if reason_txt:
        if re.search(DECLINE, reason_txt):
            fr.premise_direction = -1
        elif re.search(RISE, reason_txt):
            fr.premise_direction = 1
        if fr.premise_direction:
            fr.premise_metric = (reason_tgt[0] if reason_tgt else fr.target)
    elif re.search(r"\b(declin\w*|falling|dropping)\b", text):
        fr.premise_direction, fr.premise_metric = -1, fr.target

    # secondary outcomes (side effects such as profit/margin)
    fr.secondary_outcomes = [c for c in nums if c not in (fr.target, fr.lever) and _is_kind(c, OUTCOME_WORDS)][:3]

    # grouping columns for context
    fr.group_cols = _pick_groups(df, sch, decision, exclude=list(fr.focus) + ([fr.lever] if fr.lever_kind == "categorical" else []))
    _check_answerable(fr, decision, df, sch)
    return fr


def _pick_groups(df, sch, text, exclude):
    cands = [c for c in sch.categorical if c not in exclude and 2 <= df[c].nunique() <= 15]
    cands.sort(key=lambda c: (-_col_score(c, text), df[c].nunique()))
    return cands[:2]


def _check_answerable(fr: DecisionFrame, decision: str, df, sch):
    """Detect concepts the user mentioned that do not exist in the data."""
    t = _norm(decision)
    concepts = {"profit": ["profit", "margin"], "price": ["price"], "discount": ["discount", "markdown", "promo"],
                "marketing spend": ["marketing", "advertis", "ad spend", "campaign"], "competitor": ["competitor"],
                "customer satisfaction": ["satisfaction", "csat", "nps"], "cost": ["cost"], "inventory": ["inventory", "stock"],
                "time / dates": ["month", "trend", "over time", "declin", "growth", "season", "year"]}
    cols = " ".join(_norm(c) for c in df.columns)
    for concept, keys in concepts.items():
        if any(k in t for k in keys):
            if concept == "time / dates":
                if not sch.datetime:
                    fr.unanswerable.append("The question refers to change over time, but the dataset has no date column — trend claims cannot be tested.")
                continue
            if not any(k in cols for k in keys):
                fr.unanswerable.append(f"The question mentions {concept}, but no matching column exists in the data.")
    if fr.target is None:
        fr.unanswerable.append("No numeric outcome measure could be identified.")


# --------------------------------------------------------------------------- #
# AI interpreter (validated)
# --------------------------------------------------------------------------- #
INTERPRET_SYSTEM = """You are the planning module of DecisionCheck AI, a business-analytics agent.
Map a manager's proposed decision onto the columns of their dataset. Use ONLY column names and values that appear in the schema.
Return strict JSON with keys:
 target (numeric column the decision aims to improve), lever (column the manager would change, or null),
 action_direction (1 increase, -1 decrease, 0 unclear), lever_value (category value for categorical levers, or null),
 focus (object {column: value} for the segment the decision is about, e.g. a product or region, or {}),
 premise_metric (column the manager's reasoning refers to, or null), premise_direction (-1 declining, 1 rising, 0 none),
 secondary_outcomes (list of numeric columns that could be harmed, e.g. profit), group_cols (up to 2 categorical columns useful for comparison),
 unanswerable (list of short sentences naming things the question needs that the data does not contain),
 rationale (one sentence)."""


def describe_schema(df: pd.DataFrame, sch: Schema) -> str:
    lines = []
    for c in sch.numeric:
        s = df[c]
        lines.append(f"- {c} (numeric; min {s.min():.4g}, mean {s.mean():.4g}, max {s.max():.4g})")
    for c in sch.categorical:
        vals = df[c].dropna().astype(str).value_counts().index[:12].tolist()
        lines.append(f"- {c} (categorical; {df[c].nunique()} values e.g. {vals})")
    for c in sch.datetime:
        lines.append(f"- {c} (date; {str(df[c].min())[:10]} to {str(df[c].max())[:10]})")
    for c in sch.identifier:
        lines.append(f"- {c} (identifier, ignore)")
    return "\n".join(lines)


def interpret_ai(decision, df, sch, llm) -> DecisionFrame | None:
    prompt = f"Dataset schema ({len(df)} rows):\n{describe_schema(df, sch)}\n\nProposed decision:\n\"{decision}\"\n\nReturn the JSON."
    out = llm.generate_json(prompt, INTERPRET_SYSTEM)
    if not out:
        return None
    rules = interpret_rules(decision, df, sch)
    fr = DecisionFrame(decision=decision, interpreter="ai", date_col=rules.date_col)
    cols = set(df.columns)

    def ok(c, pool):
        return c if isinstance(c, str) and c in pool else None

    fr.target = ok(out.get("target"), sch.numeric) or rules.target
    lever = out.get("lever")
    if lever in sch.numeric and lever != fr.target:
        fr.lever, fr.lever_kind = lever, "numeric"
    elif lever in sch.categorical:
        fr.lever, fr.lever_kind = lever, "categorical"
        lv = out.get("lever_value")
        if lv is not None and str(lv) in df[lever].astype(str).unique():
            fr.lever_value = str(lv)
    elif lever:
        fr.notes.append(f"AI suggested lever '{lever}' which is not a usable column — ignored.")
    try:
        fr.action_direction = int(np.sign(float(out.get("action_direction") or 0)))
    except (TypeError, ValueError):
        fr.action_direction = rules.action_direction
    for c, v in (out.get("focus") or {}).items():
        if c in sch.categorical and str(v) in df[c].astype(str).unique():
            fr.focus[c] = str(v)
        else:
            fr.notes.append(f"AI focus '{c} = {v}' not found in data — ignored.")
    fr.premise_metric = ok(out.get("premise_metric"), sch.numeric)
    try:
        fr.premise_direction = int(np.sign(float(out.get("premise_direction") or 0)))
    except (TypeError, ValueError):
        fr.premise_direction = 0
    if fr.premise_direction and not fr.premise_metric:
        fr.premise_metric = fr.target
    fr.secondary_outcomes = [c for c in (out.get("secondary_outcomes") or []) if c in sch.numeric and c not in (fr.target, fr.lever)][:3]
    fr.group_cols = [c for c in (out.get("group_cols") or []) if c in sch.categorical and c not in fr.focus and c != fr.lever][:2] \
        or _pick_groups(df, sch, decision, list(fr.focus) + ([fr.lever] if fr.lever_kind == "categorical" else []))
    fr.unanswerable = [str(u) for u in (out.get("unanswerable") or [])][:4]
    if out.get("rationale"):
        fr.notes.append("AI reading: " + str(out["rationale"]))
    _check_answerable_merge(fr, rules)
    return fr


def _check_answerable_merge(fr, rules):
    for u in rules.unanswerable:
        if u not in fr.unanswerable:
            fr.unanswerable.append(u)


def interpret(decision, df, sch, llm=None) -> DecisionFrame:
    if llm is not None and llm.enabled:
        fr = interpret_ai(decision, df, sch, llm)
        if fr is not None and fr.target:
            return fr
    fr = interpret_rules(decision, df, sch)
    if llm is not None and llm.enabled:
        fr.notes.append("AI interpretation unavailable — used rule-based interpreter.")
    return fr


# --------------------------------------------------------------------------- #
# Planning
# --------------------------------------------------------------------------- #
def _predictors(df, sch, fr: DecisionFrame):
    """Choose regression predictors, avoiding leakage (columns that mechanically make up the target)."""
    preds = []
    for c in sch.numeric:
        if c == fr.target:
            continue
        # Other outcomes (quantity, profit, ratings...) are results or mechanical components of the
        # target, not causes - controlling for them would hide the lever's real effect.
        if c != fr.lever and _is_kind(c, OUTCOME_WORDS):
            continue
        r = df[[c, fr.target]].dropna().corr().iloc[0, 1] if df[[c, fr.target]].dropna().shape[0] > 5 else 0
        if abs(r) > 0.95 and c != fr.lever:
            continue
        preds.append(c)
    preds = sorted(preds, key=lambda c: c != fr.lever)[:6]
    # control for the focus column (e.g. Product) so the lever effect is not confounded by segment mix
    cats = [c for c in list(fr.focus) if df[c].nunique() <= 15] + [c for c in fr.group_cols if c not in fr.focus]
    cats = cats[:3]
    return preds + cats, cats


def build_plan(fr: DecisionFrame, df: pd.DataFrame, sch: Schema) -> list[dict]:
    """Return an ordered list of analysis steps with the reason for each."""
    plan = []
    T, F = fr.target, fr.focus or None
    if T is None:
        return plan

    def add(tool, args, purpose, role):
        plan.append({"tool": tool, "args": args, "purpose": purpose, "role": role, "status": "planned"})

    add("descriptive", {"target": T, "focus": F}, f"Understand the size and spread of {T}" + (f" for {F}" if F else ""), "context")

    PF = F or ({fr.lever: fr.lever_value} if fr.lever_kind == "categorical" and fr.lever_value else None)
    if fr.premise_direction and fr.premise_metric and fr.date_col:
        add("trend", {"target": fr.premise_metric, "date_col": fr.date_col, "focus": PF},
            f"Verify the premise that {fr.premise_metric} is {'declining' if fr.premise_direction < 0 else 'rising'}"
            + (f" for {list(PF.values())[0]}" if PF else ""), "premise")
    elif fr.date_col:
        add("trend", {"target": T, "date_col": fr.date_col, "focus": F}, f"Check how {T} has moved over time", "context")

    for col, val in (fr.focus or {}).items():
        add("segment_vs_rest", {"target": T, "col": col, "value": val}, f"Compare {val} with other {col} values", "segment")

    if fr.lever and fr.lever_kind == "numeric":
        add("correlation", {"x": fr.lever, "y": T, "focus": F},
            f"Does {T} move with {fr.lever}" + (f" within {list(F.values())[0]}" if F else "") + "?", "lever")
        preds, cats = _predictors(df, sch, fr)
        if len(preds) >= 1:
            add("regression", {"target": T, "predictors": preds, "categorical": cats, "focus": None},
                f"Isolate the effect of {fr.lever} on {T} while controlling for {', '.join([p for p in preds if p != fr.lever][:4]) or 'nothing else'}", "lever")
        for so in fr.secondary_outcomes[:2]:
            add("correlation", {"x": fr.lever, "y": so, "focus": F}, f"Check for side-effects of changing {fr.lever} on {so}", "side_effect")
    elif fr.lever and fr.lever_kind == "categorical":
        add("group_compare", {"target": T, "group": fr.lever, "focus": F},
            f"Does {T} differ across {fr.lever}" + (f" (is {fr.lever_value} better?)" if fr.lever_value else "") + "?", "lever")
        if fr.lever_value:
            add("segment_vs_rest", {"target": T, "col": fr.lever, "value": fr.lever_value},
                f"Compare {fr.lever_value} with other {fr.lever} options", "lever")
    else:
        # no lever identified - explore drivers of the target
        others = [c for c in sch.numeric if c != T][:4]
        for o in others[:2]:
            add("correlation", {"x": o, "y": T, "focus": F}, f"Explore whether {o} relates to {T}", "context")

    for g in fr.group_cols[:2]:
        add("group_compare", {"target": T, "group": g, "focus": None}, f"Check whether {T} differs by {g}", "context")

    add("outliers", {"target": T, "focus": F}, f"Find unusual {T} values that could distort conclusions", "robustness")
    if len(sch.numeric) >= 3:
        add("correlation_matrix", {"cols": sch.numeric[:10], "focus": None}, "See all numeric relationships at a glance", "context")
    return plan


# --------------------------------------------------------------------------- #
# Adaptive follow-ups (observe -> decide next action)
# --------------------------------------------------------------------------- #
def follow_ups(fr: DecisionFrame, results: list[dict], df, sch) -> list[dict]:
    steps = []
    by_role = {}
    for r in results:
        by_role.setdefault(r.get("role"), []).append(r)

    # 1. Premise confirmed for a segment -> is it market-wide?
    for r in by_role.get("premise", []):
        if r["ok"] and r["significant"] and (fr.focus or fr.lever_value) and fr.date_col:
            steps.append({"tool": "trend", "args": {"target": fr.premise_metric, "date_col": fr.date_col, "focus": None},
                          "purpose": "Premise confirmed for the segment — check whether the whole business shows the same pattern (market-wide vs segment-specific)",
                          "role": "market", "status": "follow-up"})

    # 2. Outliers distort the mean -> re-test lever relationship without them
    out = next((r for r in results if r["tool"] == "outliers" and r["ok"]), None)
    lever_corr = next((r for r in by_role.get("lever", []) if r["tool"] == "correlation" and r["ok"]), None)
    if out and lever_corr and abs(out["stats"].get("mean_shift_pct", 0)) > 5:
        steps.append({"tool": "correlation_trimmed", "args": {"x": fr.lever, "y": fr.target, "focus": fr.focus or None,
                                                              "lo": out["stats"]["lower_fence"], "hi": out["stats"]["upper_fence"]},
                      "purpose": "Outliers shift the average — re-test the lever relationship with outliers removed",
                      "role": "robustness_lever", "status": "follow-up"})

    # 3. Lever correlated in segment but sample is small -> check across all data
    if lever_corr and fr.focus and lever_corr["n"] < 60:
        steps.append({"tool": "correlation", "args": {"x": fr.lever, "y": fr.target, "focus": None},
                      "purpose": "Segment sample is small — check the lever relationship across all data for a more stable estimate",
                      "role": "lever_all", "status": "follow-up"})

    # 4. Lever looks significant in correlation but not in regression -> possible confounding (noted by evidence engine)
    # 5. Categorical lever significant -> nothing further
    return steps


# --------------------------------------------------------------------------- #
# Example decisions generated from ANY dataset (shown as one-click suggestions)
# --------------------------------------------------------------------------- #
def suggest_decisions(df: pd.DataFrame, sch: Schema, k: int = 4) -> list[str]:
    nums = sch.numeric
    outcomes = [c for c in nums if _is_kind(c, OUTCOME_WORDS)]
    outcomes.sort(key=lambda c: (not _is_kind(c, ["sales", "revenue", "performance", "profit"]),))
    levers = [c for c in nums if _is_kind(c, LEVER_WORDS) and c not in outcomes[:1]] or \
             [c for c in nums if c not in outcomes[:1]]
    prio = ["discount", "promo", "marketing", "training", "advert", "spend", "budget", "campaign", "staff", "price"]
    levers.sort(key=lambda c: next((i for i, w in enumerate(prio) if w in c.lower()), len(prio)))
    target = outcomes[0] if outcomes else (nums[0] if nums else None)
    if not target:
        return []
    nice = lambda c: c.replace("_", " ").lower()  # noqa: E731
    out = []
    seg_col = next((c for c in sch.categorical if 2 < df[c].nunique() <= 20), None)
    if levers and seg_col:
        val = df[seg_col].value_counts().index[0]
        out.append(f"Should we increase {nice(levers[0])} for {val} because its {nice(target)} is declining?" if sch.datetime
                   else f"Should we increase {nice(levers[0])} for {val} to improve {nice(target)}?")
    for lv in levers[:2]:
        out.append(f"Should we increase {nice(lv)} to grow {nice(target)}?")
    if seg_col:
        best = df.groupby(seg_col)[target].mean().idxmax()
        out.append(f"Should we focus investment on {best} to increase {nice(target)}?")
    two = next((c for c in sch.categorical if df[c].nunique() == 2), None)
    if two and len(out) < k:
        out.append(f"Does {nice(two)} make a difference to {nice(target)}, and should we change it?")
    return list(dict.fromkeys(out))[:k]
