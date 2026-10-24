"""
"Ask a follow-up question" agent.

For each question the agent decides which analysis tool to run (Generative AI chooses
the tool when a key is configured; otherwise transparent keyword rules do), Python
runs it, and the answer is written ONLY from that tool's computed result.
"""
from __future__ import annotations

import json
import re

import pandas as pd

import analysis as A
from agent import grounding_check
from planner import DecisionFrame, _col_score, _find_values, _predictors

TOOL_HELP = {k: v for k, v in A.TOOL_DESCRIPTIONS.items() if k != "correlation_matrix"}

ROUTER_SYSTEM = """You are the tool-routing module of DecisionCheck AI. A manager asks a follow-up question about their dataset.
Choose ONE analysis tool and its arguments, using ONLY column names and values from the schema.
Tools:
{tools}
Return strict JSON: {{"tool": "<name>", "args": {{...}}, "focus": {{"<categorical column>": "<value>"}} or {{}}}}.
Argument names must match the tool description. For regression, "predictors" is a list of columns and "categorical" lists which of them are categorical."""

ANSWER_SYSTEM = """You are DecisionCheck AI answering a manager's follow-up question.
Answer in 2-5 short sentences of plain business English, using ONLY the numbers in the computed result below.
Do not invent figures. If the result cannot answer the question, say so and suggest what data would help.
Mention one caveat if relevant (e.g. correlation is not causation)."""


def _mentions(q, df, sch):
    nums = sorted([c for c in sch.numeric if _col_score(c, q) >= 0.6], key=lambda c: -_col_score(c, q))
    cats = sorted([c for c in sch.categorical if _col_score(c, q) >= 0.6], key=lambda c: -_col_score(c, q))
    vals = _find_values(df, sch, q)
    return nums, cats, vals


def route_rules(q: str, df, sch, fr: DecisionFrame):
    ql = q.lower()
    nums, cats, vals = _mentions(q, df, sch)
    target = nums[0] if nums else fr.target
    focus = {c: v for c, v in vals.items()} or None
    group = cats[0] if cats else next((c for c in fr.group_cols if c not in (focus or {})), None) or \
        next((c for c in sch.categorical if 2 <= df[c].nunique() <= 15), None)

    if len(cats) >= 2 and not nums:
        return "chi_square", {"a": cats[0], "b": cats[1], "focus": None}
    if re.search(r"\bwhy\b|what drives|reason|cause|explain", ql):
        tmp = DecisionFrame(decision=q, target=target, lever=fr.lever if fr.lever in sch.numeric else None,
                            focus=focus or {}, group_cols=fr.group_cols)
        preds, cc = _predictors(df, sch, tmp)
        if preds:
            return "regression", {"target": target, "predictors": preds, "categorical": cc, "focus": focus}
    if re.search(r"outlier|unusual|anomal|extreme|strange|odd values", ql):
        return "outliers", {"target": target, "focus": focus}
    if re.search(r"trend|over time|month|week|year|season|declin|grow|rising|falling|going down|going up|increase over", ql) and fr.date_col:
        return "trend", {"target": target, "date_col": fr.date_col, "focus": focus}
    if re.search(r"relat|correl|affect|impact|effect|influence|depend|link|associated|drive[sn]?\b", ql):
        others = [n for n in nums if n != target]
        x = others[0] if others else (fr.lever if fr.lever in sch.numeric and fr.lever != target else None)
        if x:
            return "correlation", {"x": x, "y": target, "focus": focus}
        if cats:
            return "group_compare", {"target": target, "group": cats[0], "focus": None}
    if vals and re.search(r"compare|vs|versus|than|differ|better|worse", ql):
        col, val = next(iter(vals.items()))
        return "segment_vs_rest", {"target": target, "col": col, "value": val}
    if group and re.search(r"which|best|worst|highest|lowest|top|bottom|compare|differ|by |per |across|rank|most|least", ql):
        return "group_compare", {"target": target, "group": group, "focus": focus}
    return "descriptive", {"target": target, "group": group if re.search(r"\bby\b|per|each|across", ql) else None, "focus": focus}


def _validate(tool, args, df, sch):
    cols = set(df.columns)
    if tool not in A.TOOLS:
        return False
    for k, v in args.items():
        if k in ("target", "x", "y", "group", "col", "date_col", "a", "b") and v not in cols:
            return False
        if k == "predictors" and (not isinstance(v, list) or not all(p in cols for p in v)):
            return False
    if args.get("value") is not None and args.get("col") in cols:
        if str(args["value"]) not in df[args["col"]].astype(str).unique():
            return False
    return True


def route_ai(q, df, sch, fr, llm):
    from planner import describe_schema
    tools = "\n".join(f"- {k}: {v}" for k, v in TOOL_HELP.items())
    prompt = (f"Schema:\n{describe_schema(df, sch)}\nDate column: {fr.date_col}\n"
              f"Current decision context: target={fr.target}, lever={fr.lever}, focus={fr.focus}\n\nQuestion: {q}")
    out = llm.generate_json(prompt, ROUTER_SYSTEM.format(tools=tools))
    if not out or not isinstance(out, dict):
        return None
    tool, args = out.get("tool"), dict(out.get("args") or {})
    focus = {c: str(v) for c, v in (out.get("focus") or {}).items()
             if c in sch.categorical and str(v) in df[c].astype(str).unique()}
    if tool in ("trend",) and "date_col" not in args:
        args["date_col"] = fr.date_col
    if tool in ("descriptive", "trend", "correlation", "group_compare", "outliers", "regression", "chi_square"):
        args["focus"] = focus or None
    if tool == "regression":
        args["categorical"] = [c for c in args.get("categorical", []) if c in args.get("predictors", [])]
    allowed = {"descriptive": {"target", "group", "focus"}, "trend": {"target", "date_col", "focus"},
               "correlation": {"x", "y", "focus"}, "regression": {"target", "predictors", "categorical", "focus"},
               "group_compare": {"target", "group", "focus"}, "segment_vs_rest": {"target", "col", "value"},
               "outliers": {"target", "focus"}, "chi_square": {"a", "b", "focus"}}
    if tool not in allowed:
        return None
    args = {k: v for k, v in args.items() if k in allowed[tool]}
    return (tool, args) if _validate(tool, args, df, sch) else None


def _rule_text(res):
    txt = res["finding"]
    t = res.get("table")
    if res["tool"] == "group_compare" and t is not None and len(t) > 1:
        g, tgt = t.columns[0], "mean"
        txt += f"\n\n**Ranking by average:** " + " > ".join(f"{r[g]} ({A._fmt(r[tgt])})" for _, r in t.head(6).iterrows())
    cav = sorted(res.get("caveats", []), key=lambda c: "R²" not in c)
    if cav:
        txt += "\n\n" + "  \n".join(f"*Caveat: {c}*" for c in cav[:2])
    return txt


def answer(q: str, df: pd.DataFrame, sch, fr: DecisionFrame, llm=None) -> dict:
    if not q or not q.strip():
        return {"text": "Please type a question.", "result": None, "tool": None, "by": "-"}
    routed, by = None, "rules"
    if llm is not None and llm.enabled:
        routed = route_ai(q, df, sch, fr, llm)
        by = "ai" if routed else "rules"
    if routed is None:
        routed = route_rules(q, df, sch, fr)
    tool, args = routed
    if not args.get("target", True) and tool != "chi_square":
        return {"text": "I couldn't find a numeric measure to analyse in your data.", "result": None, "tool": tool, "by": by}
    try:
        res = A.TOOLS[tool](df, **args)
    except Exception as e:  # noqa: BLE001
        return {"text": f"I tried a {tool.replace('_', ' ')} analysis but it failed: {e}", "result": None, "tool": tool, "by": by}
    text, grounding = _rule_text(res), None
    extra = None
    if tool == "regression" and fr.date_col and re.search(r"\bwhy\b|reason|cause", q.lower()):
        extra = A.trend(df, args["target"], fr.date_col, focus=args.get("focus"))
        if extra["ok"]:
            text = "**First, is it really changing?** " + extra["finding"] + "\n\n**What drives it:** " + text
    if llm is not None and llm.enabled and res["ok"]:
        payload = {"question": q, "analysis": res["title"], "method": res["method"],
                   "finding": res["finding"] + ((" | Trend check: " + extra["finding"]) if extra and extra["ok"] else ""),
                   "stats": {k: v for k, v in res["stats"].items() if not isinstance(v, dict)},
                   "table": res["table"].head(12).to_dict("records") if isinstance(res.get("table"), pd.DataFrame) else None,
                   "caveats": res.get("caveats", [])}
        ai = llm.generate("Computed result (JSON):\n" + json.dumps(payload, default=str) + f"\n\nQuestion: {q}",
                          ANSWER_SYSTEM, temperature=0.2)
        if ai:
            text = ai.strip()
            grounding = grounding_check(text, payload, [res])
            by = "ai"
    return {"text": text, "result": res, "extra": extra, "tool": tool, "args": args, "by": by, "grounding": grounding}
