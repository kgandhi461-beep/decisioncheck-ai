"""
DecisionCheck agent orchestrator.

    PERCEIVE  -> profile data & check quality
    INTERPRET -> understand the decision (AI or rules, validated)
    PLAN      -> choose analyses that fit the question and the columns
    ACT       -> run deterministic statistical tools
    OBSERVE   -> look at results and add follow-up analyses where needed
    EVALUATE  -> sort evidence into supports / challenges / limitations
    EXPLAIN   -> Generative AI writes the business explanation (grounded)
    VERIFY    -> check every number in the AI text exists in computed results
    HAND OFF  -> human makes and records the final decision
"""
from __future__ import annotations

import json
import re
import time

import numpy as np
import pandas as pd
import plotly.express as px

import analysis as A
from evidence import assess
from planner import build_plan, follow_ups, interpret


class Trace:
    """Human-readable log of what the agent did and why (shown in the UI)."""

    def __init__(self):
        self.steps = []
        self.t0 = time.time()

    def log(self, phase, message, detail=None):
        self.steps.append({"t": round(time.time() - self.t0, 2), "phase": phase, "message": message, "detail": detail})


def correlation_trimmed(df, x, y, focus=None, lo=None, hi=None):
    d = A.apply_focus(df, focus)
    d = d[(d[y] >= lo) & (d[y] <= hi)] if lo is not None else d
    r = A.correlation(d, x, y)
    r["tool"] = "correlation_trimmed"
    r["title"] = f"Robustness: {x} vs {y} without outliers" + (f" ({A.focus_label(focus)})" if focus else "")
    return r


TOOLS = dict(A.TOOLS, correlation_trimmed=correlation_trimmed)


def run_step(df, step):
    fn = TOOLS[step["tool"]]
    try:
        res = fn(df, **step["args"])
    except Exception as e:  # noqa: BLE001
        res = A._fail(step["tool"], step["purpose"], f"Analysis failed: {e}")
    res["role"] = step["role"]
    res["purpose"] = step["purpose"]
    res["step_status"] = step["status"]
    return res


# --------------------------------------------------------------------------- #
# Explanation
# --------------------------------------------------------------------------- #
EXPLAIN_SYSTEM = """You are the explanation module of DecisionCheck AI, a decision-support tool for business managers.
You receive a proposed decision and statistical evidence that was ALREADY CALCULATED by Python.
Rules:
1. Use ONLY numbers that appear in the evidence. Never invent, estimate or recompute figures.
2. Write for a non-technical manager: plain English, short sentences, explain what p-values/correlations mean in business terms.
3. Be balanced: present supporting AND challenging evidence. Correlation is not causation.
4. Do NOT make the decision. End by handing the decision back to the human.
Format in Markdown with exactly these headings:
### Bottom line
(2-3 sentences)
### What supports the decision
### What challenges the decision
### What the data cannot tell us
### Questions to consider before deciding
### Suggested next steps
(e.g. a pilot test, extra data to collect)"""


def _evidence_payload(fr, results, ev, quality):
    return {
        "decision": fr.decision,
        "interpretation": {"target": fr.target, "lever": fr.lever, "action": "increase" if fr.action_direction > 0 else "decrease" if fr.action_direction < 0 else "unclear",
                           "focus": fr.focus, "premise": f"{fr.premise_metric} {'declining' if fr.premise_direction < 0 else 'rising'}" if fr.premise_direction else None},
        "data_quality_score": quality["score"], "rows": quality["rows"],
        "overall_balance": ev["verdict"], "analysis_confidence": f"{ev['confidence_label']} ({ev['confidence']}/100)",
        "evidence": [{"stance": e["stance"], "strength": e["strength"], "headline": e["headline"], "detail": e["detail"]} for e in ev["items"]],
        "caveats": sorted({c for r in results for c in r.get("caveats", [])}),
    }


def template_explanation(fr, results, ev):
    sup = [e for e in ev["items"] if e["stance"] == "supports"]
    cha = [e for e in ev["items"] if e["stance"] == "challenges"]
    lim = [e for e in ev["items"] if e["stance"] == "limitation"]
    ctx = [e for e in ev["items"] if e["stance"] == "context"]
    out = ["### Bottom line",
           f"**{ev['verdict']}.** {len(sup)} finding(s) support and {len(cha)} challenge the proposal "
           f"\"{fr.decision.strip()}\". Confidence in this analysis is **{ev['confidence_label'].lower()}** ({ev['confidence']}/100). "
           "This is evidence to inform your judgement — the decision remains yours.", ""]
    out.append("### What supports the decision")
    out += [f"- **{e['headline']}** — {e['detail']}" for e in sup] or ["- No statistically supported evidence in favour was found."]
    if ctx:
        out += [f"- *Context:* {e['detail']}" for e in ctx[:2]]
    out += ["", "### What challenges the decision"]
    out += [f"- **{e['headline']}** — {e['detail']}" for e in cha] or ["- No statistically significant evidence against was found."]
    out += ["", "### What the data cannot tell us"]
    out += [f"- {e['detail']}" for e in lim[:6]]
    q = []
    if fr.lever:
        q.append(f"What else changed at the same time as {fr.lever} (seasonality, competitors, stock-outs, marketing)?")
    if any(e["category"] == "side_effect" for e in cha):
        q.append("Is the gain in the main outcome worth the cost in the side-effect metric?")
    if fr.premise_direction:
        q.append(f"Is the change in {fr.premise_metric} large enough to matter commercially, not just statistically?")
    q.append("What would you expect to see within 4–8 weeks if the decision is working, and what would make you reverse it?")
    out += ["", "### Questions to consider before deciding"] + [f"- {x}" for x in q]
    out += ["", "### Suggested next steps",
            "- Run a small, time-limited pilot (e.g. one region or store) with a comparison group before a full rollout.",
            "- Collect any missing data flagged above.",
            "- Record your final decision and reasoning below so it can be reviewed later."]
    return "\n".join(out)


def _numbers(text):
    vals = []
    for m in re.finditer(r"(?<![A-Za-z])[-+]?\d[\d,]*\.?\d*\s*%?", text):
        s = m.group(0).replace(",", "").replace("%", "").strip()
        try:
            vals.append(float(s))
        except ValueError:
            pass
    return vals


def grounding_check(ai_text: str, payload: dict, results: list[dict]) -> dict:
    """Flag numbers in the AI text that do not appear in the computed evidence (hallucination guard)."""
    source = json.dumps(payload, default=str)
    for r in results:
        source += " " + json.dumps({k: v for k, v in r.get("stats", {}).items()}, default=str)
    pool = _numbers(source)
    unverified = []
    for v in set(_numbers(ai_text)):
        if abs(v) < 10 and float(v).is_integer():
            continue  # small integers (list numbering, "3 groups") are not checked
        ok = any(abs(v - p) <= max(0.011, 0.006 * abs(p)) or abs(v - p * 100) <= 0.6 or abs(v * 100 - p) <= 0.6 for p in pool)
        if not ok:
            unverified.append(v)
    checked = len([v for v in set(_numbers(ai_text)) if not (abs(v) < 10 and float(v).is_integer())])
    return {"checked": checked, "unverified": sorted(unverified)[:10],
            "passed": len(unverified) == 0}


def explain(fr, results, ev, quality, llm):
    payload = _evidence_payload(fr, results, ev, quality)
    if llm is not None and llm.enabled:
        prompt = "Evidence (JSON):\n" + json.dumps(payload, indent=1, default=str) + "\n\nWrite the explanation."
        txt = llm.generate(prompt, EXPLAIN_SYSTEM, temperature=0.3)
        if txt:
            return {"text": txt.strip(), "source": "ai", "model": llm.label,
                    "grounding": grounding_check(txt, payload, results), "payload": payload}
    return {"text": template_explanation(fr, results, ev), "source": "template", "model": "Built-in rule engine",
            "grounding": {"checked": 0, "unverified": [], "passed": True}, "payload": payload,
            "error": getattr(llm, "last_error", None) if llm is not None else None}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_agent(decision, df, sch, quality, llm=None, plan_override=None, frame_override=None, progress=None):
    tr = Trace()
    say = progress or (lambda *_: None)

    tr.log("Perceive", f"Dataset has {quality['rows']:,} rows, {quality['columns']} columns; data-quality score {quality['score']}/100.",
           {"numeric": sch.numeric, "categorical": sch.categorical, "dates": sch.datetime})
    say("Understanding your decision…", 0.1)
    fr = frame_override or interpret(decision, df, sch, llm)
    tr.log("Interpret", f"Interpreted with {'Generative AI' if fr.interpreter == 'ai' else 'rule engine'}: "
                        f"target = {fr.target}, lever = {fr.lever}, direction = {fr.action_direction:+d}, focus = {fr.focus or 'none'}, "
                        f"premise = {fr.premise_metric} {'↓' if fr.premise_direction < 0 else '↑' if fr.premise_direction > 0 else '—'}",
           {"notes": fr.notes, "unanswerable": fr.unanswerable})

    plan = plan_override or build_plan(fr, df, sch)
    tr.log("Plan", f"Selected {len(plan)} analyses suited to this question.", [f"{p['tool']}: {p['purpose']}" for p in plan])

    results = []
    for k, step in enumerate(plan):
        say(f"Running {step['tool'].replace('_', ' ')}…", 0.15 + 0.55 * k / max(len(plan), 1))
        res = run_step(df, step)
        results.append(res)
        tr.log("Act", f"{step['tool']} → {'✓' if res['ok'] else '✗'} {res['finding'][:160]}")

    say("Reviewing results and deciding on follow-ups…", 0.72)
    extra = follow_ups(fr, results, df, sch)
    if extra:
        tr.log("Observe", f"Results triggered {len(extra)} follow-up analysis(es).", [e["purpose"] for e in extra])
        for step in extra:
            res = run_step(df, step)
            results.append(res)
            plan.append(step)
            tr.log("Act", f"[follow-up] {step['tool']} → {res['finding'][:160]}")
    else:
        tr.log("Observe", "No follow-up analysis needed.")

    say("Weighing supporting and challenging evidence…", 0.8)
    ev = assess(fr, results, quality)
    tr.log("Evaluate", f"{sum(e['stance']=='supports' for e in ev['items'])} supporting, "
                       f"{sum(e['stance']=='challenges' for e in ev['items'])} challenging, "
                       f"{sum(e['stance']=='limitation' for e in ev['items'])} limitations → {ev['verdict']}.")

    say("Writing the business explanation…", 0.88)
    ex = explain(fr, results, ev, quality, llm)
    tr.log("Explain", f"Explanation written by {ex['model']}.")
    if ex["source"] == "ai":
        g = ex["grounding"]
        tr.log("Verify", f"Grounding check: {g['checked']} numbers checked, {len(g['unverified'])} not found in computed results.",
               g["unverified"] or None)
    tr.log("Hand-off", "Waiting for the human decision-maker to review and record a decision.")
    say("Done", 1.0)
    return {"frame": fr, "plan": plan, "results": results, "evidence": ev, "explanation": ex,
            "trace": tr.steps, "elapsed": round(time.time() - tr.t0, 2)}
