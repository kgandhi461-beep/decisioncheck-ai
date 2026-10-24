"""
Evidence engine: turn statistical results into evidence that SUPPORTS or
CHALLENGES the proposed decision. Fully rule-based and transparent, so the
same data always produces the same evidence (the AI only explains it).
"""
from __future__ import annotations

import numpy as np

from planner import DecisionFrame

STRENGTH_W = {"strong": 3, "moderate": 2, "weak": 1}


def _strength_from(p, effect_abs, thresholds=(0.1, 0.3)):
    if p is None:
        return "weak"
    if p < 0.01 and effect_abs >= thresholds[1]:
        return "strong"
    if p < 0.05 and effect_abs >= thresholds[0]:
        return "moderate"
    return "weak"


def _item(stance, strength, headline, detail, result_idx, category):
    return {"stance": stance, "strength": strength, "headline": headline, "detail": detail,
            "result_idx": result_idx, "category": category}


def lever_effect(fr: DecisionFrame, results: list[dict]):
    """Best estimate of the lever's effect on the target: (sign, p, source)."""
    reg = next((r for r in results if r["tool"] == "regression" and r["ok"] and r.get("role") == "lever"), None)
    if reg and fr.lever:
        c = reg["stats"]["coefficients"].get(fr.lever)
        if c:
            return np.sign(c["coef"]), c["p"], "regression", c["coef"]
    corr = next((r for r in results if r["tool"] == "correlation" and r["ok"] and r.get("role") == "lever"), None)
    if corr:
        return np.sign(corr["effect"]), corr["p_value"], "correlation", corr["effect"]
    return None


def assess(fr: DecisionFrame, results: list[dict], quality: dict) -> dict:
    ev = []
    act = fr.action_direction or 1
    verb = "increasing" if act > 0 else "decreasing"

    for i, r in enumerate(results):
        if not r["ok"]:
            ev.append(_item("limitation", "weak", f"Could not run: {r['title']}", r["finding"], i, "limitation"))
            continue
        role, tool = r.get("role"), r["tool"]
        p = r.get("p_value")

        # ---- Premise check
        if role == "premise":
            obs = r["direction"]
            if obs == fr.premise_direction and obs != 0:
                ev.append(_item("supports", _strength_from(p, abs(r["stats"].get("r_squared", 0)) ** 0.5),
                                "The stated problem is real", r["finding"], i, "premise"))
            elif obs == 0:
                ev.append(_item("challenges", "moderate", "The stated problem is not statistically confirmed",
                                r["finding"], i, "premise"))
            else:
                ev.append(_item("challenges", "strong", "Data shows the OPPOSITE of the stated premise",
                                r["finding"], i, "premise"))

        # ---- Market-wide check
        elif role == "market":
            prem = next((x for x in results if x.get("role") == "premise" and x["ok"]), None)
            if r["significant"] and prem and r["direction"] == prem["direction"]:
                seg_rate, all_rate = prem["effect"], r["effect"]
                if abs(all_rate) >= 0.6 * abs(seg_rate):
                    ev.append(_item("challenges", "moderate", "The pattern is business-wide, not specific to the segment",
                                    r["finding"] + " A segment-specific action may not fix a market-wide cause.", i, "market"))
                else:
                    ev.append(_item("supports", "weak", "The segment is doing worse than the business overall",
                                    r["finding"], i, "market"))
            else:
                ev.append(_item("supports", "moderate", "The pattern is specific to the segment",
                                r["finding"] + " The overall business does not show the same pattern, so a targeted action is plausible.", i, "market"))

        # ---- Lever effect
        elif role in ("lever", "lever_all", "robustness_lever") and tool in ("correlation", "correlation_trimmed"):
            eff = r["effect"]
            label = {"lever": "Relationship", "lever_all": "Across all data, relationship",
                     "robustness_lever": "Without outliers, relationship"}[role]
            if r["significant"]:
                helpful = np.sign(eff) * act > 0
                ev.append(_item("supports" if helpful else "challenges",
                                _strength_from(p, abs(eff)),
                                f"{label}: {verb} {fr.lever} is associated with {'better' if helpful else 'worse'} {fr.target}",
                                r["finding"], i, "lever"))
            else:
                ev.append(_item("challenges", "moderate" if r["n"] >= 30 else "weak",
                                f"{label}: no evidence that {fr.lever} moves {fr.target}", r["finding"], i, "lever"))

        elif role == "lever" and tool == "regression":
            c = r["stats"]["coefficients"].get(fr.lever)
            if c is None:
                continue
            if c["p"] < 0.05:
                helpful = np.sign(c["coef"]) * act > 0
                ev.append(_item("supports" if helpful else "challenges", "strong" if c["p"] < 0.01 else "moderate",
                                f"After controlling for other factors, {fr.lever} {'helps' if helpful else 'hurts'} {fr.target} in the proposed direction"
                                if helpful else f"After controlling for other factors, {verb} {fr.lever} is linked to worse {fr.target}",
                                f"Each 1-unit change in {fr.lever} is associated with a {c['coef']:+,.3g} change in {fr.target} "
                                f"(p = {c['p']:.3g}). {r['finding']}", i, "lever"))
            else:
                ev.append(_item("challenges", "moderate",
                                f"After controlling for other factors, {fr.lever} has no significant effect on {fr.target}",
                                f"Coefficient {c['coef']:+,.3g}, p = {c['p']:.3g}. {r['finding']}", i, "lever"))
            # confounding signal
            corr = next((x for x in results if x["tool"] == "correlation" and x.get("role") == "lever" and x["ok"]), None)
            if corr and corr["significant"] and c["p"] >= 0.05:
                ev.append(_item("limitation", "moderate", "Possible confounding",
                                f"{fr.lever} correlates with {fr.target} on its own, but not after adjusting for other variables — "
                                "the simple relationship may be driven by something else.", i, "lever"))

        elif role == "lever" and tool in ("group_compare", "segment_vs_rest"):
            if fr.lever_value and tool == "segment_vs_rest":
                if r["significant"]:
                    helpful = r["direction"] * act > 0
                    ev.append(_item("supports" if helpful else "challenges", _strength_from(p, abs(r["effect"]), (0.2, 0.5)),
                                    f"{fr.lever_value} {'outperforms' if r['direction'] > 0 else 'underperforms'} other {fr.lever} options",
                                    r["finding"], i, "lever"))
                else:
                    ev.append(_item("challenges", "moderate", f"{fr.lever_value} is not significantly different from alternatives",
                                    r["finding"], i, "lever"))
            elif tool == "group_compare" and fr.lever_value:
                if r["significant"]:
                    ev.append(_item("context", "moderate", f"{fr.target} differs across {fr.lever} options", r["finding"], i, "lever"))
            elif tool == "group_compare":
                ev.append(_item("supports" if r["significant"] else "challenges",
                                _strength_from(p, r["effect"] or 0, (0.01, 0.06)) if r["significant"] else "moderate",
                                f"{fr.target} {'does' if r['significant'] else 'does not'} differ meaningfully across {fr.lever}",
                                r["finding"], i, "lever"))

        # ---- Side effects
        elif role == "side_effect":
            if r["significant"]:
                harmful = np.sign(r["effect"]) * act < 0
                y = r["title"].split(" vs ")[-1].split(" (")[0]
                ev.append(_item("challenges" if harmful else "supports", _strength_from(p, abs(r["effect"])),
                                f"{'Side-effect' if harmful else 'Other outcome'}: {verb} {fr.lever} is linked to {'lower' if harmful else 'higher'} {y}",
                                r["finding"], i, "side_effect"))

        # ---- Segment context
        elif role == "segment":
            if r["significant"]:
                under = r["direction"] < 0
                ev.append(_item("context", "moderate",
                                f"The focus segment {'underperforms' if under else 'outperforms'} the rest",
                                r["finding"], i, "segment"))

        # ---- Robustness
        elif tool == "outliers":
            if r["stats"]["pct_outliers"] > 5 or abs(r["stats"]["mean_shift_pct"]) > 10:
                ev.append(_item("limitation", "moderate", "Outliers may distort averages", r["finding"], i, "robustness"))

        elif role == "context" and tool == "group_compare" and r["significant"]:
            ev.append(_item("context", "weak", r["title"], r["finding"], i, "context"))

    # data limitations
    for issue in quality.get("issues", []):
        if issue["severity"] == "critical":
            ev.append(_item("limitation", "strong", f"Data issue: {issue['check']}", issue["detail"], None, "data"))
    for u in fr.unanswerable:
        ev.append(_item("limitation", "strong", "Missing information", u, None, "scope"))
    ev.append(_item("limitation", "moderate", "Observational data",
                    "These results show associations in historical data. They cannot prove that the decision will cause "
                    "the expected outcome — consider a small controlled test (A/B or pilot) before a full rollout.", None, "method"))

    # verdict
    sup = sum(STRENGTH_W[e["strength"]] for e in ev if e["stance"] == "supports")
    cha = sum(STRENGTH_W[e["strength"]] for e in ev if e["stance"] == "challenges")
    total = sup + cha
    core_tested = any(e["category"] in ("lever", "premise") for e in ev if e["stance"] in ("supports", "challenges"))
    blocking = any(e["category"] in ("scope",) for e in ev) and not core_tested
    if total == 0 or blocking or fr.target is None:
        verdict, tone = "Insufficient evidence", "grey"
    else:
        share = sup / total
        if share >= 0.7:
            verdict, tone = "Evidence mostly supports the decision", "green"
        elif share >= 0.45:
            verdict, tone = "Mixed evidence — proceed with caution", "amber"
        else:
            verdict, tone = "Evidence mostly challenges the decision", "red"

    # confidence in the analysis (not in the decision)
    conf = 100
    n = results[0]["n"] if results and results[0]["ok"] else 0
    if n < 20: conf -= 50
    elif n < 30: conf -= 35
    elif n < 100: conf -= 12
    conf -= (100 - quality.get("score", 100)) * 0.4
    conf -= 10 * sum(1 for e in ev if e["category"] == "scope")
    conf -= 8 * sum(1 for r in results if not r["ok"])
    if not core_tested: conf -= 25
    conf -= 5 * sum(1 for e in ev if e["stance"] == "limitation" and e["category"] in ("robustness", "lever"))
    conf = int(max(5, min(100, conf)))
    conf_label = "High" if conf >= 75 else "Medium" if conf >= 50 else "Low"

    return {"items": ev, "support_score": sup, "challenge_score": cha, "verdict": verdict, "tone": tone,
            "support_share": (sup / total) if total else None, "confidence": conf, "confidence_label": conf_label}
