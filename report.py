"""Build a self-contained HTML decision report (opens in any browser, printable to PDF)."""
from __future__ import annotations

import datetime as dt
import html

import markdown as md_lib

TONE = {"green": "#059669", "amber": "#D97706", "red": "#DC2626", "grey": "#64748B"}


def _md(text):
    try:
        return md_lib.markdown(text, extensions=["tables"])
    except Exception:  # noqa: BLE001
        return "<pre>" + html.escape(text) + "</pre>"


def build_html_report(out: dict, quality: dict, dataset_name: str, human: dict | None) -> str:
    fr, ev, ex = out["frame"], out["evidence"], out["explanation"]
    col = TONE.get(ev["tone"], "#64748B")
    now = dt.datetime.now().strftime("%d %b %Y, %H:%M")

    def ev_list(stance):
        items = [e for e in ev["items"] if e["stance"] == stance]
        if not items:
            return "<p class='muted'>None.</p>"
        return "<ul>" + "".join(f"<li><b>{html.escape(e['headline'])}</b> <span class='tag'>{e['strength']}</span><br>"
                                f"<span class='muted'>{html.escape(e['detail'])}</span></li>" for e in items) + "</ul>"

    figs = []
    first = True
    for r in out["results"]:
        if not r["ok"]:
            continue
        block = f"<div class='card'><h3>{html.escape(r['title'])}</h3><p class='muted'>Method: {html.escape(r['method'])} · n = {r['n']:,}</p>"
        block += f"<p>{html.escape(r['finding'])}</p>"
        if r.get("fig") is not None:
            block += r["fig"].to_html(full_html=False, include_plotlyjs="cdn" if first else False)
            first = False
        if r.get("caveats"):
            block += "<ul class='cav'>" + "".join(f"<li>{html.escape(c)}</li>" for c in r["caveats"]) + "</ul>"
        figs.append(block + "</div>")

    issues = "".join(f"<li><b>{i['severity'].title()}</b> — {html.escape(i['check'])}: {html.escape(i['detail'])}</li>"
                     for i in quality["issues"]) or "<li>No issues detected.</li>"
    trace = "".join(f"<tr><td>{s['t']}s</td><td><b>{s['phase']}</b></td><td>{html.escape(s['message'])}</td></tr>" for s in out["trace"])

    hd = ""
    if human:
        hd = f"""<div class='card human'><h2>Human decision (final)</h2>
        <p><b>Decision:</b> {html.escape(human['choice'])}</p>
        <p><b>Rationale:</b> {html.escape(human['rationale'])}</p>
        <p class='muted'>Recorded by {html.escape(human.get('name') or 'reviewer')} on {human['time']}.
        Reviewer confirmed they reviewed supporting and challenging evidence.</p></div>"""
    else:
        hd = "<div class='card human'><h2>Human decision</h2><p class='muted'>Not yet recorded.</p></div>"

    g = ex.get("grounding", {})
    gnote = ""
    if ex["source"] == "ai":
        gnote = (f"<p class='muted'>AI explanation by {html.escape(ex['model'])}. Grounding check: {g.get('checked', 0)} numbers checked, "
                 f"{len(g.get('unverified', []))} not found in computed results"
                 + (f" ({', '.join(str(v) for v in g['unverified'])})" if g.get("unverified") else "") + ".</p>")

    return f"""<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>
<title>DecisionCheck AI report</title>
<style>
body{{font-family:Inter,Segoe UI,Arial,sans-serif;max-width:980px;margin:0 auto;padding:24px;color:#0f172a;background:#f8fafc;line-height:1.5}}
h1{{margin:0;font-size:26px}} h2{{font-size:19px;margin-top:0}} h3{{font-size:16px;margin:0 0 4px}}
.card{{background:#fff;border:1px solid #e2e8f0;border-radius:12px;padding:18px 20px;margin:14px 0}}
.verdict{{border-left:6px solid {col}}} .verdict .v{{color:{col};font-size:20px;font-weight:700}}
.muted{{color:#64748b;font-size:14px}} .tag{{background:#eef2ff;color:#3730a3;border-radius:6px;padding:1px 7px;font-size:12px}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}} .sup h2{{color:#059669}} .cha h2{{color:#DC2626}}
.cav{{color:#92400e;font-size:13px}} table{{border-collapse:collapse;width:100%;font-size:13px}} td{{border-bottom:1px solid #e2e8f0;padding:5px;vertical-align:top}}
.human{{border:2px solid #2563eb}} @media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}
.printbtn{{position:fixed;top:14px;right:14px;background:#2563eb;color:#fff;border:0;border-radius:8px;padding:9px 14px;font-weight:600;cursor:pointer}}
@media print{{body{{background:#fff}} .card{{break-inside:avoid}} .printbtn{{display:none}}}}
</style></head><body>
<button class='printbtn' onclick='window.print()'>🖨️ Print / Save as PDF</button>
<p class='muted'>DecisionCheck AI · Evidence report · {now}</p>
<h1>{html.escape(fr.decision)}</h1>
<p class='muted'>Dataset: {html.escape(dataset_name)} · {quality['rows']:,} rows · data quality {quality['score']}/100 · analysis time {out['elapsed']}s</p>
<div class='card verdict'><div class='v'>{ev['verdict']}</div>
<p>Analysis confidence: <b>{ev['confidence_label']}</b> ({ev['confidence']}/100). Support score {ev['support_score']} vs challenge score {ev['challenge_score']}.</p>
<p class='muted'>This report presents statistical evidence. It does not make the decision — a human reviewer is accountable for the final action.</p></div>
<div class='card'><h2>How the question was understood</h2>
<p>Outcome measure: <b>{fr.target}</b> · Lever: <b>{fr.lever or '—'}</b> · Direction: <b>{'increase' if fr.action_direction > 0 else 'decrease' if fr.action_direction < 0 else 'unclear'}</b>
 · Focus: <b>{', '.join(f'{k} = {v}' for k, v in fr.focus.items()) or 'all data'}</b>
 · Premise: <b>{(fr.premise_metric + (' declining' if fr.premise_direction < 0 else ' rising')) if fr.premise_direction else '—'}</b></p></div>
<div class='grid'><div class='card sup'><h2>Supporting evidence</h2>{ev_list('supports')}</div>
<div class='card cha'><h2>Challenging evidence</h2>{ev_list('challenges')}</div></div>
<div class='card'><h2>Limitations</h2>{ev_list('limitation')}</div>
<div class='card'><h2>Business explanation</h2>{gnote}{_md(ex['text'])}</div>
{hd}
<h2 style='margin-top:28px'>Statistical analyses</h2>{''.join(figs)}
<div class='card'><h2>Data quality</h2><ul>{issues}</ul></div>
<div class='card'><h2>Agent trace</h2><table>{trace}</table></div>
<p class='muted'>Generated by DecisionCheck AI. Statistics computed with Python (pandas, SciPy, statsmodels); significance level α = 0.05.
Associations in observational data do not prove causation.</p>
</body></html>"""
