"""
DecisionCheck AI — An AI-powered statistical evidence checker for business decisions.
Run locally:   streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import pandas as pd
import streamlit as st

import analysis as A
from agent import run_agent
import history
from chat import answer as chat_answer
from data_quality import clean_for_analysis, infer_schema, list_sheets, load_file, quality_report
from llm import LLM, test_connection
from planner import DecisionFrame, build_plan, interpret, suggest_decisions
from report import build_html_report
from whatif import categorical_whatif, lever_range, numeric_whatif
from ui_extra import render_guide, render_history, render_pilot

ROOT = Path(__file__).parent
st.set_page_config(page_title="DecisionCheck AI", page_icon="✅", layout="wide", initial_sidebar_state="expanded")

# --------------------------------------------------------------------------- #
# Styling
# --------------------------------------------------------------------------- #
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"], .stMarkdown, .stText {font-family: 'Inter', sans-serif;}
.block-container {padding-top: 1.6rem; max-width: 1250px;}
.hero {background: linear-gradient(120deg,#0f172a 0%,#1e3a8a 55%,#2563eb 100%); color:#fff; border-radius:18px;
       padding:26px 30px; margin-bottom:18px;}
.hero h1 {color:#fff; margin:0; font-size:2.0rem; font-weight:800; letter-spacing:-0.5px}
.hero p {color:#cbd5e1; margin:6px 0 0; font-size:1.02rem}
.hero .pill {display:inline-block; background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.25);
       border-radius:99px; padding:3px 11px; margin:10px 6px 0 0; font-size:.8rem; color:#e2e8f0}
.step {display:flex; align-items:center; gap:10px; margin:22px 0 8px}
.step .n {background:#2563eb; color:#fff; width:30px; height:30px; border-radius:50%; display:flex;
       align-items:center; justify-content:center; font-weight:700}
.step .t {font-size:1.25rem; font-weight:700; color:#0f172a}
.verdict {border-radius:16px; padding:20px 24px; color:#fff; margin:8px 0 14px}
.verdict .big {font-size:1.45rem; font-weight:800}
.verdict .sub {opacity:.92; margin-top:4px}
.v-green{background:linear-gradient(120deg,#047857,#10b981)} .v-amber{background:linear-gradient(120deg,#b45309,#f59e0b)}
.v-red{background:linear-gradient(120deg,#b91c1c,#ef4444)} .v-grey{background:linear-gradient(120deg,#334155,#64748b)}
.ev {border-radius:12px; padding:12px 14px; margin-bottom:10px; border:1px solid #e2e8f0; background:#fff}
.ev.sup{border-left:5px solid #10b981} .ev.cha{border-left:5px solid #ef4444} .ev.lim{border-left:5px solid #94a3b8}
.ev.ctx{border-left:5px solid #6366f1}
.ev .h {font-weight:650; color:#0f172a} .ev .d {color:#475569; font-size:.9rem; margin-top:3px}
.badge {display:inline-block; font-size:.72rem; font-weight:600; border-radius:6px; padding:1px 8px; margin-left:6px}
.b-strong{background:#fee2e2;color:#991b1b} .b-moderate{background:#fef3c7;color:#92400e} .b-weak{background:#e2e8f0;color:#334155}
.trace {border-left:2px solid #cbd5e1; padding-left:14px; margin-left:6px}
.trace .row {margin-bottom:9px} .trace .ph {font-weight:700; color:#1d4ed8; margin-right:6px}
.kpi {background:#fff; border:1px solid #e2e8f0; border-radius:12px; padding:12px 16px}
.kpi .l {color:#64748b; font-size:.8rem} .kpi .v {font-size:1.4rem; font-weight:750; color:#0f172a}
.human {border:2px solid #2563eb; border-radius:16px; padding:6px 18px 14px; background:#f8fbff}
.note {font-size:.85rem; color:#64748b}
@media (prefers-color-scheme: dark) {
  .ev, .kpi {background:#111827; border-color:#334155} .ev .h, .kpi .v, .step .t {color:#f1f5f9} .ev .d {color:#cbd5e1}
  .human {background:#0b1220}
}
</style>""", unsafe_allow_html=True)

ss = st.session_state
for k, v in {"df": None, "raw": None, "sch": None, "quality": None, "name": None, "frame": None,
             "plan": None, "out": None, "human": None, "decision": "", "chat": []}.items():
    ss.setdefault(k, v)


def _secret(name):
    try:
        return st.secrets.get(name, "")
    except Exception:  # noqa: BLE001
        return ""


# --------------------------------------------------------------------------- #
# Sidebar: AI settings + about
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## ✅ DecisionCheck AI")
    st.caption("Statistical evidence checker for business decisions")
    page = st.radio("Go to", ["🔍 Check a decision", "📈 Did it work?", "🗂️ Decision history", "📖 How to use"],
                    label_visibility="collapsed", key="page")
    st.divider()
    st.markdown("#### 🤖 Generative AI")
    default_provider = "Gemini" if _secret("GEMINI_API_KEY") else "Groq" if _secret("GROQ_API_KEY") else "None (rule engine)"
    provider = st.selectbox("AI provider", ["Gemini", "Groq", "None (rule engine)"],
                            index=["Gemini", "Groq", "None (rule engine)"].index(default_provider),
                            help="Gemini and Groq both offer free API keys. Without a key the app still works using its built-in rule engine.")
    key = ""
    if provider != "None (rule engine)":
        secret_key = _secret("GEMINI_API_KEY" if provider == "Gemini" else "GROQ_API_KEY")
        if secret_key:
            st.success("API key loaded from app secrets", icon="🔐")
            key = secret_key
        else:
            key = st.text_input(f"{provider} API key", type="password",
                                help="Get a free key at aistudio.google.com (Gemini) or console.groq.com (Groq). Used only for this session.")
    llm = LLM(provider.split()[0].lower() if key else "none", key, _secret("AI_MODEL") or None)
    if llm.enabled and st.button("Test AI connection", width="stretch"):
        ok, msg = test_connection(llm)
        (st.success if ok else st.error)(msg)
    if llm.enabled:
        st.caption("🔒 Only column names, summary statistics and computed results are sent to the AI — never your raw rows.")
    else:
        st.caption("Running without AI: interpretation and explanations use transparent rules.")

    st.divider()
    with st.expander("ℹ️ How it works"):
        st.markdown("""
1. **Perceive** – profiles your data and checks quality
2. **Interpret** – AI maps your decision to columns
3. **Plan** – picks suitable statistical tests
4. **Act** – Python runs the calculations
5. **Observe** – adds follow-up tests if results need it
6. **Evaluate** – sorts supporting vs challenging evidence
7. **Explain** – AI writes a plain-English summary
8. **Verify** – checks AI numbers against calculations
9. **You decide** – human checkpoint & audit log
""")
    with st.expander("⚖️ Responsible AI"):
        st.markdown("""
- The app **never executes** a decision; a human must record it.
- All statistics are calculated by Python, **not** by the AI.
- AI numbers are **grounding-checked** against computed results.
- Correlation ≠ causation is flagged on every relevant result.
- Use synthetic/public data; don't upload confidential data without approval.
- **Do not use** for decisions about individuals (hiring, credit, health) or with tiny samples.
""")
    n_hist = len(history.load())
    if n_hist:
        st.caption(f"🗂️ {n_hist} decision(s) saved in history")

# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown("""
<div class="hero">
  <h1>DecisionCheck AI</h1>
  <p>Check whether your data actually supports a business decision — before you act.</p>
  <span class="pill">📂 Any CSV / Excel</span><span class="pill">🧠 Agentic analysis plan</span>
  <span class="pill">📊 Statistical tests</span><span class="pill">⚖️ Evidence for & against</span><span class="pill">🙋 Human decides</span>
</div>""", unsafe_allow_html=True)


if page == "📈 Did it work?":
    render_pilot(ROOT, llm)
    st.stop()
if page == "🗂️ Decision history":
    render_history()
    st.stop()
if page == "📖 How to use":
    render_guide()
    st.stop()


def step(n, title):
    st.markdown(f'<div class="step"><div class="n">{n}</div><div class="t">{title}</div></div>', unsafe_allow_html=True)


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="l">{label}</div><div class="v">{value}</div></div>', unsafe_allow_html=True)


def load_into_state(src, name, sheet=None):
    raw = load_file(src, sheet)
    df, sch = infer_schema(raw)
    q = quality_report(raw, df, sch)
    ss.raw, ss.sch, ss.quality, ss.name = raw, sch, q, name
    ss.df = clean_for_analysis(df, sch)
    ss.frame = ss.plan = ss.out = ss.human = None
    ss.decision = ""
    ss.chat = []


# --------------------------------------------------------------------------- #
# STEP 1 — Data
# --------------------------------------------------------------------------- #
step(1, "Upload your business data")
c1, c2 = st.columns([3, 2])
with c1:
    up = st.file_uploader("CSV or Excel file", type=["csv", "xlsx", "xls"], label_visibility="collapsed")
    if up is not None:
        sheets = list_sheets(up)
        sheet = None
        if len(sheets) > 1:
            sheet = st.selectbox("This Excel file has several sheets — which one holds the data?", sheets,
                                 index=max(range(len(sheets)), key=lambda i: "note" not in sheets[i].lower() and "about" not in sheets[i].lower()))
        label = up.name + (f" › {sheet}" if sheet else "")
        if label != ss.name:
            try:
                load_into_state(up, label, sheet)
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read this file: {e}")
with c2:
    st.caption("No data? Try a synthetic sample:")
    b1, b2 = st.columns(2)
    if b1.button("🛒 Retail sales", width="stretch"):
        load_into_state(str(ROOT / "sample_retail_sales.csv"), "sample_retail_sales.csv")
    if b2.button("👥 HR performance", width="stretch"):
        load_into_state(str(ROOT / "sample_hr_performance.csv"), "sample_hr_performance.csv")

if ss.df is None:
    st.info("Upload a dataset (or pick a sample) to begin. Any tabular business data works — sales, HR, marketing, operations, finance.")
    st.stop()

q, sch, df = ss.quality, ss.sch, ss.df
k1, k2, k3, k4, k5 = st.columns(5)
kpi(k1, "Dataset", ss.name if len(ss.name) < 22 else ss.name[:20] + "…")
kpi(k2, "Rows", f"{q['rows']:,}")
kpi(k3, "Columns", q["columns"])
kpi(k4, "Measures / Categories", f"{len(sch.numeric)} / {len(sch.categorical)}")
kpi(k5, "Data quality", f"{q['score']}/100")

with st.expander("🔎 Data profile & quality checks", expanded=q["score"] < 80):
    t1, t2, t3 = st.tabs(["Quality issues", "Column profile", "Preview"])
    with t1:
        if not q["issues"]:
            st.success("No data-quality issues detected.")
        icon = {"critical": "🔴", "warning": "🟠", "info": "🔵"}
        for i in q["issues"]:
            st.markdown(f"{icon[i['severity']]} **{i['check']}** — {i['detail']}")
        st.caption("Automatic cleaning applied: exact duplicates removed, text numbers/dates converted, label case unified. "
                   "Missing values are excluded test-by-test (not filled in).")
    with t2:
        st.dataframe(q["profile"], hide_index=True, width="stretch")
    with t3:
        st.dataframe(ss.raw.head(50), width="stretch")

# --------------------------------------------------------------------------- #
# STEP 2 — Decision
# --------------------------------------------------------------------------- #
step(2, "Describe the decision you are considering")
sugg = suggest_decisions(df, sch)
if sugg:
    pick = st.pills("Examples for this dataset", sugg, selection_mode="single", key="pill")
    if pick and pick != ss.get("_last_pick"):
        ss.decision, ss._last_pick = pick, pick
        ss.frame = ss.plan = ss.out = ss.human = None
decision = st.text_area("Proposed decision", value=ss.decision, height=80,
                        placeholder="e.g. Should we increase the discount on Product A because its sales are declining?")
if decision != ss.decision:
    ss.decision = decision
    ss.frame = ss.plan = ss.out = ss.human = None

if st.button("🧠 Let the agent interpret & plan", type="primary", disabled=not decision.strip()):
    with st.spinner("Interpreting your decision…"):
        ss.frame = interpret(decision, df, sch, llm)
        ss.plan = build_plan(ss.frame, df, sch)
        ss.out = ss.human = None

if ss.frame is None:
    st.stop()

# ---- Human review of the agent's interpretation & plan
fr: DecisionFrame = ss.frame
with st.container(border=True):
    st.markdown(f"**Agent's interpretation** · <span class='note'>by {'Generative AI' if fr.interpreter == 'ai' else 'rule engine'} — "
                "check it and correct anything that's wrong</span>", unsafe_allow_html=True)
    none = "— none —"
    c = st.columns(4)
    tgt = c[0].selectbox("Outcome to improve", sch.numeric, index=sch.numeric.index(fr.target) if fr.target in sch.numeric else 0)
    lever_opts = [none] + [x for x in sch.numeric + sch.categorical if x != tgt]
    lev = c[1].selectbox("Lever you would change", lever_opts, index=lever_opts.index(fr.lever) if fr.lever in lever_opts else 0)
    dirs = {"Increase": 1, "Decrease": -1, "Unclear": 0}
    d_lbl = c[2].selectbox("Direction", list(dirs), index=[1, -1, 0].index(fr.action_direction))
    prem_opts = {"No claim": 0, "is declining": -1, "is rising": 1}
    p_lbl = c[3].selectbox(f"Premise about {tgt}", list(prem_opts), index=[0, -1, 1].index(fr.premise_direction))
    c = st.columns(4)
    focus_cols = [none] + sch.categorical
    cur_fc = next(iter(fr.focus), none)
    fcol = c[0].selectbox("Focus segment (column)", focus_cols, index=focus_cols.index(cur_fc) if cur_fc in focus_cols else 0)
    fval = None
    if fcol != none:
        vals = sorted(df[fcol].dropna().astype(str).unique().tolist())
        cur_v = fr.focus.get(fcol)
        fval = c[1].selectbox("Focus value", vals, index=vals.index(cur_v) if cur_v in vals else 0)
    lev_val = None
    if lev in sch.categorical:
        vals = sorted(df[lev].dropna().astype(str).unique().tolist())
        lev_val = c[2].selectbox("Lever option", [none] + vals,
                                 index=([none] + vals).index(fr.lever_value) if fr.lever_value in vals else 0)
        lev_val = None if lev_val == none else lev_val
    for n in fr.notes:
        st.caption("ℹ️ " + n)
    for u in fr.unanswerable:
        st.warning(u, icon="⚠️")

    new = DecisionFrame(decision=fr.decision, action_direction=dirs[d_lbl], lever=None if lev == none else lev,
                        lever_kind=None if lev == none else ("numeric" if lev in sch.numeric else "categorical"),
                        lever_value=lev_val, target=tgt,
                        secondary_outcomes=[x for x in fr.secondary_outcomes if x not in (tgt, lev)] or
                        [x for x in sch.numeric if x not in (tgt, lev) and any(w in x.lower() for w in ("profit", "margin", "quantity", "units"))][:2],
                        focus={fcol: fval} if fval else {}, premise_metric=tgt if prem_opts[p_lbl] else None,
                        premise_direction=prem_opts[p_lbl], date_col=fr.date_col,
                        group_cols=[g for g in fr.group_cols if g != fcol and g != lev], interpreter=fr.interpreter,
                        notes=fr.notes, unanswerable=fr.unanswerable)
    changed = (new.target, new.lever, new.action_direction, new.focus, new.premise_direction, new.lever_value) != \
              (fr.target, fr.lever, fr.action_direction, fr.focus, fr.premise_direction, fr.lever_value)
    if changed:
        ss.frame, ss.plan, ss.out, ss.human = new, build_plan(new, df, sch), None, None
        st.rerun()

    st.markdown("**Analysis plan** <span class='note'>— untick any test you don't want; the agent may add follow-ups after seeing results</span>",
                unsafe_allow_html=True)
    plan_df = pd.DataFrame([{"Run": True, "Analysis": p["tool"].replace("_", " ").title(), "Why": p["purpose"]} for p in ss.plan])
    edited = st.data_editor(plan_df, hide_index=True, width="stretch", disabled=["Analysis", "Why"],
                            column_config={"Run": st.column_config.CheckboxColumn(width="small")}, key=f"plan_{hash(str(ss.plan))}")
    run = st.button("▶️ Run DecisionCheck", type="primary")

if run:
    chosen = [p for p, keep in zip(ss.plan, edited["Run"]) if keep]
    bar = st.progress(0.0, text="Starting…")
    ss.out = run_agent(decision, df, sch, q, llm, plan_override=[dict(p) for p in chosen], frame_override=ss.frame,
                       progress=lambda m, f: bar.progress(min(f, 1.0), text=m))
    bar.empty()
    ss.human = None

if ss.out is None:
    st.stop()

# --------------------------------------------------------------------------- #
# STEP 3 — Evidence
# --------------------------------------------------------------------------- #
out = ss.out
ev, ex = out["evidence"], out["explanation"]
step(3, "Review the evidence")
st.markdown(f"""<div class="verdict v-{ev['tone']}"><div class="big">{ev['verdict']}</div>
<div class="sub">Analysis confidence: <b>{ev['confidence_label']}</b> ({ev['confidence']}/100) ·
{sum(e['stance']=='supports' for e in ev['items'])} supporting · {sum(e['stance']=='challenges' for e in ev['items'])} challenging ·
{sum(e['stance']=='limitation' for e in ev['items'])} limitations · {len(out['results'])} analyses in {out['elapsed']}s</div>
<div class="sub" style="font-size:.85rem">This is an evidence summary, not a decision. You make the final call in Step 4.</div></div>""",
            unsafe_allow_html=True)

if ev["support_share"] is not None:
    import plotly.graph_objects as go
    fig = go.Figure()
    fig.add_trace(go.Bar(y=["Evidence"], x=[ev["support_score"]], orientation="h", name="Supports", marker_color="#10b981",
                         text=f"Supports ({ev['support_score']})", textposition="inside"))
    fig.add_trace(go.Bar(y=["Evidence"], x=[ev["challenge_score"]], orientation="h", name="Challenges", marker_color="#ef4444",
                         text=f"Challenges ({ev['challenge_score']})", textposition="inside"))
    fig.update_layout(barmode="stack", height=90, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      xaxis=dict(visible=False), yaxis=dict(visible=False), plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.caption("Weighted by strength: strong = 3, moderate = 2, weak = 1.")


def ev_card(e, cls):
    st.markdown(f"""<div class="ev {cls}"><div class="h">{e['headline']}<span class="badge b-{e['strength']}">{e['strength']}</span></div>
<div class="d">{e['detail']}</div></div>""", unsafe_allow_html=True)


tabs = st.tabs(["⚖️ Evidence", "💬 AI explanation", "📊 Analyses & charts", "🔮 What-if", "🙋 Ask a question", "🧭 Agent trace"])
with tabs[0]:
    a, b = st.columns(2)
    with a:
        st.markdown("#### ✅ Supports the decision")
        items = [e for e in ev["items"] if e["stance"] == "supports"]
        for e in items:
            ev_card(e, "sup")
        if not items:
            st.caption("No supporting evidence found.")
    with b:
        st.markdown("#### ❌ Challenges the decision")
        items = [e for e in ev["items"] if e["stance"] == "challenges"]
        for e in items:
            ev_card(e, "cha")
        if not items:
            st.caption("No challenging evidence found.")
    ctx = [e for e in ev["items"] if e["stance"] == "context"]
    if ctx:
        st.markdown("#### 🧩 Useful context")
        for e in ctx:
            ev_card(e, "ctx")
    st.markdown("#### ⚠️ Limitations — what the data cannot tell you")
    for e in [e for e in ev["items"] if e["stance"] == "limitation"]:
        ev_card(e, "lim")

with tabs[1]:
    g = ex["grounding"]
    if ex["source"] == "ai":
        if g["passed"]:
            st.success(f"Written by {ex['model']} · Grounding check passed: all {g['checked']} numbers match computed results.", icon="🛡️")
        else:
            st.warning(f"Written by {ex['model']} · Grounding check: {len(g['unverified'])} of {g['checked']} numbers could not be "
                       f"matched to computed results ({', '.join(map(str, g['unverified']))}). Verify these against the Analyses tab.", icon="⚠️")
    else:
        st.info("Written by the built-in rule engine (no AI key configured" + (f"; AI error: {ex['error']}" if ex.get("error") else "") +
                "). Add a free Gemini or Groq key in the sidebar for a natural-language explanation.", icon="📝")
    st.markdown(ex["text"])
    st.caption("AI-generated interpretation of calculated results. Statistics come from Python, not the AI.")

with tabs[2]:
    for i, r in enumerate(out["results"]):
        tag = " · follow-up" if r.get("step_status") == "follow-up" else ""
        sig = "" if r["significant"] is None else (" · 🟢 significant" if r["significant"] else " · ⚪ not significant")
        with st.expander(f"{'✅' if r['ok'] else '⛔'} {r['title']}{sig}{tag}", expanded=(i < 2)):
            st.markdown(f"**Why:** {r.get('purpose', '')}  \n**Method:** {r['method']} · n = {r['n']:,}")
            st.markdown(f"**Finding:** {r['finding']}")
            if r.get("fig") is not None:
                st.plotly_chart(r["fig"], width="stretch", key=f"fig{i}")
            if r.get("caveats"):
                for cv in r["caveats"]:
                    st.caption("⚠️ " + cv)
            if r.get("table") is not None and len(r["table"]):
                st.dataframe(r["table"], hide_index=True, width="stretch")
            if r.get("posthoc") is not None:
                st.caption("Tukey HSD post-hoc comparisons")
                st.dataframe(r["posthoc"], hide_index=True, width="stretch")
    st.caption(f"Significance level α = {A.ALPHA}. p < 0.05 means the pattern is unlikely to be random chance.")

with tabs[3]:
    frw = out["frame"]
    st.markdown("Estimate what **could** happen if you make the change — a range based on past data, not a promise.")
    if frw.lever and frw.lever in sch.numeric:
        lo_v, cur_v, hi_v, sd_v = lever_range(df, frw)
        step_v = float(f"{max(sd_v / 10, 1e-6):.1g}")
        default = round(sd_v / 2 * (frw.action_direction or 1), 2)
        cA, cB = st.columns([2, 3])
        chg = cA.number_input(f"Change in {frw.lever} (current average {cur_v:,.2f})", value=float(default), step=step_v,
                              help=f"Positive = increase, negative = decrease. Your data ranges {lo_v:,.2f} to {hi_v:,.2f}.")
        cA.caption(f"New level: **{cur_v + chg:,.2f}**")
        w = numeric_whatif(df, sch, frw, chg)
    elif frw.lever and frw.lever_value:
        share = st.slider(f"How much activity would {'move to' if frw.action_direction >= 0 else 'move away from'} {frw.lever_value}? (%)", 5, 100, 20, 5)
        w = categorical_whatif(df, frw, share)
    else:
        w = None
        st.info("No lever was identified for this decision. Set a lever in Step 2 (e.g. discount, price, marketing spend) to use the what-if estimator.")
    if w and w["rows"]:
        for r in w["rows"]:
            good = r["Estimated change (per row)"] >= 0
            mtxt = f" · about **{r['Monthly impact']:+,.0f} per month** in total" if r.get("Monthly impact") is not None else ""
            st.markdown(f"{'🟢' if good else '🔴'} **{r['Outcome']}**: {r['Estimated change (per row)']:+,.2f} per row "
                        f"(**{r['% change']:+.1f}%**), likely between {r['Likely range low']:+,.2f} and {r['Likely range high']:+,.2f}{mtxt}"
                        f"{'' if r['Reliable?'].startswith('Yes') else ' — ⚠️ *not reliable: the range includes zero*'}")
        if w.get("fig") is not None:
            st.plotly_chart(w["fig"], width="stretch", key="whatif_fig")
        for n_ in w["notes"]:
            st.caption(n_)
    elif w:
        for n_ in w["notes"]:
            st.warning(n_)

with tabs[4]:
    st.markdown("Ask anything about your data. The agent picks the right analysis, runs it, and answers from the result.")
    ex_q = [f"Which {sch.categorical[0].replace('_', ' ').lower()} has the highest {out['frame'].target.replace('_', ' ').lower()}?"] if sch.categorical else []
    ex_q += [f"Why is {out['frame'].target.replace('_', ' ').lower()} changing?", "Are there any unusual values?"]
    picked = st.pills("Try:", ex_q, key="chat_pills")
    for i_, m_ in enumerate(ss.chat):
        with st.chat_message(m_["role"]):
            st.markdown(m_["text"])
            if m_.get("fig") is not None:
                st.plotly_chart(m_["fig"], width="stretch", key=f"chatfig{i_}")
            if m_.get("meta"):
                st.caption(m_["meta"])
    qtext = st.chat_input("e.g. Which product has the lowest profit?")
    question = qtext or (picked if picked and picked != ss.get("_last_chat_pill") else None)
    if picked:
        ss._last_chat_pill = picked
    if question:
        with st.spinner("Choosing and running an analysis…"):
            a_ = chat_answer(question, df, sch, out["frame"], llm)
        meta = (f"Tool used: {a_['tool'].replace('_', ' ') if a_['tool'] else '—'} · chosen by "
                f"{'Generative AI' if a_['by'] == 'ai' else 'rule engine'}")
        if a_.get("grounding") and not a_["grounding"]["passed"]:
            meta += f" · ⚠️ unverified numbers: {a_['grounding']['unverified']}"
        ss.chat += [{"role": "user", "text": question},
                    {"role": "assistant", "text": a_["text"], "fig": (a_["result"] or {}).get("fig"), "meta": meta}]
        st.rerun()
    if ss.chat and st.button("Clear conversation"):
        ss.chat = []
        st.rerun()

with tabs[5]:
    rows = "".join(f"<div class='row'><span class='ph'>{s['phase']}</span><span class='note'>{s['t']}s</span><br>{s['message']}</div>"
                   for s in out["trace"])
    st.markdown(f"<div class='trace'>{rows}</div>", unsafe_allow_html=True)
    with st.expander("Raw interpretation & plan (JSON)"):
        st.json({"interpretation": {k: v for k, v in out["frame"].to_dict().items()},
                 "plan": [{k: v for k, v in p.items()} for p in out["plan"]]})

# --------------------------------------------------------------------------- #
# STEP 4 — Human decision
# --------------------------------------------------------------------------- #
step(4, "Your decision (human-in-the-loop checkpoint)")
with st.container(border=True):
    if ss.human:
        h = ss.human
        st.success(f"**Decision recorded:** {h['choice']} — by {h['name'] or 'reviewer'} at {h['time']}", icon="✅")
        st.markdown(f"**Rationale:** {h['rationale']}")
        if h.get("entry"):
            st.caption(f"Saved to Decision history as #{h['entry']['ID']}."
                       + (" After implementing it, use **📈 Did it work?** in the sidebar to check the results." if h['entry']['Follow-up status'] != "Closed" else ""))
        if st.button("Edit decision"):
            ss.human = None
            st.rerun()
    else:
        with st.form("human_form"):
            choice = st.radio("What will you do?", ["Approve as proposed", "Approve with modifications", "Run a pilot / A-B test first",
                                                    "Reject", "Collect more data first"], horizontal=True, index=None)
            rationale = st.text_area("Your reasoning (required)", placeholder="e.g. Decline is real but discount hurts profit; test a bundle offer in one region instead.")
            name = st.text_input("Your name / role (optional)")
            ack = st.checkbox("I have reviewed both the supporting and the challenging evidence, and I accept responsibility for this decision.")
            if st.form_submit_button("Record my decision", type="primary"):
                if not choice or not rationale.strip() or not ack:
                    st.error("Please choose an option, write your reasoning and tick the confirmation box.")
                else:
                    ss.human = {"choice": choice, "rationale": rationale.strip(), "name": name.strip(),
                                "time": dt.datetime.now().strftime("%d %b %Y %H:%M")}
                    ss.human["entry"] = history.record(out, ss.human, ss.name)
                    st.rerun()

report = build_html_report(out, q, ss.name, ss.human)
st.download_button("⬇️ Download full evidence report (HTML — open in browser, print to PDF)", report,
                   file_name=f"DecisionCheck_report_{dt.datetime.now():%Y%m%d_%H%M}.html", mime="text/html",
                   type="primary" if ss.human else "secondary")
st.caption("DecisionCheck AI · Business Analytics decision support · Statistics: pandas, SciPy, statsmodels · Visuals: Plotly · UI: Streamlit")
