"""Extra pages: 'Did it work?' pilot checker, Decision history, and the How-to guide."""
from __future__ import annotations

import datetime as dt
import json
import re

import pandas as pd
import streamlit as st

import analysis as A
import history
from data_quality import clean_for_analysis, infer_schema, list_sheets, load_file, quality_report
from impact import evaluate

ss = st.session_state
TONE_CLS = {"green": "v-green", "amber": "v-amber", "red": "v-red", "grey": "v-grey"}
LOWER_IS_BETTER = r"cost|expense|churn|return|complaint|defect|delay|waiting|attrition|absence|refund|error"


def _hero(title, sub):
    st.markdown(f'<div class="hero"><h1>{title}</h1><p>{sub}</p></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Did it work?
# --------------------------------------------------------------------------- #
def render_pilot(root, llm=None):
    _hero("📈 Did it work?", "After you implement a decision or a pilot, upload the new data and check whether it actually made a difference.")
    hist = history.load()
    pending = hist[hist["Follow-up status"] == "Check results later"]

    st.markdown("##### 1 · Data after the change")
    c1, c2 = st.columns([3, 2])
    with c1:
        up = st.file_uploader("Upload data that covers BEFORE and AFTER the change (CSV/Excel)", type=["csv", "xlsx", "xls"], key="pilot_up")
    with c2:
        st.caption("Or use:")
        if st.button("🧪 Sample pilot results", width="stretch",
                     help="Synthetic data: extra 10-point discount in the North region from 15 Oct 2025, during the festive season"):
            ss.pilot_src = ("file", str(root / "sample_pilot_results.csv"), "sample_pilot_results.csv", None)
        if ss.get("df") is not None and st.button("📂 Data from 'Check a decision'", width="stretch"):
            ss.pilot_src = ("loaded", None, ss.name, None)
    if up is not None:
        sheets = list_sheets(up)
        sheet = st.selectbox("Sheet", sheets, key="pilot_sheet") if len(sheets) > 1 else None
        ss.pilot_src = ("upload", up, up.name, sheet)

    src = ss.get("pilot_src")
    if not src:
        st.info("Tip: include a date column plus a column that identifies the pilot group (e.g. Region = North). "
                "Comparing the pilot group with the rest, before and after, gives the most trustworthy answer.")
        return
    try:
        if src[0] == "loaded":
            df, sch = ss.df, ss.sch
        else:
            raw = load_file(src[1], src[3])
            df, sch = infer_schema(raw)
            q = quality_report(raw, df, sch)
            df = clean_for_analysis(df, sch)
            if q["score"] < 70:
                st.warning(f"Data quality score {q['score']}/100 — check the data: " + "; ".join(i["detail"] for i in q["issues"][:3]))
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")
        return
    st.caption(f"Using **{src[2]}** · {len(df):,} rows")
    if not sch.numeric:
        st.error("No numeric measures found in this data.")
        return

    st.markdown("##### 2 · What changed, where and when?")
    link_opts = ["— not linked —"] + [f"#{r.ID} · {r['Decision question'][:70]}" for _, r in pending.iterrows()]
    link = st.selectbox("Link to a past decision (optional — the result will be saved to its history)", link_opts)
    linked = pending.iloc[link_opts.index(link) - 1] if link != link_opts[0] else None

    default_out = [c for c in sch.numeric if linked is not None and c == linked["Outcome"]] or \
                  [c for c in sch.numeric if re.search(r"sales|revenue|profit|performance|quantity", c, re.I)][:3] or sch.numeric[:1]
    outcomes = st.multiselect("Which results do you want to check? (pick the main goal AND possible side-effects, e.g. Sales and Profit)",
                              sch.numeric, default=default_out)
    c = st.columns(4)
    none = "— none —"
    date_col = c[0].selectbox("Date column", [none] + sch.datetime, index=1 if sch.datetime else 0)
    change_date = None
    if date_col != none:
        dmin, dmax = df[date_col].min().date(), df[date_col].max().date()
        mid = (df[date_col].min() + (df[date_col].max() - df[date_col].min()) / 2).date()
        change_date = c[1].date_input("Date the change started", value=mid, min_value=dmin, max_value=dmax)
    lf = eval_focus(linked, sch) if linked is not None else {}
    gopts = [none] + sch.categorical
    group_col = c[2].selectbox("Column identifying the pilot group", gopts,
                               index=gopts.index(next(iter(lf))) if lf and next(iter(lf)) in gopts else 0)
    test_value = None
    if group_col != none:
        vals = sorted(df[group_col].dropna().astype(str).unique())
        fv = lf.get(group_col)
        test_value = c[3].selectbox("Pilot / test group value", vals, index=vals.index(fv) if fv in vals else 0)
    if date_col == none and group_col == none:
        st.info("Choose a date column with the change date, and/or the pilot group.")
        return
    design = ("Difference-in-differences" if date_col != none and group_col != none else
              "Before vs after" if date_col != none else "Test vs control")
    st.caption(f"Method that will be used: **{design}**" +
               (" — the strongest option: it removes changes that would have happened anyway (season, market)."
                if design.startswith("Difference") else " — add both a date and a pilot group for a more reliable answer."))

    if not st.button("✅ Check the results", type="primary", disabled=not outcomes):
        return
    results = []
    for y in outcomes:
        desired = -1 if re.search(LOWER_IS_BETTER, y, re.I) else 1
        r = evaluate(df, y, None if date_col == none else date_col, change_date, None if group_col == none else group_col,
                     test_value, desired=desired)
        results.append(r)
    ok = [r for r in results if r["ok"]]
    for r in results:
        if not r["ok"]:
            st.error(r["message"])
    if not ok:
        return

    st.markdown("##### 3 · Results")
    summary = "; ".join(f"{r['target']}: {r['verdict'].split(' — ')[0].lower()} ({r['pct']:+.1f}%)" for r in ok)
    worked = sum(r["tone"] == "green" for r in ok)
    hurt = sum(r["tone"] == "red" for r in ok)
    overall = ("Mixed result — gains in some measures, losses in others" if worked and hurt else
               "The change worked on the measures checked" if worked else
               "The change made things worse" if hurt else "No clear evidence the change made a difference")
    tone = "amber" if worked and hurt else "green" if worked else "red" if hurt else "grey"
    st.markdown(f'<div class="verdict {TONE_CLS[tone]}"><div class="big">{overall}</div><div class="sub">{summary}</div></div>',
                unsafe_allow_html=True)
    for r in ok:
        with st.container(border=True):
            icon = {"green": "🟢", "red": "🔴", "amber": "🟡", "grey": "⚪"}[r["tone"]]
            st.markdown(f"#### {icon} {r['target']} — {r['verdict']}")
            st.markdown(f"Estimated effect: **{r['effect']:+,.2f} per row ({r['pct']:+.1f}%)**, likely between "
                        f"{r['ci'][0]:+,.2f} and {r['ci'][1]:+,.2f} · p = {A._p(r['p'])} · {r['design']} · n = {r['n']:,}")
            st.markdown(r["explain"])
            st.plotly_chart(r["fig"], width="stretch", key=f"pilotfig_{r['target']}")
            st.dataframe(r["table"].round(2), hide_index=True, width="stretch")
            for cv in r["caveats"]:
                st.caption("⚠️ " + cv)

    if llm is not None and llm.enabled:
        payload = [{k: r[k] for k in ("target", "design", "effect", "pct", "p", "verdict", "explain", "caveats")} for r in ok]
        txt = llm.generate("Pilot results (computed):\n" + json.dumps(payload, default=str) +
                           "\nWrite 3-5 sentences for a manager: did it work, trade-offs, and whether to roll out, adjust or stop. "
                           "Use only these numbers. End by saying the manager makes the final call.",
                           "You are DecisionCheck AI. Be balanced and never invent numbers.")
        if txt:
            st.markdown("**AI summary**")
            st.markdown(txt)

    if linked is not None:
        history.update_result(linked["ID"], f"{overall}. {summary} ({design}, checked {dt.date.today():%d %b %Y})")
        st.success(f"Saved to decision #{linked['ID']} in your history.", icon="🗂️")


def eval_focus(linked, sch):
    if linked is None or not linked.get("Focus"):
        lever = str(linked["Lever"]) if linked is not None else ""
        if " = " in lever:
            c, v = lever.split(" = ", 1)
            return {c: v} if c in sch.categorical else {}
        return {}
    out = {}
    for part in str(linked["Focus"]).split(", "):
        if " = " in part:
            c, v = part.split(" = ", 1)
            if c in sch.categorical:
                out[c] = v
    return out


# --------------------------------------------------------------------------- #
# Decision history
# --------------------------------------------------------------------------- #
def render_history():
    _hero("🗂️ Decision history", "Every decision recorded in DecisionCheck — what the evidence said, what was decided, why, and whether it worked.")
    df = history.load()
    c = st.columns(4)
    c[0].metric("Decisions recorded", len(df))
    c[1].metric("Awaiting result check", int((df["Follow-up status"] == "Check results later").sum()))
    c[2].metric("Results checked", int(df["Follow-up status"].str.startswith("Checked").sum()))
    agree = ((df["Evidence verdict"].str.contains("supports") & df["Human decision"].str.startswith("Approve")) |
             (df["Evidence verdict"].str.contains("challenges") & ~df["Human decision"].str.startswith("Approve"))).sum()
    c[3].metric("Decisions in line with evidence", f"{agree}/{len(df)}" if len(df) else "—")

    if df.empty:
        st.info("No decisions yet. Record one in Step 4 of **🔍 Check a decision**, or upload a history file below.")
    else:
        search = st.text_input("Search", placeholder="e.g. discount, Product A, Reject")
        view = df[df.apply(lambda r: search.lower() in " ".join(r.astype(str)).lower(), axis=1)] if search else df
        st.dataframe(view.iloc[::-1], hide_index=True, width="stretch",
                     column_config={"Rationale": st.column_config.TextColumn(width="large"),
                                    "Decision question": st.column_config.TextColumn(width="large")})
        st.download_button("⬇️ Download history (CSV / opens in Excel)", df.to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"decision_history_{dt.date.today():%Y%m%d}.csv", mime="text/csv", type="primary")
        with st.expander("Delete an entry"):
            sel = st.selectbox("Entry", [f"#{r.ID} · {r['Decision question'][:60]}" for _, r in df.iterrows()])
            if st.button("Delete permanently"):
                history.delete(sel.split(" · ")[0][1:])
                st.rerun()
    with st.expander("⬆️ Restore / merge a downloaded history file"):
        upf = st.file_uploader("decision_history CSV", type=["csv"], key="hist_up")
        if upf is not None and st.button("Merge into history"):
            try:
                n = history.merge_upload(upf)
                st.success(f"Added {n} decision(s).")
            except Exception as e:  # noqa: BLE001
                st.error(f"Could not read the file: {e}")
    st.caption("History is saved on the app's server. Free hosting can reset after updates or long inactivity — "
               "download the CSV regularly and upload it here to restore.")


# --------------------------------------------------------------------------- #
# Guide
# --------------------------------------------------------------------------- #
def render_guide():
    _hero("📖 How to use DecisionCheck AI", "A 2-minute guide for managers — no statistics background needed.")
    st.markdown("""
### The workflow
1. **🔍 Check a decision** — upload your data, type the decision you're considering, let the agent plan and run the analysis, review the evidence, then record what *you* decide.
2. **Implement it** (or run a small pilot in one region / store / team).
3. **📈 Did it work?** — upload the newer data and check whether the change really made a difference.
4. **🗂️ Decision history** — every decision, its evidence and its outcome in one place.

### Writing a good decision question
Mention **the action**, **the result you want**, and (optionally) **the segment** and **your reason**:

| Good ✅ | Too vague ❌ |
|---|---|
| Should we **increase the discount** on **Product A** because its **sales are declining**? | What should we do about sales? |
| Should we **increase marketing spend** to **grow revenue** in the **West region**? | Is marketing good? |
| Should we **move more orders to the Online channel** to **improve profit**? | Online or store? |

Use words close to your column names. If the agent misunderstands, simply correct it in the interpretation panel.

### Your data
- One row per transaction / employee / customer / week, one column per measure. CSV or Excel (any sheet).
- A **date column** enables trend checks; **category columns** (product, region, channel) enable comparisons.
- At least **30 rows** recommended; 100+ is better.
- Messy files are fine: text numbers like "₹1,234", title rows, "Total" rows and duplicate rows are handled automatically.

### Reading the results
| Term | Plain meaning |
|---|---|
| **p-value** | Chance of seeing this pattern if nothing real was going on. Below **0.05** = "statistically significant" (unlikely to be luck). |
| **Correlation (r)** | How closely two measures move together, from −1 to +1. 0.1 weak, 0.3 moderate, 0.5+ strong. |
| **Regression** | The effect of one factor *while holding the others constant* — fairer than a simple correlation. |
| **R²** | Share of the ups and downs explained by the model (0–100%). |
| **t-test / ANOVA** | Is the average really different between groups (2 groups / 3+ groups)? |
| **Effect size** | How *big* the difference is — significant is not the same as important. |
| **Difference-in-differences** | Compares the pilot group's change with the change in other groups, removing what would have happened anyway. |

### What this tool will not do
- It will not make or carry out the decision — you are accountable.
- It cannot prove cause and effect from historical data — use a pilot to be sure.
- Don't use it for decisions about individual people (hiring, pay, credit, health).
- Use synthetic or approved data; don't upload confidential data to public AI services without permission.
""")
