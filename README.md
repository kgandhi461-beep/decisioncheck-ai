# DecisionCheck AI
**An AI-powered statistical evidence checker for business decisions**
Khushi Gandhi · Business Analytics · Agentic and Generative AI Applications for Business

A manager uploads any Excel/CSV dataset and types a proposed decision, for example
*"Should we increase the discount on Product A because its sales are declining?"*
An AI agent interprets the question, plans suitable statistical tests, runs them, adds follow-up tests when results call for it, sorts the evidence into **supports / challenges / limitations**, explains it in plain business language, and then **hands the final decision to a human**.

It is **generic**: it never assumes column names, so it works on sales, HR, marketing, operations or finance data.

**Built for repeated, real use, not a one-time demo:**
| Page / feature | What it does |
|---|---|
| 🔍 **Check a decision** | Upload data → describe decision → agent plans & runs tests → evidence for/against → you decide |
| 🔮 **What-if** | "If we raise discount by 5 points, sales may change by about X (range Y–Z), profit by …" |
| 🙋 **Ask a question** | Chat with your data: the agent picks the right test, runs it and answers ("Which region is most profitable?") |
| 📈 **Did it work?** | After implementing a decision/pilot, upload new data → difference-in-differences tells you if it really worked |
| 🗂️ **Decision history** | Every decision, evidence, reasoning and result saved; searchable; download/restore as CSV |
| 📖 **How to use** | Built-in guide and plain-English glossary for managers |
| Real-world files | Multi-sheet Excel (pick the sheet), title rows, "Total" rows, "₹1,234" text numbers, mixed date formats |
| 🖨️ Report | Downloadable evidence report with a Print / Save-as-PDF button |

---

## Part A — Put the app online for free (no coding needed, about 20 minutes)

You need three free accounts: **GitHub** (stores the files), **Streamlit Community Cloud** (runs the website) and **Google AI Studio** (free AI key, optional).

### Step 1. Put the files on GitHub
1. Go to **github.com** and sign up (free).
2. Click the **+** at the top right → **New repository**.
3. Name it `decisioncheck-ai`, choose **Public**, and click **Create repository**.
4. On the next page click **"uploading an existing file"**.
5. Unzip `decisioncheck_ai.zip` on your computer. Open the folder, select **everything inside it** (`app.py`, `ui_extra.py`, the `core`, `data`, `tests`, `docs`, `.streamlit` folders, `requirements.txt`, `README.md`) and drag them into the GitHub page.
   - On a Mac, press `Cmd + Shift + .` in Finder to see the hidden `.streamlit` folder. On Windows, enable *View → Hidden items*. If it won't upload, skip it; the app still works.
6. Click **Commit changes**.

### Step 2. Get a free AI key (optional, but recommended for the "Generative AI" part)
1. Go to **aistudio.google.com** and sign in with your Google account.
2. Click **Get API key** → **Create API key**. Copy the key (a long code).
3. Keep it private. Never put it inside a GitHub file.

> Without a key the app still works fully; explanations then come from a built-in rule engine. With a key, Gemini interprets the question and writes the explanation.
> Alternative free option: **console.groq.com** → API Keys.

### Step 3. Launch the website
1. Go to **share.streamlit.io** and click **Continue with GitHub**.
2. Click **Create app** → **Deploy a public app from GitHub**.
3. Repository: `your-username/decisioncheck-ai` · Branch: `main` · Main file path: `app.py`.
4. Click **Advanced settings** → in the **Secrets** box paste:
   ```
   GEMINI_API_KEY = "paste-your-key-here"
   ```
5. Click **Deploy**. Wait 2–4 minutes. You'll get a link like `https://decisioncheck-ai.streamlit.app` to share with your faculty.

**If something goes wrong:** open *Manage app* (bottom right of your app) to see the error log. Most issues come from a file missing on GitHub. Check that `requirements.txt` and the `core` folder are there.

> Free Streamlit apps "sleep" after a few days without visitors. Open the link once before your viva and click *"Yes, get this app back up"*.

### (Optional) Run on your own computer
Install Python 3.11+ from python.org, then in a terminal inside the folder:
```
pip install -r requirements.txt
streamlit run app.py
```

---

## Part B — How to demo it in your viva (5 minutes)

1. Click **🛒 Retail sales** to load the sample data. Point out the **data-quality score** and open *Data profile & quality checks*.
2. Click the example *"Should we increase discount pct for Product A because its sales is declining?"*, or type the proposal's own example.
3. Click **🧠 Let the agent interpret & plan**. Show that the agent identified the **outcome (Sales)**, **lever (Discount)**, **direction (increase)**, **focus (Product A)** and **premise (declining)**, and that **you can correct it**. This is the first human checkpoint.
4. Show the **analysis plan** with the reason for each test. You can untick tests.
5. Click **▶️ Run DecisionCheck**. Expected story from the sample data:
   - ✅ Product A's decline **is real** and **specific to Product A** (the business overall is flat).
   - ❌ Discount shows **no positive link to sales**, and is linked to **much lower profit**.
   - ⇒ *"Evidence mostly challenges the decision"*. A pilot or a different action is wiser.
6. Open the **🧭 Agent trace** tab to show the agentic loop: Perceive → Interpret → Plan → Act → **Observe (follow-up tests it decided to add)** → Evaluate → Explain → Verify → Hand-off.
7. Open **💬 AI explanation** and point to the **grounding check** (every number the AI writes is checked against the Python results).
8. In **Step 4**, record a human decision with reasoning, then **download the report**.
9. To prove it is **generic**, click **👥 HR performance** and ask *"Should we increase training hours to improve performance score?"*

10. Open the **🔮 What-if** tab: a +5-point discount shows roughly **+3% quantity, −4% sales, −30% profit**, with ranges.
11. Open **🙋 Ask a question** and click *"Which region has the highest sales?"* or type *"Why is sales changing?"*. Point out the caption showing which tool the agent chose.
12. Click **📈 Did it work?** in the sidebar → **🧪 Sample pilot results** → link your recorded decision → Date column `Order_Date`, change date **15 Oct 2025**, pilot group `Region` = **North** → tick Quantity, Sales, Profit → **Check the results**.
    Story: quantity **+15%** (it worked), sales **no real change**, profit **−39%** (it backfired). The app also warns that a simple before/after comparison would have been fooled by the festive season. This is a strong point for your viva.
13. Click **🗂️ Decision history** to show the decision with its saved result, and download it as a CSV.

Other good demos: *"Should we cut the price of Product C because its sales are falling?"* (the data shows Product C is actually **rising**, so the premise is false) and uploading a file from `data/test_datasets/`.

---

## Part C — What's inside (for your report)

### Architecture
```
app.py                 Streamlit web interface: Check a decision (4 steps, What-if, Ask a question tabs)
ui_extra.py            Pages: Did it work?, Decision history, How to use
core/data_quality.py   Load CSV/Excel, detect column types, quality checks, light cleaning
core/planner.py        Agent brain: interpret decision (AI or rules) → validated frame → analysis plan → follow-ups
core/analysis.py       Deterministic statistical tools (Python only, never the AI)
core/evidence.py       Rules that turn results into supporting / challenging / limitation evidence + confidence
core/agent.py          Orchestrator loop, AI explanation, grounding (hallucination) check, trace log
core/llm.py            Gemini / Groq connector with automatic fallback to the rule engine
core/report.py         Downloadable HTML evidence report (print / save as PDF)
core/whatif.py         What-if impact estimator with likely ranges
core/chat.py           Ask-a-question agent: AI/rules choose a tool, Python runs it, answer is grounding-checked
core/impact.py         "Did it work?": difference-in-differences, before/after, test vs control
core/history.py        Saved decision history (CSV) with result tracking, download & restore
data/                  Synthetic datasets + generator script
tests/                 14 automated test cases → tests/test_results.md
```

### Agentic workflow (TO-BE process)
| Agent phase | What happens | Who does it |
|---|---|---|
| Perceive | Profile data, detect types, quality score, clean duplicates and labels | Python |
| Interpret | Map decision → outcome, lever, direction, focus segment, premise | **Gen AI** (validated) or rules |
| Human check 1 | User reviews and corrects interpretation and plan | **Human** |
| Plan | Choose tests that fit the question and the available columns | Agent rules |
| Act | Run statistics and charts | Python (SciPy, statsmodels) |
| Observe | Add follow-ups: market-wide check, outlier-robust re-test, all-data re-test | Agent rules |
| Evaluate | Classify evidence; compute balance and confidence | Python |
| Explain | Plain-English summary, questions, next steps | **Gen AI** |
| Verify | Grounding check of AI numbers vs computed results | Python |
| Human check 2 | Final decision + mandatory reasoning + accountability tick → audit log & report | **Human** |
| Follow-up | What-if estimates and follow-up questions (agent chooses and runs a tool each time) | Gen AI + Python |
| Close the loop | After implementation, measure the real effect and save it to the decision's history | Python + Human |

### Statistical methods
| Method | Used for | Tool |
|---|---|---|
| Descriptive statistics (mean, median, SD, CV, skewness) | Size and spread of the outcome | `descriptive` |
| Trend analysis (linear trend + Spearman, auto day/week/month/quarter) | Verify "sales are declining" premises | `trend` |
| Pearson & Spearman correlation with 95% CI | Lever ↔ outcome, side-effects | `correlation` |
| Correlation heatmap | All numeric relationships | `correlation_matrix` |
| Multiple OLS regression (robust HC3 SE, VIF, residual normality, leakage control) | Lever effect holding other factors constant | `regression` |
| Welch t-test + Mann-Whitney (2 groups), one-way ANOVA + Kruskal-Wallis + Tukey HSD (3+ groups), Levene | Group differences | `group_compare` |
| Segment vs rest (Welch t-test, Cohen's d) | Is the focus segment different? | `segment_vs_rest` |
| Outlier detection (IQR + z-score, mean-shift impact) | Robustness | `outliers` |
| Chi-square + Cramér's V | Two categorical variables | `chi_square` |
| Difference-in-differences (OLS with interaction, HC3) + pre-trend check | Did the pilot work? | `impact.evaluate` |
| Regression-based what-if with 95% ranges | Expected impact of a change | `whatif` |

Significance level α = 0.05. Effect sizes are always reported alongside p-values.

### Responsible AI controls
- **Human-in-the-loop twice**: interpretation/plan review, and a mandatory final decision with reasoning and an accountability checkbox.
- **No automatic execution** of any business action.
- **Calculations ≠ interpretation**: all numbers come from Python; the AI only explains them.
- **Validation of AI output**: any column or value the AI proposes that isn't in the data is rejected (tested in TC14).
- **Grounding check**: every number in the AI's explanation is matched to computed results; unmatched numbers are flagged.
- **Privacy**: only column names, summary statistics and results are sent to the AI, never raw rows. The API key is kept in Secrets, not in code.
- **Graceful fallback**: works with no AI or if the AI fails.
- **Visible limitations**: correlation ≠ causation, small samples, outliers, missing variables, data issues.
- **Don't use for**: decisions about individual people (hiring, credit, health), tiny samples, or confidential data without approval.

### Testing
Run `python tests/run_test_cases.py`. Results are saved to `tests/test_results.md`.
The 19 cases cover normal cases, a false premise, a non-significant result, a different domain (HR), missing values, duplicates, wrong data types (₹ text, dd/mm/yyyy), a small sample, random no-relationship data, an unanswerable question, vague input, an AI hallucination guard, a messy multi-sheet Excel report, the what-if estimator, the question-answering agent's tool choice, the pilot checker (where difference-in-differences catches what before/after misses), and saving and restoring decision history.

### Known limitations
- Observational data only: the app cannot prove causation. It recommends pilots or A/B tests.
- The rule-based interpreter works best when the question uses words similar to the column names. The AI interpreter and the editable interpretation panel cover the rest.
- Time-series analysis is a linear trend test, not a full forecasting model. Strong seasonality is flagged rather than modelled.
- Decision history is saved on the server. Free hosting may reset it after app updates or long inactivity, so download the CSV now and then (one click on the history page) and re-upload it to restore.
- What-if estimates assume a straight-line relationship and are only as good as past data; changes far outside past experience are flagged.
- Free AI tiers have rate limits, and Google may use free-tier inputs to improve its products, so use synthetic or public data.
