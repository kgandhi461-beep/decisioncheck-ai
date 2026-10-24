"""
Decision history (audit log).

Saved to decision_history.csv next to the app, so it survives page refreshes.
On free cloud hosting the server can be reset, so users can also DOWNLOAD the
history and UPLOAD it again later - nothing is lost.
"""
from __future__ import annotations

import datetime as dt
import uuid
from pathlib import Path

import pandas as pd

FILE = Path(__file__).resolve().parent / "decision_history.csv"
COLUMNS = ["ID", "Date", "Dataset", "Decision question", "Outcome", "Lever", "Focus", "Evidence verdict",
           "Analysis confidence", "Supporting", "Challenging", "Human decision", "Rationale", "Decided by",
           "Follow-up status", "Result check"]


def load() -> pd.DataFrame:
    try:
        if FILE.exists():
            df = pd.read_csv(FILE, dtype=str).fillna("")
            return df.reindex(columns=COLUMNS, fill_value="")
    except Exception:  # noqa: BLE001
        pass
    return pd.DataFrame(columns=COLUMNS)


def _save(df: pd.DataFrame) -> bool:
    try:
        df.to_csv(FILE, index=False)
        return True
    except Exception:  # noqa: BLE001
        return False


def record(out: dict, human: dict, dataset: str) -> dict:
    fr, ev = out["frame"], out["evidence"]
    needs_check = human["choice"] in ("Approve as proposed", "Approve with modifications", "Run a pilot / A-B test first")
    entry = {
        "ID": uuid.uuid4().hex[:6].upper(), "Date": human["time"], "Dataset": dataset, "Decision question": fr.decision,
        "Outcome": fr.target or "", "Lever": (fr.lever or "") + (f" = {fr.lever_value}" if fr.lever_value else ""),
        "Focus": ", ".join(f"{k} = {v}" for k, v in fr.focus.items()),
        "Evidence verdict": ev["verdict"], "Analysis confidence": f"{ev['confidence_label']} ({ev['confidence']})",
        "Supporting": str(sum(e["stance"] == "supports" for e in ev["items"])),
        "Challenging": str(sum(e["stance"] == "challenges" for e in ev["items"])),
        "Human decision": human["choice"], "Rationale": human["rationale"], "Decided by": human.get("name", ""),
        "Follow-up status": "Check results later" if needs_check else "Closed", "Result check": "",
    }
    df = pd.concat([load(), pd.DataFrame([entry])], ignore_index=True)
    _save(df)
    return entry


def update_result(entry_id: str, text: str) -> None:
    df = load()
    m = df["ID"] == entry_id
    df.loc[m, "Result check"] = text
    df.loc[m, "Follow-up status"] = "Checked " + dt.datetime.now().strftime("%d %b %Y")
    _save(df)


def merge_upload(uploaded) -> int:
    new = pd.read_csv(uploaded, dtype=str).fillna("").reindex(columns=COLUMNS, fill_value="")
    cur = load()
    df = pd.concat([cur, new[~new["ID"].isin(cur["ID"])]], ignore_index=True)
    _save(df)
    return len(df) - len(cur)


def delete(entry_id: str) -> None:
    df = load()
    _save(df[df["ID"] != entry_id])
