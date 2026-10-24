"""
Data loading, column-type detection and data-quality checks.

Works on ANY tabular dataset: it never assumes column names.
"""
from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _read_bytes(uploaded) -> tuple[str, bytes]:
    name = getattr(uploaded, "name", str(uploaded)).lower()
    if hasattr(uploaded, "getvalue"):
        return name, uploaded.getvalue()
    if isinstance(uploaded, (bytes, bytearray)):
        return name, bytes(uploaded)
    with open(uploaded, "rb") as fh:
        return name, fh.read()


def list_sheets(uploaded) -> list[str]:
    """Sheet names of an Excel file (empty list for CSV)."""
    name, raw = _read_bytes(uploaded)
    if not name.endswith((".xlsx", ".xlsm", ".xls")):
        return []
    try:
        return [str(s) for s in pd.ExcelFile(io.BytesIO(raw)).sheet_names]
    except Exception:  # noqa: BLE001
        return []


def _fix_header(df: pd.DataFrame) -> pd.DataFrame:
    """If a report has title rows above the real table, find the real header row."""
    unnamed = sum(str(c).lower().startswith("unnamed") for c in df.columns)
    if unnamed < max(2, len(df.columns) // 2) or len(df) < 3:
        return df
    for i in range(min(15, len(df))):
        row = df.iloc[i]
        if row.notna().mean() >= 0.7 and row.astype(str).str.match(r"^[A-Za-z]").mean() >= 0.6:
            new = df.iloc[i + 1:].copy()
            new.columns = [str(v).strip() if pd.notna(v) else f"Column_{j+1}" for j, v in enumerate(row)]
            return new.reset_index(drop=True)
    return df


def load_file(uploaded, sheet: str | None = None) -> pd.DataFrame:
    """Load a CSV / Excel file (Streamlit UploadedFile, path or bytes). `sheet` picks an Excel sheet."""
    name, raw = _read_bytes(uploaded)

    if name.endswith((".xlsx", ".xlsm", ".xls")):
        df = pd.read_excel(io.BytesIO(raw), sheet_name=sheet if sheet else 0)
        df = _fix_header(df)
    else:
        df = None
        for enc in ("utf-8", "utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
                break
            except Exception:  # noqa: BLE001
                continue
        if df is None:
            raise ValueError("Could not read the file. Please upload a valid CSV or Excel file.")

    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not c.lower().startswith("unnamed")] or df.columns]
    df = df.dropna(how="all").dropna(axis=1, how="all")
    # drop "Total" / "Grand total" summary rows that Excel reports often contain
    first = df.columns[0]
    if not pd.api.types.is_numeric_dtype(df[first]):
        df = df[~df[first].astype(str).str.strip().str.lower().isin(["total", "grand total", "totals", "sum"])]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Column typing
# --------------------------------------------------------------------------- #
_CURRENCY = re.compile(r"[₹$€£,%\s]")


def _try_numeric(s: pd.Series) -> pd.Series | None:
    if pd.api.types.is_numeric_dtype(s):
        return s
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        cleaned = s.astype(str).str.replace(_CURRENCY, "", regex=True).replace({"": np.nan, "nan": np.nan, "None": np.nan})
        conv = pd.to_numeric(cleaned, errors="coerce")
        non_null = s.notna().sum()
        if non_null and conv.notna().sum() / non_null >= 0.9:
            return conv
    return None


def _try_datetime(s: pd.Series, name: str) -> pd.Series | None:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):
        # Years like 2021, 2022 are numeric; only treat as date if named so.
        if re.search(r"\b(year|yr)\b", name.lower()) and s.dropna().between(1900, 2100).all():
            return pd.to_datetime(s.astype("Int64").astype(str), format="%Y", errors="coerce")
        return None
    if s.dtype == object or pd.api.types.is_string_dtype(s):
        # plain numbers (e.g. prices stored as text) are not dates unless the column name says so
        if not re.search(r"date|time|day|month|period|week", name.lower()) and _try_numeric(s) is not None:
            return None
        sample = s.dropna().astype(str).head(200)
        if sample.empty or not sample.str.contains(r"\d").mean() > 0.8:
            return None
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            conv = pd.to_datetime(s, errors="coerce", dayfirst=False)
            if conv.notna().sum() < 0.9 * s.notna().sum():
                conv2 = pd.to_datetime(s, errors="coerce", dayfirst=True)
                if conv2.notna().sum() > conv.notna().sum():
                    conv = conv2
        if s.notna().sum() and conv.notna().sum() / s.notna().sum() >= 0.9:
            return conv
    return None


@dataclass
class Schema:
    numeric: list[str] = field(default_factory=list)
    categorical: list[str] = field(default_factory=list)
    datetime: list[str] = field(default_factory=list)
    identifier: list[str] = field(default_factory=list)
    text: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {k: list(v) for k, v in self.__dict__.items()}


_ID_HINT = re.compile(r"(^id$|_id$|^id_|\bid\b|code$|number$|^no$|serial|invoice|order.?no|txn|transaction)", re.I)


def infer_schema(df: pd.DataFrame) -> tuple[pd.DataFrame, Schema]:
    """Return a type-cleaned copy of df and its schema."""
    df = df.copy()
    sch = Schema()
    n = len(df)
    for col in df.columns:
        s = df[col]
        dt = _try_datetime(s, col)
        if dt is not None:
            df[col] = dt
            sch.datetime.append(col)
            continue
        num = _try_numeric(s)
        nunique = s.nunique(dropna=True)
        if num is not None:
            df[col] = num
            is_int = np.allclose(num.dropna() % 1, 0) if num.notna().any() else False
            if _ID_HINT.search(col) and (nunique > 0.9 * n or is_int):
                sch.identifier.append(col)
            elif nunique <= 2 and n > 10:
                sch.categorical.append(col)  # binary flags
            elif is_int and nunique <= 10 and n >= 50 and not re.search(r"(qty|quantity|units|count|sales|revenue|amount|price|cost)", col, re.I):
                sch.categorical.append(col)  # rating-like codes
            else:
                sch.numeric.append(col)
            continue
        # strings
        if nunique == 0:
            sch.text.append(col)
        elif _ID_HINT.search(col) and nunique > 0.5 * n:
            sch.identifier.append(col)
        elif nunique > 0.9 * n and n > 30:
            avg_len = s.dropna().astype(str).str.len().mean()
            (sch.text if avg_len > 25 else sch.identifier).append(col)
        elif nunique <= max(50, 0.5 * n) or n <= 30:
            df[col] = s.astype(str).str.strip().where(s.notna())
            sch.categorical.append(col)
        else:
            sch.text.append(col)
    return df, sch


# --------------------------------------------------------------------------- #
# Data quality
# --------------------------------------------------------------------------- #
def quality_report(raw: pd.DataFrame, df: pd.DataFrame, sch: Schema) -> dict:
    """Profile the dataset and list quality issues with severity."""
    issues: list[dict] = []
    n_rows, n_cols = df.shape

    dup = int(raw.duplicated().sum())
    if dup:
        issues.append({"severity": "warning", "check": "Duplicate rows",
                       "detail": f"{dup} exact duplicate row(s) found ({dup / max(n_rows, 1):.1%}). They were removed before analysis."})

    miss = df.isna().mean()
    for col, frac in miss[miss > 0].sort_values(ascending=False).items():
        sev = "critical" if frac > 0.4 else "warning" if frac > 0.05 else "info"
        issues.append({"severity": sev, "check": "Missing values",
                       "detail": f"'{col}' has {frac:.1%} missing values."})

    for col in raw.columns:
        if col in sch.numeric and not pd.api.types.is_numeric_dtype(raw[col]):
            bad = int(pd.to_numeric(raw[col].astype(str).str.replace(_CURRENCY, "", regex=True), errors="coerce").isna().sum() - raw[col].isna().sum())
            msg = f"'{col}' was stored as text; converted to numbers"
            if bad > 0:
                msg += f" ({bad} value(s) could not be converted and were treated as missing)"
            issues.append({"severity": "warning" if bad else "info", "check": "Data type", "detail": msg + "."})

    for col in sch.numeric:
        s = df[col].dropna()
        if len(s) and s.nunique() == 1:
            issues.append({"severity": "warning", "check": "Constant column", "detail": f"'{col}' has the same value in every row, so it cannot explain variation."})
        if len(s) and (s < 0).any() and re.search(r"(qty|quantity|units|price|sales|revenue|amount|discount)", col, re.I):
            issues.append({"severity": "warning", "check": "Negative values",
                           "detail": f"'{col}' contains {(s < 0).sum()} negative value(s) — could be returns or data-entry errors."})

    for col in sch.categorical:
        vals = df[col].dropna().astype(str)
        norm = vals.str.lower().str.strip()
        if norm.nunique() < vals.nunique():
            issues.append({"severity": "info", "check": "Inconsistent labels",
                           "detail": f"'{col}' has labels that differ only by spelling/case (e.g. 'North' vs 'north')."})

    if n_rows < 30:
        issues.append({"severity": "critical" if n_rows < 10 else "warning", "check": "Small sample",
                       "detail": f"Only {n_rows} rows. Statistical tests will have low power; treat results with caution."})
    if not sch.numeric:
        issues.append({"severity": "critical", "check": "No numeric columns",
                       "detail": "No numeric measures were found, so most statistical analysis is not possible."})

    sev_w = {"critical": 25, "warning": 8, "info": 2}
    score = max(0, 100 - sum(sev_w[i["severity"]] for i in issues))

    profile = []
    for col in df.columns:
        kind = next((k for k, v in sch.as_dict().items() if col in v), "other")
        s = df[col]
        row = {"Column": col, "Type": kind, "Missing %": round(s.isna().mean() * 100, 1), "Unique": int(s.nunique())}
        if kind == "numeric":
            row.update({"Min": f"{s.min():,.4g}", "Mean": f"{s.mean():,.4g}", "Max": f"{s.max():,.4g}"})
        elif kind == "datetime":
            row.update({"Min": str(s.min())[:10], "Max": str(s.max())[:10]})
        elif kind == "categorical":
            top = s.value_counts().head(3).index.astype(str).tolist()
            row.update({"Top values": ", ".join(top)})
        profile.append(row)

    return {"rows": n_rows, "columns": n_cols, "duplicates": dup, "score": score,
            "issues": issues, "profile": pd.DataFrame(profile)}


def clean_for_analysis(df: pd.DataFrame, sch: Schema) -> pd.DataFrame:
    """Light, transparent cleaning: drop exact duplicates and normalise label case."""
    out = df.drop_duplicates().copy()
    for col in sch.categorical:
        if not pd.api.types.is_numeric_dtype(out[col]):
            s = out[col].astype(str).str.strip()
            # unify case variants to the most common spelling
            canon = s.groupby(s.str.lower()).agg(lambda x: x.value_counts().index[0])
            out[col] = s.str.lower().map(canon).where(out[col].notna())
    return out
