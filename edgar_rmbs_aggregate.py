#!/usr/bin/env python3
"""Build a unique-loan extract and zip/state aggregates from EDGAR RMBS tapes.

Amended FWPs repeat the same pool, so zip counts would be inflated without
deduping first. Unique loans keep one row per (issuer, loan_id), preferring
the copy that has zip / FICO / LTV.

Usage:
  python edgar_rmbs_aggregate.py
  python edgar_rmbs_aggregate.py --loans edgar_rmbs/loans_2006_2007.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

NUM_COLS = ["fico", "cltv", "ltv", "dti", "current_balance", "orig_balance", "rate"]
RANGES = {
    "fico": (300, 850),
    "ltv": (1, 150),
    "cltv": (1, 200),
    "dti": (0, 100),
    "rate": (1, 25),
    "orig_balance": (1_000, 10_000_000),
    "current_balance": (1_000, 10_000_000),
}


def parse_zip(series):
    """5-digit ZIP. CSV often stores 37385 as 37385.0."""
    z = series.astype(str).str.replace(r"\.0+$", "", regex=True)
    z = z.str.replace(r"\D", "", regex=True)
    z = z.where(z.str.len() != 9, z.str[:5]).str.zfill(5)
    zn = pd.to_numeric(z, errors="coerce")
    return z.where(z.str.fullmatch(r"\d{5}") & zn.between(501, 99950))


def clean_loans(df):
    out = df.copy()
    out["filing_date"] = pd.to_datetime(out["filing_date"], errors="coerce")
    out["year"] = out["filing_date"].dt.year
    out["zip"] = parse_zip(out["zip"])
    st = out["state"].astype(str).str.strip().str.upper()
    out["state"] = st.where(st.str.fullmatch(r"[A-Z]{2}"))
    for col in NUM_COLS:
        if col not in out.columns:
            continue
        v = pd.to_numeric(out[col], errors="coerce")
        lo, hi = RANGES[col]
        out[col] = v.where(v.between(lo, hi))
    lid = out["loan_id"].astype(str).str.strip()
    out["loan_id"] = lid.where(~lid.isin(["", "nan", "None", "<NA>"]))
    return out


def unique_loans(df):
    """One row per loan. Prefer the copy with zip/FICO/LTV; filing date breaks ties.

    Latest-FWP-only is a bad rule here: later amendments are often thinner than
    the original tape and would drop zip coverage.
    """
    df = df.copy()
    epoch = pd.Timestamp("1970-01-01")
    days = (df["filing_date"].fillna(epoch) - epoch).dt.days.clip(lower=0)
    df["_keep"] = (
        df["zip"].notna().astype(int) * 8
        + df["fico"].notna().astype(int) * 4
        + df["ltv"].notna().astype(int) * 2
        + df["rate"].notna().astype(int)
        + days / 100_000
    )
    df = df.sort_values("_keep")
    has_id = df["loan_id"].notna()
    with_id = df.loc[has_id].drop_duplicates(["issuer", "loan_id"], keep="last")
    no_id = df.loc[~has_id].drop_duplicates(
        ["issuer", "zip", "fico", "ltv", "orig_balance", "rate"], keep="last"
    )
    out = pd.concat([with_id, no_id], ignore_index=True)
    return out.drop(columns=["_keep"]).sort_values(["year", "issuer"]).reset_index(drop=True)


def _agg(frame, keys):
    bal = frame["orig_balance"].where(frame["orig_balance"].notna(), frame["current_balance"])
    frame = frame.assign(_bal=bal)
    g = frame.groupby(keys, dropna=False)
    out = g.agg(
        n_loans=("accession", "size"),
        n_fico=("fico", "count"),
        n_ltv=("ltv", "count"),
        n_rate=("rate", "count"),
        mean_fico=("fico", "mean"),
        med_fico=("fico", "median"),
        mean_ltv=("ltv", "mean"),
        med_ltv=("ltv", "median"),
        mean_cltv=("cltv", "mean"),
        med_cltv=("cltv", "median"),
        mean_dti=("dti", "mean"),
        med_dti=("dti", "median"),
        mean_rate=("rate", "mean"),
        med_rate=("rate", "median"),
        mean_balance=("_bal", "mean"),
        med_balance=("_bal", "median"),
        sum_balance=("_bal", "sum"),
    )
    return out.reset_index()


def zip_aggregates(df):
    z = df.loc[df["zip"].notna()].copy()
    yearly = _agg(z, ["zip", "year"])
    pooled = _agg(z, ["zip"])
    pooled.insert(1, "year", "2006-2007")
    return yearly, pooled


def state_aggregates(df):
    s = df.loc[df["state"].notna()].copy()
    return _agg(s, ["state", "year"])


def deal_aggregates(df):
    """One row per RMBS filing (accession): the loan pool in that FWP."""
    stats = _agg(df, ["accession"])
    meta = df.groupby("accession", dropna=False).agg(
        cik=("cik", "first"),
        issuer=("issuer", "first"),
        filing_date=("filing_date", "max"),
        year=("year", "max"),
        n_states=("state", "nunique"),
        n_zips=("zip", "nunique"),
    ).reset_index()
    out = meta.merge(stats, on="accession")
    front = ["accession", "cik", "issuer", "filing_date", "year", "n_loans",
             "n_states", "n_zips"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loans", default="edgar_rmbs/loans_2006_2007.csv")
    ap.add_argument("--out", default="edgar_rmbs")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.loans}")
    raw = pd.read_csv(args.loans, low_memory=False)
    print(f"  {len(raw):,} raw rows")
    raw["zip"] = parse_zip(raw["zip"])
    n_raw = len(raw)
    raw = raw.loc[raw["zip"].notna()].copy()
    print(f"  {len(raw):,} with valid zip (dropped {n_raw - len(raw):,})")
    raw.to_csv(args.loans, index=False)
    print(f"Wrote {args.loans}")

    cleaned = clean_loans(raw)
    uniq = unique_loans(cleaned)
    uniq = uniq.loc[uniq["zip"].notna()].copy()
    print(f"  {len(uniq):,} unique loans with zip")

    uniq_path = out / "loans_unique_2006_2007.csv"
    uniq.to_csv(uniq_path, index=False)
    print(f"Wrote {uniq_path}")

    zip_year, zip_pool = zip_aggregates(uniq)
    zip_year_path = out / "zip_rmbs_2006_2007.csv"
    zip_pool_path = out / "zip_rmbs_pooled.csv"
    zip_year.to_csv(zip_year_path, index=False)
    zip_pool.to_csv(zip_pool_path, index=False)
    print(f"Wrote {zip_year_path} ({len(zip_year):,} zip-years, "
          f"{zip_year['zip'].nunique()} zips)")
    print(f"Wrote {zip_pool_path} ({len(zip_pool):,} zips)")

    state = state_aggregates(uniq)
    state_path = out / "state_rmbs_2006_2007.csv"
    state.to_csv(state_path, index=False)
    print(f"Wrote {state_path} ({len(state):,} state-years)")

    deals = deal_aggregates(uniq)
    deals = deals.loc[deals["n_zips"] > 0].copy()
    deals_path = out / "deals_rmbs_2006_2007.csv"
    deals.to_csv(deals_path, index=False)
    print(f"Wrote {deals_path} ({len(deals):,} RMBS deals with zip)")

    print("\nUnique-loan coverage")
    for col in ["zip", "state", "fico", "ltv", "rate"]:
        print(f"  {col}: {uniq[col].notna().mean()*100:.1f}%")
    print("median fico / ltv / rate:",
          float(uniq.fico.median()), float(uniq.ltv.median()), float(uniq.rate.median()))


if __name__ == "__main__":
    main()
