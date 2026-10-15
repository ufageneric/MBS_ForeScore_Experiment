#!/usr/bin/env python3
"""Balance-weighted pool average of ForeScore LifeOfLoanDefault.

For each RMBS deal i and loan j in that deal:

    sum_j (current_balance_ij * LifeOfLoanDefault_ij) / sum_j current_balance_ij

LifeOfLoanDefault is the ForeScore value for the loan's ZIP and origination
month. A loan with no positive current balance, no origination date, or no
matching ForeScore row is left out of both sums.

Usage:
  python forescore_pool_avg.py
  python forescore_pool_avg.py --deals edgar_rmbs/deals_rmbs_final.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCORE_COLS = ["forescore_default", "forescore_n_loans", "forescore_balance"]


def parse_orig_date(series):
    """Tape dates arrive as M/D/YYYY or YYYYMMDD."""
    raw = series.fillna("").astype(str).str.strip()
    parsed = pd.to_datetime(raw, errors="coerce")
    ymd = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    return parsed.fillna(ymd)


def load_forescore(path, years):
    parts = []
    usecols = ["Year", "Month", "zipcode", "LifeOfLoanDefault"]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        chunk = chunk.loc[chunk["Year"].isin(years)]
        if len(chunk):
            parts.append(chunk)
    if not parts:
        return pd.DataFrame(columns=["Year", "Month", "zip_i", "LifeOfLoanDefault"])
    scores = pd.concat(parts, ignore_index=True)
    scores["zip_i"] = pd.to_numeric(scores["zipcode"], errors="coerce")
    scores["LifeOfLoanDefault"] = pd.to_numeric(scores["LifeOfLoanDefault"], errors="coerce")
    scores = scores.dropna(subset=["zip_i", "LifeOfLoanDefault"])
    return scores[["Year", "Month", "zip_i", "LifeOfLoanDefault"]]


def pool_averages(loans, scores):
    """One row per deal. Weight is current balance."""
    frame = loans.copy()
    frame["orig_dt"] = parse_orig_date(frame["orig_date"])
    frame["bal"] = pd.to_numeric(frame["current_balance"], errors="coerce")
    frame["zip_i"] = pd.to_numeric(frame["zip"], errors="coerce")
    use = frame.loc[frame["orig_dt"].notna() & frame["bal"].gt(0) & frame["zip_i"].notna()].copy()
    use["Year"] = use["orig_dt"].dt.year.astype(int)
    use["Month"] = use["orig_dt"].dt.month.astype(int)

    matched = use.merge(scores, on=["Year", "Month", "zip_i"], how="left")
    ok = matched.loc[matched["LifeOfLoanDefault"].notna()]
    num = (ok["bal"] * ok["LifeOfLoanDefault"]).groupby(ok["accession"]).sum()
    den = ok.groupby("accession")["bal"].sum()
    cnt = ok.groupby("accession").size()
    out = pd.DataFrame({
        "forescore_default": num / den,
        "forescore_n_loans": cnt,
        "forescore_balance": den,
    }).reset_index()
    return out, len(use), int(matched["LifeOfLoanDefault"].isna().sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loans", default="edgar_rmbs/loans_rmbs_final.csv")
    ap.add_argument("--scores", default="edgar_rmbs/UFA_LS_FSZips_2605a.csv")
    ap.add_argument("--deals", default="edgar_rmbs/deals_rmbs_final.csv")
    ap.add_argument("--out", default="edgar_rmbs/deals_rmbs_final.csv")
    args = ap.parse_args()

    print(f"Loading {args.loans}")
    loans = pd.read_csv(
        args.loans,
        usecols=["accession", "zip", "orig_date", "current_balance"],
        dtype={"accession": str},
        low_memory=False,
    )
    dated = parse_orig_date(loans["orig_date"])
    years = set(dated.dt.year.dropna().astype(int))
    print(f"  {len(loans):,} loans, {dated.notna().sum():,} with an origination date")

    print(f"Loading {args.scores}")
    scores = load_forescore(args.scores, years)
    print(f"  {len(scores):,} ZIP-month scores for origination years {min(years)}-{max(years)}")

    averages, n_weighted, n_unmatched = pool_averages(loans, scores)
    print(f"  {n_weighted:,} loans with date and current balance, {n_unmatched:,} unmatched")

    deals = pd.read_csv(args.deals, dtype={"accession": str})
    deals = deals.drop(columns=[c for c in SCORE_COLS if c in deals.columns])
    deals = deals.merge(averages, on="accession", how="left")

    out = Path(args.out)
    deals.to_csv(out, index=False)
    n_scored = int(deals["forescore_default"].notna().sum())
    print(f"Wrote {out} ({n_scored:,} of {len(deals):,} deals with a ForeScore default)")
    show = deals.loc[deals["forescore_default"].notna(), ["issuer", "forescore_n_loans", "forescore_default"]]
    print(show.to_string(index=False))


if __name__ == "__main__":
    main()
