#!/usr/bin/env python3
"""Balance-weighted pool averages of ForeScore life-of-loan default and prepayment.

For each RMBS deal i and loan j in that deal:

    sum_j (balance_ij * score_ij) / sum_j balance_ij

score is LifeOfLoanDefault or LifeOfLoanPrepayment for the loan's ZIP and
origination month. When the origination date is missing, the pool filing
date is used. When the current balance is missing, the original balance is
used. A loan with no positive balance, no usable date, or no matching
ForeScore row is left out of both sums. Each pool also gets the min,
25th percentile, median, 75th percentile, max, standard deviation, and
range of the loan-level scores in that pool.

Usage:
  python forescore_pool_avg.py
  python forescore_pool_avg.py --out output_file/deals_rmbs_final.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCORE_FIELDS = {
    "forescore_default": "LifeOfLoanDefault",
    "forescore_prepayment": "LifeOfLoanPrepayment",
}
DATE_FORMATS = ("%m/%d/%Y", "%Y%m%d", "%Y-%m-%d", "%m/%d/%y")


def parse_date(series):
    """Tape dates arrive as M/D/YYYY, YYYYMMDD, or YYYY-MM-DD."""
    raw = series.fillna("").astype(str).str.strip()
    parsed = pd.Series(pd.NaT, index=series.index)
    for fmt in DATE_FORMATS:
        parsed = parsed.fillna(pd.to_datetime(raw, format=fmt, errors="coerce"))
    return parsed


def load_forescore(path, years):
    parts = []
    factor_cols = list(SCORE_FIELDS.values())
    usecols = ["Year", "Month", "zipcode", *factor_cols]
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=1_000_000):
        chunk = chunk.loc[chunk["Year"].isin(years)]
        if len(chunk):
            parts.append(chunk)
    columns = ["Year", "Month", "zip_i", *factor_cols]
    if not parts:
        return pd.DataFrame(columns=columns)
    scores = pd.concat(parts, ignore_index=True)
    scores["zip_i"] = pd.to_numeric(scores["zipcode"], errors="coerce")
    for col in factor_cols:
        scores[col] = pd.to_numeric(scores[col], errors="coerce")
    scores = scores.dropna(subset=["zip_i"])
    return scores[columns]


def loan_weights(loans):
    """Origination date, else filing date. Current balance, else original balance."""
    frame = loans.copy()
    orig = parse_date(frame["orig_date"])
    filing = parse_date(frame["filing_date"])
    frame["score_dt"] = orig.fillna(filing)
    frame["used_filing_date"] = orig.isna() & frame["score_dt"].notna()

    current = pd.to_numeric(frame["current_balance"], errors="coerce")
    original = pd.to_numeric(frame["orig_balance"], errors="coerce")
    frame["bal"] = current.where(current.gt(0), original.where(original.gt(0)))
    frame["used_orig_balance"] = ~current.gt(0) & frame["bal"].notna()
    frame["zip_i"] = pd.to_numeric(frame["zip"], errors="coerce")
    return frame


def pool_averages(loans, scores):
    """One row per deal. Weight is current balance, or original balance."""
    frame = loan_weights(loans)
    use = frame.loc[frame["score_dt"].notna() & frame["bal"].notna() & frame["zip_i"].notna()].copy()
    use["Year"] = use["score_dt"].dt.year.astype(int)
    use["Month"] = use["score_dt"].dt.month.astype(int)

    matched = use.merge(scores, on=["Year", "Month", "zip_i"], how="left")
    pieces = []
    coverage = {"n_eligible": len(use)}
    for name, col in SCORE_FIELDS.items():
        ok = matched.loc[matched[col].notna()]
        num = (ok["bal"] * ok[col]).groupby(ok["accession"]).sum()
        den = ok.groupby("accession")["bal"].sum()
        cnt = ok.groupby("accession").size()
        grouped = ok.groupby("accession")[col]
        piece = pd.DataFrame({
            name: num / den,
            f"{name}_min": grouped.min(),
            f"{name}_p25": grouped.quantile(0.25),
            f"{name}_median": grouped.median(),
            f"{name}_p75": grouped.quantile(0.75),
            f"{name}_max": grouped.max(),
            f"{name}_std": grouped.std(),
        })
        piece[f"{name}_range"] = piece[f"{name}_max"] - piece[f"{name}_min"]
        if name == "forescore_default":
            piece["forescore_n_loans"] = cnt
            piece["forescore_balance"] = den
            coverage["n_unmatched"] = int(matched[col].isna().sum())
            coverage["n_filing_date"] = int(ok["used_filing_date"].sum())
            coverage["n_orig_balance"] = int(ok["used_orig_balance"].sum())
        pieces.append(piece)
    out = pd.concat(pieces, axis=1).reset_index()
    return out, coverage


def score_summary(scores):
    """Deal-level distribution of a pool average ForeScore."""
    values = scores.dropna()
    return pd.Series({
        "count": int(values.count()),
        "mean": values.mean(),
        "std": values.std(),
        "min": values.min(),
        "p25": values.quantile(0.25),
        "median": values.median(),
        "p75": values.quantile(0.75),
        "max": values.max(),
        "range": values.max() - values.min(),
    })


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--loans", default="edgar_rmbs/loans_rmbs_final.csv")
    ap.add_argument("--scores", default="edgar_rmbs/UFA_LS_FSZips_2605a.csv")
    ap.add_argument("--deals", default="edgar_rmbs/deals_rmbs_final.csv")
    ap.add_argument("--out", default="output_file/deals_rmbs_final.csv")
    args = ap.parse_args()

    print(f"Loading {args.loans}")
    loans = pd.read_csv(
        args.loans,
        usecols=["accession", "zip", "orig_date", "filing_date", "current_balance", "orig_balance"],
        dtype={"accession": str},
        low_memory=False,
    )
    weighted = loan_weights(loans)
    years = set(weighted.loc[weighted["score_dt"].notna(), "score_dt"].dt.year.astype(int))
    n_orig = int((~weighted["used_filing_date"] & weighted["score_dt"].notna()).sum())
    print(f"  {len(loans):,} loans, {n_orig:,} with an origination date, "
          f"{int(weighted['used_filing_date'].sum()):,} using the filing date")

    print(f"Loading {args.scores}")
    scores = load_forescore(args.scores, years)
    print(f"  {len(scores):,} ZIP-month scores for {min(years)}-{max(years)}")

    averages, coverage = pool_averages(loans, scores)
    print(f"  {coverage['n_eligible']:,} loans with a date and a balance, "
          f"{coverage['n_unmatched']:,} unmatched")
    print(f"  matched loans using filing date: {coverage['n_filing_date']:,}")
    print(f"  matched loans using original balance: {coverage['n_orig_balance']:,}")

    deals = pd.read_csv(args.deals, dtype={"accession": str})
    deals = deals.drop(columns=[c for c in deals.columns if c.startswith("forescore_")])
    deals = deals.merge(averages, on="accession", how="left")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    deals.to_csv(out, index=False)
    deals_path = Path(args.deals)
    if deals_path.resolve() != out.resolve():
        deals.to_csv(deals_path, index=False)
    n_default = int(deals["forescore_default"].notna().sum())
    n_prepay = int(deals["forescore_prepayment"].notna().sum())
    print(f"Wrote {out} ({n_default:,} deals with a default score, "
          f"{n_prepay:,} with a prepayment score, of {len(deals):,})")
    print("\nForeScore default across deals")
    print(score_summary(deals["forescore_default"]).to_string())
    print("\nForeScore prepayment across deals")
    print(score_summary(deals["forescore_prepayment"]).to_string())
    both = deals.dropna(subset=["forescore_default", "forescore_prepayment"])
    if len(both) > 1:
        corr = both["forescore_default"].corr(both["forescore_prepayment"])
        print(f"\nCorrelation of the two pool scores: {corr:.3f}")
    missing = deals.loc[deals["forescore_default"].isna(), "issuer"]
    if len(missing):
        print("\nDeals without a score")
        print("\n".join(missing))


if __name__ == "__main__":
    main()
