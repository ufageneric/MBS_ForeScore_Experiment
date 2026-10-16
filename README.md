# MBS ForeScore Experiment

Pool-level ForeScore default and prepayment for a set of 2006–2007 private-label RMBS deals. Loan tapes come from SEC EDGAR free-writing prospectuses. ForeScore is the University Financial Associates ZIP-and-origination-month index of how local conditions change mortgage default and prepayment. A score of 1 is the baseline.

## Scripts

- `edgar_rmbs_tapes.py` downloads the 2006–2007 FWP loan tapes into `edgar_rmbs/`.
- `edgar_rmbs_aggregate.py` deduplicates loans and writes balance-weighted deal, ZIP, and state summaries. FICO, LTV, CLTV, DTI, and rate are weighted by loan balance.
- `forescore_pool_avg.py` attaches life-of-loan ForeScore default and prepayment to the 66-deal file.

```bash
python forescore_pool_avg.py
```

## Output

`output_file/deals_rmbs_final.csv` is the deal file: one row per RMBS, with the pool characteristics and the ForeScore columns. The same table is also written back to `edgar_rmbs/deals_rmbs_final.csv`.

For each pool the script computes

```text
sum(balance * LifeOfLoan score) / sum(balance)
```

for `LifeOfLoanDefault` and `LifeOfLoanPrepayment`. The score is matched on the loan’s ZIP and origination month. If the origination date is missing, the pool filing date is used. If the current balance is missing, the original balance is used. Loans with no balance, no date, or no ForeScore row are left out of both sums.

`forescore_default` and `forescore_prepayment` are those balance-weighted means. The `_min`, `_p25`, `_median`, `_p75`, `_max`, `_std`, and `_range` columns describe the loan-level scores inside that pool, with each matched loan counted once.

64 of the 66 deals have both scores. Asset Backed Securities Corp NC 2006-HE4 and Bayview 2007-B have no loan balance on the tape, so those ForeScore cells are blank.
