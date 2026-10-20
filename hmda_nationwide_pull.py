#!/usr/bin/env python3
"""Download nationwide HMDA originations and attach a 5-digit ZIP to each loan.

HMDA identifies property location by census tract, not by ZIP code. A tract
often crosses more than one ZIP, so there is no exact ZIP for every loan.
This script assigns the ZIP (or Census ZCTA) that contains the largest share
of the tract and keeps that share in zip_share. zip_exact is true when that
share is at least 99 percent.

The default crosswalk is the Census ZCTA-to-tract relationship file: 2020
tracts for HMDA years 2022 and later, 2010 tracts before that. A ZCTA is the
Census area built to match a ZIP code, not the USPS delivery boundary. Pass
--crosswalk with a HUD TRACT_ZIP file (columns TRACT, ZIP, RES_RATIO) to
assign the ZIP that holds the most residential addresses instead.

The nationwide CSV is tens of gigabytes. Files are written under data/hmda,
which git does not track.

Usage:
  python hmda_nationwide_pull.py --year 2024
  python hmda_nationwide_pull.py --year 2024 --conventional --first-lien
  python hmda_nationwide_pull.py --year 2024 --crosswalk data/hmda/crosswalk/TRACT_ZIP.xlsx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import requests

BASE = "https://ffiec.cfpb.gov/v2/data-browser-api/view/nationwide/csv"
DATA_DIR = Path("data/hmda")
CROSSWALK_DIR = DATA_DIR / "crosswalk"

# Read these as text so codes keep leading zeros and "Exempt"/"NA" values survive.
TEXT_COLS = [
    "census_tract", "county_code", "state_code", "derived_msa_md", "lei",
    "loan_amount", "interest_rate", "rate_spread", "combined_loan_to_value_ratio",
    "debt_to_income_ratio", "property_value", "income", "loan_term",
    "total_loan_costs", "origination_charges", "discount_points",
    "lender_credits", "total_points_and_fees", "prepayment_penalty_term",
    "intro_rate_period", "multifamily_affordable_units",
]

# Census relationship files. HMDA uses 2020 tracts from data year 2022 on.
CENSUS_CROSSWALK = {
    2020: (
        "https://www2.census.gov/geo/docs/maps-data/data/rel2020/zcta520/tab20_zcta520_tract20_natl.txt",
        "tab20_zcta520_tract20_natl.txt",
    ),
    2010: (
        "https://www2.census.gov/geo/docs/maps-data/data/rel/zcta_tract_rel_10.txt",
        "zcta_tract_rel_10.txt",
    ),
}
EXACT_SHARE = 0.99

# Public HMDA has no loan id, origination month, filing date, or current balance.
# activity_year is the origination year. loan_amount is the original amount.
SLIM_COLS = [
    "activity_year",
    "loan_amount",
    "census_tract",
    "zipcode",
    "zip_share",
    "zip_exact",
]


def build_url(year, conventional, first_lien):
    params = [f"years={year}", "actions_taken=1"]
    if conventional:
        params.append("loan_types=1")
    if first_lien:
        params.append("lien_statuses=1")
    return f"{BASE}?{'&'.join(params)}"


def download(url, dest, tries=4):
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, tries + 1):
        try:
            print(f"Downloading (attempt {attempt}):\n  {url}")
            with requests.get(url, stream=True, timeout=(30, 600)) as r:
                r.raise_for_status()
                done, t0, last = 0, time.time(), 0
                with open(part, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
                        done += len(chunk)
                        if done - last >= 250 * 1024 * 1024:
                            rate = done / 1e6 / max(time.time() - t0, 1)
                            print(f"  {done / 1e9:.2f} GB  ({rate:.1f} MB/s)")
                            last = done
            if done == 0:
                raise RuntimeError("empty response")
            part.replace(dest)
            print(f"Saved {dest} ({done / 1e9:.2f} GB)")
            return
        except Exception as e:
            print(f"  failed: {e}", file=sys.stderr)
            if attempt == tries:
                raise
            time.sleep(30 * attempt)


def geography_vintage(year):
    """HMDA census tracts switch from 2010 to 2020 boundaries in data year 2022."""
    return 2020 if year >= 2022 else 2010


def _read_table(path):
    sample = path.read_text(encoding="utf-8", errors="replace")[:800]
    header = sample.splitlines()[0]
    sep = "|" if "|" in header else ","
    frame = pd.read_csv(path, sep=sep, dtype=str, low_memory=False)
    frame.columns = [str(c).strip() for c in frame.columns]
    return frame


def _best_zip(frame, tract_col, zip_col, share):
    """One ZIP per tract: the ZIP with the largest share of that tract."""
    work = frame[[tract_col, zip_col]].copy()
    work["tract"] = work[tract_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(11)
    work["zipcode"] = work[zip_col].astype(str).str.replace(r"\D", "", regex=True).str.zfill(5)
    work["zip_share"] = pd.to_numeric(share, errors="coerce")
    work = work.loc[
        work["tract"].str.fullmatch(r"\d{11}")
        & work["zipcode"].str.fullmatch(r"\d{5}")
        & work["zip_share"].notna()
    ]
    if work["zip_share"].max() > 1.5:
        work["zip_share"] = work["zip_share"] / 100.0
    work = work.sort_values("zip_share").drop_duplicates("tract", keep="last")
    n_split = int((work["zip_share"] < EXACT_SHARE).sum())
    print(f"  {len(work):,} tracts, {n_split:,} split across more than one ZIP "
          f"(share under {EXACT_SHARE:.0%})")
    return work.set_index("tract")[["zipcode", "zip_share"]]


def load_census_crosswalk(year):
    vintage = geography_vintage(year)
    url, name = CENSUS_CROSSWALK[vintage]
    path = CROSSWALK_DIR / name
    if not path.exists():
        download(url, path)
    else:
        print(f"Using existing {path}")
    print(f"Census {vintage} ZCTA-tract crosswalk")
    frame = _read_table(path)
    if vintage == 2020:
        part = pd.to_numeric(frame["AREALAND_PART"], errors="coerce")
        total = pd.to_numeric(frame["AREALAND_TRACT_20"], errors="coerce")
        share = part / total.where(total > 0)
        return _best_zip(frame, "GEOID_TRACT_20", "GEOID_ZCTA5_20", share), "census_zcta_land"
    return _best_zip(frame, "GEOID", "ZCTA5", frame["TRAREALANDPCT"]), "census_zcta_land"


def load_hud_crosswalk(path):
    """HUD TRACT_ZIP file. RES_RATIO is the share of the tract's residences in each ZIP."""
    print(f"HUD tract-to-ZIP crosswalk: {path}")
    suffix = Path(path).suffix.lower()
    if suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path, dtype=str)
    else:
        frame = _read_table(Path(path))
    frame.columns = [str(c).strip().upper() for c in frame.columns]
    return _best_zip(frame, "TRACT", "ZIP", frame["RES_RATIO"]), "hud_residential"


def load_crosswalk(year, crosswalk):
    if crosswalk:
        table, source = load_hud_crosswalk(crosswalk)
    else:
        table, source = load_census_crosswalk(year)
    print(f"  assignment: {source}")
    return table


def attach_zip(table, crosswalk):
    """Add zipcode, zip_share, and zip_exact. Unmatched tracts stay null."""
    if "census_tract" not in table.column_names:
        raise SystemExit("HMDA file has no census_tract column")
    tract = table.column("census_tract").to_pandas().astype("string").fillna("")
    digits = tract.str.replace(r"\D", "", regex=True).fillna("")
    # zfill keeps FIPS states 01-09. A rule that requires the first digit to be
    # 1-9 drops Alabama through Connecticut even when the stored tract is 11 digits.
    valid = digits.str.fullmatch(r"\d{1,11}").fillna(False)
    key = digits.str.zfill(11)
    key = key.where(valid & key.ne("0" * 11))
    zipcode = key.map(crosswalk["zipcode"])
    share = pd.to_numeric(key.map(crosswalk["zip_share"]), errors="coerce")
    exact = share.ge(EXACT_SHARE)
    exact = exact.where(share.notna())
    out = table.append_column("zipcode", pa.array(zipcode, type=pa.string()))
    out = out.append_column("zip_share", pa.array(share, type=pa.float64()))
    out = out.append_column("zip_exact", pa.array(exact, type=pa.bool_()))
    return out


def csv_to_parquet(csv_path, pq_path, crosswalk):
    print(f"Converting to Parquet: {pq_path}")
    with open(csv_path, "r", encoding="utf-8", errors="replace") as fh:
        header = fh.readline().strip().split(",")
    types = {c: pa.string() for c in TEXT_COLS if c in header}
    reader = pacsv.open_csv(
        csv_path,
        read_options=pacsv.ReadOptions(block_size=64 * 1024 * 1024),
        convert_options=pacsv.ConvertOptions(column_types=types,
                                             strings_can_be_null=True),
    )
    writer, rows = None, 0
    n_zip = n_exact = 0
    for batch in reader:
        table = attach_zip(pa.Table.from_batches([batch]), crosswalk)
        if writer is None:
            writer = pq.ParquetWriter(pq_path, table.schema, compression="zstd")
        else:
            table = table.cast(writer.schema)
        writer.write_table(table)
        rows += table.num_rows
        n_zip += pc_sum_valid(table.column("zipcode"))
        n_exact += pc_sum_true(table.column("zip_exact"))
        print(f"  {rows:,} rows", end="\r")
    if writer:
        writer.close()
    matched = n_zip / rows if rows else 0
    exact = n_exact / rows if rows else 0
    print(f"\nWrote {rows:,} loans to {pq_path}")
    print(f"  ZIP assigned: {n_zip:,} ({matched:.1%}); "
          f"tract essentially inside one ZIP: {n_exact:,} ({exact:.1%})")
    return rows


def write_slim(pq_path, slim_pq, slim_csv):
    """Keep origination year, original amount, tract, and the ZIP assignment."""
    print(f"Writing {slim_pq.name} and {slim_csv.name}")
    reader = pq.ParquetFile(pq_path)
    writer = None
    header = True
    rows = 0
    for batch in reader.iter_batches(columns=SLIM_COLS, batch_size=500_000):
        table = pa.Table.from_batches([batch])
        if writer is None:
            writer = pq.ParquetWriter(slim_pq, table.schema, compression="zstd")
        writer.write_table(table)
        table.to_pandas().to_csv(slim_csv, mode="w" if header else "a",
                                  header=header, index=False)
        header = False
        rows += table.num_rows
        print(f"  {rows:,} rows", end="\r")
    if writer:
        writer.close()
    print(f"\nWrote {rows:,} loans")


def pc_sum_valid(column):
    return int(len(column) - column.null_count)


def pc_sum_true(column):
    values = column.to_pandas()
    return int(values.fillna(False).sum())


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--conventional", action="store_true",
                    help="only conventional loans (exclude FHA/VA/USDA)")
    ap.add_argument("--first-lien", action="store_true", help="only first liens")
    ap.add_argument("--out", default=str(DATA_DIR),
                    help="directory for the CSV and Parquet (default: data/hmda)")
    ap.add_argument("--crosswalk", default=None,
                    help="HUD TRACT_ZIP csv/xlsx; default is the Census ZCTA file")
    ap.add_argument("--keep-csv", choices=["yes", "no"], default="yes")
    args = ap.parse_args()

    tag = "originations"
    if args.conventional:
        tag = "conventional_" + tag
    if args.first_lien:
        tag = "firstlien_" + tag
    stem = f"hmda_{args.year}_nationwide_{tag}"

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / f"{stem}.csv"
    pq_path = out / f"{stem}.parquet"

    crosswalk = load_crosswalk(args.year, args.crosswalk)

    if not csv_path.exists():
        download(build_url(args.year, args.conventional, args.first_lien), csv_path)
    else:
        print(f"Using existing {csv_path}")

    csv_to_parquet(csv_path, pq_path, crosswalk)
    write_slim(pq_path, out / f"{stem}_slim.parquet", out / f"{stem}_slim.csv")

    if args.keep_csv == "no":
        csv_path.unlink()
        print(f"Deleted {csv_path}")


if __name__ == "__main__":
    main()
