#!/usr/bin/env python3
"""
edgar_rmbs_tapes.py
-------------------
Download loan-level collateral tapes for pre-crisis private-label RMBS
from SEC EDGAR (Free Writing Prospectus filings) and save them as CSV.

Output (under --out, default ./edgar_rmbs):
  index/         cached EDGAR quarterly master.idx files
  raw/           every downloaded filing document (cache, enables resume)
  tapes/         one CSV per detected loan tape, all original columns kept
  loans_<start>_<end>.csv   combined file, standardized columns
                            (zip, state, fico, ltv, cltv, dti, rate, balance, ...)
  manifest.csv   one row per document checked: status and rows parsed

Usage:
  pip install requests pandas lxml openpyxl
  python edgar_rmbs_tapes.py --ua "Your Name you@yourfirm.com" --limit 20   # test run
  python edgar_rmbs_tapes.py --ua "Your Name you@yourfirm.com"              # full 2006-2007
  python edgar_rmbs_tapes.py --ua "..." --all-issuers                       # every FWP filer

SEC fair-access rules: declare a User-Agent with name + email and stay under
10 requests/second. This script sleeps between requests and retries on 429/503.
Re-running skips documents already listed in manifest.csv.
"""

import argparse
import csv
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------

# Issuer names (regex, case-insensitive) for the shelves to scan by default.
# FWPs for RMBS are usually filed under the trust name or the depositor name.
DEFAULT_SHELVES = (
    r"MORGAN STANLEY (?:ABS|MORTGAGE|HOME EQUITY|CAPITAL I)|"
    r"MERRILL LYNCH MORTGAGE|ML MORTGAGE|"
    r"ACE SECURITIES|"
    r"HOME EQUITY LOAN TRUST|MORTGAGE LOAN TRUST|MORTGAGE PASS.THROUGH|"
    r"ASSET.BACKED (?:CERTIFICATES|SECURITIES)|MORTGAGE INVESTORS"
)

# Exclude obvious non-residential filers that also file FWPs.
EXCLUDE = r"COMMERCIAL MORTGAGE|CMBS|AUTO|STUDENT|CREDIT CARD|EQUIPMENT|CDO"

SLEEP = 0.12            # ~8 requests/sec
MIN_DOC_BYTES = 40_000  # tapes are large; skip small term sheets without downloading
MIN_TAPE_ROWS = 25      # fewer rows than this = a summary table, not a tape

# Tokens used to recognise a tape header row.
HEADER_KEYS = {
    "fico":    r"fico|credit\s*score|\bscore\b",
    "zip":     r"zip|postal",
    "ltv":     r"ltv|loan[\s\-_]*to[\s\-_]*value",
    "rate":    r"rate|coupon|\bwac\b",
    "balance": r"bal|upb|amount|\bamt\b",
    "state":   r"\bstate\b|\bst\b",
    "doc":     r"\bdoc",
}

# Standardized output columns -> regex on the original column name.
# Order matters: cltv is matched before ltv, current balance before original.
STANDARD_COLS = [
    ("loan_id",         r"loan.*(id|num|no\b|#)|^id$|account|^loan$"),
    ("zip",             r"zip|postal"),
    ("state",           r"\bstate\b|^st$|prop.*\bst\b"),
    ("fico",            r"fico|credit\s*score|\bscore\b"),
    ("cltv",            r"cltv|combined"),
    ("ltv",             r"ltv|loan[\s\-_]*to[\s\-_]*value"),
    ("dti",             r"dti|debt.*income|back.?end"),
    ("current_balance", r"(curr|cut.?off|sched|stated).*(bal|upb|prin)"),
    ("orig_balance",    r"orig.*(bal|amount|amt|upb|prin)|loan\s*amount|^balance$|principal\s*bal"),
    ("rate",            r"(curr|gross|note|mortgage|coupon|int|init).*rate|^rate$|coupon|^wac$|note"),
    ("orig_date",       r"(orig|first|note|fund).*date"),
    ("doc_type",        r"\bdoc"),
    ("occupancy",       r"occup|owner"),
    ("purpose",         r"purpose"),
    ("property_type",   r"prop.*type|property|dwelling"),
    ("lien",            r"lien"),
    ("product",         r"product|arm|fixed|amort.*type|loan\s*type"),
]

META_COLS = ["accession", "cik", "issuer", "filing_date", "source_doc"]
LOAN_OUT_COLS = META_COLS + [name for name, _ in STANDARD_COLS]


def align_loan_frame(std, meta):
    """Force a stable column set so appending tapes does not shift fields."""
    out = std.copy()
    for key, val in meta.items():
        out[key] = val
    for col in LOAN_OUT_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    return out[LOAN_OUT_COLS]

# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

SESSION = requests.Session()


def get(url, binary=False, tries=5):
    for attempt in range(tries):
        time.sleep(SLEEP)
        try:
            r = SESSION.get(url, timeout=60)
        except requests.RequestException as e:
            wait = 2 ** attempt
            print(f"  network error {e}; retry in {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code in (429, 503):
            wait = 10 * (attempt + 1)
            print(f"  SEC throttled ({r.status_code}); waiting {wait}s", file=sys.stderr)
            time.sleep(wait)
            continue
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.content if binary else r.text
    raise RuntimeError(f"Failed after {tries} tries: {url}")


# ----------------------------------------------------------------------------
# Step 1: list FWP filings from quarterly master indexes
# ----------------------------------------------------------------------------

def load_master_index(year, qtr, cache_dir):
    path = cache_dir / f"master_{year}_Q{qtr}.idx"
    if not path.exists():
        url = f"https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{qtr}/master.idx"
        print(f"Fetching index {year} Q{qtr}")
        raw = get(url, binary=True)
        path.write_bytes(raw)
    text = path.read_bytes().decode("latin-1")
    rows = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) == 5 and parts[0].isdigit():
            rows.append(parts)
    return pd.DataFrame(rows, columns=["cik", "company", "form", "date", "filename"])


def list_fwp_filings(start, end, shelf_regex, cache_dir):
    frames = []
    for year in range(start, end + 1):
        for q in (1, 2, 3, 4):
            frames.append(load_master_index(year, q, cache_dir))
    idx = pd.concat(frames, ignore_index=True)
    fwp = idx[idx.form.str.strip().eq("FWP")].copy()
    if shelf_regex:
        fwp = fwp[fwp.company.str.contains(shelf_regex, case=False, regex=True)]
    fwp = fwp[~fwp.company.str.contains(EXCLUDE, case=False, regex=True)]
    fwp["accession"] = fwp.filename.str.extract(r"(\d{10}-\d{2}-\d{6})")
    return fwp.drop_duplicates("accession").sort_values("date").reset_index(drop=True)


# ----------------------------------------------------------------------------
# Step 2: list the documents inside a filing
# ----------------------------------------------------------------------------

SKIP_EXT = (".jpg", ".gif", ".png", ".pdf", ".zip", ".xml", ".css", ".js")


def filing_documents(cik, accession):
    """Return [(name, url, size_or_None)] for candidate documents in a filing."""
    acc = accession.replace("-", "")
    base = f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{acc}/"
    listing = get(base + "index.json")
    docs = []
    if listing:
        items = json.loads(listing).get("directory", {}).get("item", [])
        for it in items:
            name = it.get("name", "")
            low = name.lower()
            if low.endswith(SKIP_EXT) or "-index" in low or low == f"{accession}.txt":
                continue
            size = it.get("size")
            size = int(size) if str(size).isdigit() else None
            docs.append((name, base + name, size))
    if not docs:  # fall back to the full submission text file
        docs.append((f"{accession}.txt", base + f"{accession}.txt", None))
    return docs


def split_full_submission(text):
    """Split an EDGAR full-submission .txt into its <DOCUMENT> parts."""
    parts = []
    for block in re.findall(r"<DOCUMENT>(.*?)</DOCUMENT>", text, flags=re.S | re.I):
        fname = re.search(r"<FILENAME>([^\n<]+)", block, flags=re.I)
        body = re.search(r"<TEXT>(.*?)</TEXT>", block, flags=re.S | re.I)
        parts.append((fname.group(1).strip() if fname else "doc",
                      body.group(1) if body else block))
    return parts or [("doc", text)]


# ----------------------------------------------------------------------------
# Step 3: detect and parse a loan tape inside a document
# ----------------------------------------------------------------------------

def header_score(cells):
    joined = " | ".join(str(c).lower() for c in cells if c is not None)
    hits = {k for k, pat in HEADER_KEYS.items() if re.search(pat, joined)}
    ok = len(hits) >= 3 and ("fico" in hits or "ltv" in hits)
    return len(hits) if ok else 0


def is_data_row(cells):
    vals = [str(c) for c in cells if c is not None and str(c).strip() not in ("", "nan")]
    numeric = sum(bool(re.fullmatch(r"[\$\(]?-?[\d,]*\.?\d+%?\)?", v.strip())) for v in vals)
    return numeric >= 3


def clean_cell(c):
    if c is None:
        return ""
    s = str(c).replace("\xa0", " ").strip()
    return "" if s.lower() == "nan" else s


def grids_to_tape(grids):
    """grids: list of 2-D lists (tables/sheets). Returns a DataFrame or None.
    Handles tapes split across many HTML tables, with or without repeated headers."""
    header, rows = None, []
    for g in grids:
        g = [[clean_cell(c) for c in row] for row in g]
        if not g:
            continue
        start = 0
        found = None
        for i, row in enumerate(g[:40]):
            s1 = header_score(row)
            nxt = g[i + 1] if i + 1 < len(g) else None
            s2 = 0
            if nxt and len(nxt) == len(row) and not is_data_row(nxt):
                s2 = header_score([f"{a} {b}" for a, b in zip(row, nxt)])
            if s2 > s1:
                found = ([f"{a} {b}".strip() for a, b in zip(row, nxt)], i + 2)
                break
            if s1:
                found = (row, i + 1)
                break
        if found:
            if header is None:
                header = found[0]
            start = found[1]
        elif header is None:
            continue
        for row in g[start:]:
            if len(row) == len(header) and is_data_row(row) and not header_score(row):
                rows.append(row)
    if header is None or len(rows) < MIN_TAPE_ROWS:
        return None
    cols, seen = [], {}
    for c in header:
        c = re.sub(r"\s+", " ", c).strip() or "col"
        seen[c] = seen.get(c, 0) + 1
        cols.append(c if seen[c] == 1 else f"{c}_{seen[c]}")
    df = pd.DataFrame(rows, columns=cols)
    return df.loc[:, (df != "").any()]  # drop empty spacer columns


def parse_html(text):
    try:
        tables = pd.read_html(io.StringIO(text), header=None)
    except (ValueError, ImportError):
        return None
    grids = []
    for t in tables:
        g = t.astype(object).where(t.notna(), "").values.tolist()
        if not all(isinstance(c, (int,)) or str(c).isdigit() for c in t.columns):
            hdr = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
            g = [hdr] + g
        grids.append(g)
    return grids_to_tape(grids)


def parse_excel(blob):
    try:
        sheets = pd.read_excel(io.BytesIO(blob), sheet_name=None, header=None, dtype=str)
    except Exception:
        return None
    return grids_to_tape([s.fillna("").values.tolist() for s in sheets.values()])


def text_tokens(line):
    return [t for t in re.split(r"\t|\||,|\s{2,}", line.strip()) if t]


def parse_text(text):
    lines = text.replace("\r", "").split("\n")
    h = next((i for i, ln in enumerate(lines) if header_score(text_tokens(ln))), None)
    if h is None:
        return None
    hdr = lines[h]

    # Delimited text (tab, pipe, comma)
    for delim, minimum in (("\t", 4), ("|", 4), (",", 6)):
        if hdr.count(delim) >= minimum:
            grid = [ln.split(delim) for ln in lines[h:] if ln.count(delim) >= minimum]
            return grids_to_tape([grid])

    # Fixed-width text: find data lines, infer column spans from blank positions
    data, gap = [], 0
    for ln in lines[h + 1:]:
        if is_data_row(text_tokens(ln)) and not header_score(text_tokens(ln)):
            data.append(ln.rstrip())
            gap = 0
        else:
            gap += 1
            if gap > 60 and data:
                break
    if len(data) < MIN_TAPE_ROWS:
        return None
    width = max(len(ln) for ln in data)
    sample = data[:3000]
    blank = [sum(1 for ln in sample if p >= len(ln) or ln[p] == " ") / len(sample) >= 0.98
             for p in range(width)]
    spans, p = [], 0
    while p < width:
        if blank[p]:
            p += 1
            continue
        q = p
        while q < width and not blank[q]:
            q += 1
        spans.append((p, q))
        p = q

    # Header names: tokens from the header line (plus the line above it if it is a
    # header continuation), each assigned to the column span it overlaps most.
    hdr_lines = [hdr]
    if h > 0 and lines[h - 1].strip() and not is_data_row(text_tokens(lines[h - 1])):
        hdr_lines.insert(0, lines[h - 1])
    names = [[] for _ in spans]
    for hl in hdr_lines:
        for m in re.finditer(r"\S+(?: \S+)*", hl):
            a, b = m.span()
            overlaps = [max(0, min(b, s1) - max(a, s0)) for s0, s1 in spans]
            k = max(range(len(spans)), key=lambda j: (overlaps[j], -abs((s0 := spans[j][0]) - a)))
            if overlaps[k] == 0:  # header token sits in a gap: nearest span
                k = min(range(len(spans)), key=lambda j: abs((spans[j][0] + spans[j][1]) / 2 - (a + b) / 2))
            names[k].append(m.group())
    header = [" ".join(n) or f"col{j}" for j, n in enumerate(names)]
    grid = [header] + [[ln[s0:s1].strip() for s0, s1 in spans] for ln in data]
    return grids_to_tape([grid])


def extract_tape(name, blob):
    low = name.lower()
    if low.endswith((".xls", ".xlsx")):
        return parse_excel(blob)
    text = blob.decode("latin-1", errors="replace")
    if not re.search(r"fico|ltv|loan.to.value", text[:2_000_000], flags=re.I):
        return None
    if re.search(r"<table", text, flags=re.I):
        return parse_html(text)
    return parse_text(text)


# ----------------------------------------------------------------------------
# Step 4: standardize columns
# ----------------------------------------------------------------------------

def to_num(s):
    return pd.to_numeric(
        s.astype(str).str.replace(r"[\$,%\s]", "", regex=True).str.replace(r"^\((.*)\)$", r"-\1", regex=True),
        errors="coerce")


def standardize(df):
    out, used = pd.DataFrame(index=df.index), set()
    for std, pat in STANDARD_COLS:
        for c in df.columns:
            if c not in used and re.search(pat, c, flags=re.I):
                out[std] = df[c]
                used.add(c)
                break
    if "zip" in out:
        z = out["zip"].astype(str).str.replace(r"\D", "", regex=True)
        z = z.where(z.str.len() != 9, z.str[:5])
        out["zip"] = z.where(z.str.len().between(3, 5)).str.zfill(5)
    if "fico" in out:
        f = to_num(out["fico"])
        out["fico"] = f.where(f.between(300, 900))
    for c in ("ltv", "cltv", "dti", "rate"):
        if c in out:
            v = to_num(out[c])
            if v.dropna().median() < 1.5:  # stored as decimals
                v = v * 100
            out[c] = v
    for c in ("orig_balance", "current_balance"):
        if c in out:
            out[c] = to_num(out[c])
    if "state" in out:
        st = out["state"].astype(str).str.strip().str.upper()
        out["state"] = st.where(st.str.fullmatch(r"[A-Z]{2}"))
    keep = [c for c in ("zip", "fico", "ltv") if c in out]
    if keep:
        out = out[out[keep].notna().any(axis=1)]
    return out


# ----------------------------------------------------------------------------
# Rebuild combined loans file from per-tape CSVs
# ----------------------------------------------------------------------------

def rebuild_combined(out, start, end):
    """Re-standardize every file in tapes/ onto a fixed column set."""
    tapes_dir = out / "tapes"
    manifest_path = out / "manifest.csv"
    combined_path = out / f"loans_{start}_{end}.csv"
    if not tapes_dir.exists():
        raise SystemExit(f"No tapes directory at {tapes_dir}")
    meta_by_acc = {}
    if manifest_path.exists():
        man = pd.read_csv(manifest_path, dtype=str)
        for _, row in man.iterrows():
            acc = str(row.get("accession", ""))
            if acc and acc not in meta_by_acc:
                meta_by_acc[acc] = {
                    "cik": row.get("cik", ""),
                    "issuer": row.get("company", ""),
                    "filing_date": row.get("date", ""),
                }

    tmp_path = combined_path.with_suffix(".csv.tmp")
    if tmp_path.exists():
        tmp_path.unlink()
    files = sorted(tapes_dir.glob("*.csv"))
    print(f"Rebuilding {combined_path.name} from {len(files)} tapes")
    n_rows, n_files = 0, 0
    header = True
    for i, path in enumerate(files, 1):
        m = re.match(r"(\d{10}-\d{2}-\d{6})__(.+)\.csv$", path.name)
        if not m:
            continue
        accession, source_doc = m.group(1), m.group(2)
        info = meta_by_acc.get(accession, {})
        try:
            tape = pd.read_csv(path, dtype=str, low_memory=False)
        except Exception as e:
            print(f"  skip {path.name}: {e}", file=sys.stderr)
            continue
        std = align_loan_frame(standardize(tape), {
            "accession": accession,
            "cik": info.get("cik", ""),
            "issuer": info.get("issuer", ""),
            "filing_date": info.get("filing_date", ""),
            "source_doc": source_doc,
        })
        if std.empty:
            continue
        std.to_csv(tmp_path, mode="a", index=False, header=header)
        header = False
        n_rows += len(std)
        n_files += 1
        if i % 100 == 0:
            print(f"  {i}/{len(files)} tapes, {n_rows:,} loans")
    if not tmp_path.exists():
        raise SystemExit("No rows written")
    tmp_path.replace(combined_path)
    print(f"Wrote {n_rows:,} loans from {n_files} tapes -> {combined_path}")


# ----------------------------------------------------------------------------
# Main loop
# ----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ua", help='User-Agent, e.g. "Jane Doe jane@firm.com" (required unless --rebuild-combined)')
    ap.add_argument("--start", type=int, default=2006)
    ap.add_argument("--end", type=int, default=2007)
    ap.add_argument("--shelves", default=DEFAULT_SHELVES, help="issuer-name regex")
    ap.add_argument("--all-issuers", action="store_true", help="scan every FWP filer")
    ap.add_argument("--limit", type=int, default=0, help="max filings (0 = all)")
    ap.add_argument("--out", default="edgar_rmbs")
    ap.add_argument("--rebuild-combined", action="store_true",
                    help="rebuild loans_*.csv from tapes/ (no SEC download)")
    args = ap.parse_args()

    if args.rebuild_combined:
        rebuild_combined(Path(args.out), args.start, args.end)
        return
    if not args.ua:
        ap.error("--ua is required unless you pass --rebuild-combined")

    SESSION.headers.update({"User-Agent": args.ua, "Accept-Encoding": "gzip, deflate"})
    out = Path(args.out)
    for d in ("index", "raw", "tapes"):
        (out / d).mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.csv"
    combined_path = out / f"loans_{args.start}_{args.end}.csv"

    done = set()
    if manifest_path.exists():
        m = pd.read_csv(manifest_path, dtype=str)
        done = set(m.accession + "/" + m.document)

    filings = list_fwp_filings(args.start, args.end,
                               None if args.all_issuers else args.shelves, out / "index")
    if args.limit:
        filings = filings.head(args.limit)
    print(f"{len(filings)} FWP filings to scan")

    new_manifest = not manifest_path.exists()
    with open(manifest_path, "a", newline="") as mf:
        mw = csv.writer(mf)
        if new_manifest:
            mw.writerow(["accession", "cik", "company", "date", "document", "status", "rows"])
        total_loans = 0
        for n, f in filings.iterrows():
            try:
                docs = filing_documents(f.cik, f.accession)
            except Exception as e:
                mw.writerow([f.accession, f.cik, f.company, f.date, "", f"error: {e}", 0])
                continue
            for name, url, size in docs:
                key = f"{f.accession}/{name}"
                if key in done:
                    continue
                if size is not None and size < MIN_DOC_BYTES:
                    mw.writerow([f.accession, f.cik, f.company, f.date, name, "skipped_small", 0])
                    continue
                raw_path = out / "raw" / f"{f.cik}_{f.accession}" / name
                try:
                    if raw_path.exists():
                        blob = raw_path.read_bytes()
                    else:
                        blob = get(url, binary=True)
                        if blob is None:
                            mw.writerow([f.accession, f.cik, f.company, f.date, name, "not_found", 0])
                            continue
                        raw_path.parent.mkdir(parents=True, exist_ok=True)
                        raw_path.write_bytes(blob)

                    parts = [(name, blob)]
                    if name == f"{f.accession}.txt":  # full submission: split documents
                        parts = [(fn, body.encode("latin-1", errors="replace"))
                                 for fn, body in split_full_submission(blob.decode("latin-1", errors="replace"))]

                    found = 0
                    for pname, pblob in parts:
                        tape = extract_tape(pname, pblob)
                        if tape is None:
                            continue
                        stem = re.sub(r"[^\w.-]", "_", f"{f.accession}__{pname}")
                        tape.to_csv(out / "tapes" / f"{stem}.csv", index=False)
                        std = align_loan_frame(standardize(tape), {
                            "accession": f.accession,
                            "cik": f.cik,
                            "issuer": f.company,
                            "filing_date": f.date,
                            "source_doc": pname,
                        })
                        std.to_csv(combined_path, mode="a", index=False,
                                   header=not combined_path.exists())
                        found += len(std)
                    status = "tape" if found else "no_tape"
                    total_loans += found
                    mw.writerow([f.accession, f.cik, f.company, f.date, name, status, found])
                    if found:
                        print(f"[{n + 1}/{len(filings)}] {f.company[:50]} {f.date}: {found:,} loans")
                except Exception as e:
                    mw.writerow([f.accession, f.cik, f.company, f.date, name, f"error: {e}", 0])
                mf.flush()
            if (n + 1) % 25 == 0:
                print(f"... {n + 1}/{len(filings)} filings scanned, {total_loans:,} loans so far")

    print(f"\nDone. {total_loans:,} loans added this run.")
    print(f"Combined file: {combined_path}")
    print(f"Per-tape files (all original columns): {out / 'tapes'}")
    print(f"Log of every document checked: {manifest_path}")


if __name__ == "__main__":
    main()
