"""
download_data.py — Batch download Treasury yield data from FRED for multiple year ranges.

Usage:
    python download_data.py                    # Download all predefined ranges
    python download_data.py --start 2020-01-01 --end 2024-12-31   # Custom range
    python download_data.py --list             # Show available cached data
"""
import argparse
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from bond_pnl.yield_curve import fetch_yields, CACHE_DIR, TENOR_MAP

# Predefined analysis windows
WINDOWS = [
    ("2018-01-01", "2019-12-31", "Pre-COVID, late tightening cycle"),
    ("2020-01-01", "2021-12-31", "COVID crisis & zero-rate era"),
    ("2022-01-01", "2023-12-31", "Fed hiking cycle (aggressive)"),
    ("2023-06-01", "2024-06-30", "Project default window"),
    ("2024-01-01", "2025-12-31", "Recent period"),
    ("2018-01-01", "2025-12-31", "Full 8-year history"),
]


def download_all():
    print("=" * 60)
    print("  FRED Treasury Yield Data Downloader")
    print("=" * 60)
    print(f"\n  Series: {', '.join(TENOR_MAP.keys())}")
    print(f"  Cache dir: {CACHE_DIR}\n")

    for start, end, desc in WINDOWS:
        cache_file = CACHE_DIR / f"yields_{start}_{end}.csv"
        if cache_file.exists():
            import pandas as pd
            df = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            print(f"  [CACHED]  {start} → {end}  ({desc})  {len(df)} dates")
        else:
            print(f"  [DOWNLOADING] {start} → {end}  ({desc}) ...", end=" ")
            try:
                df = fetch_yields(start, end, cache=True)
                print(f"✓ {len(df)} dates")
            except Exception as e:
                print(f"✗ Error: {e}")


def download_custom(start, end):
    print(f"  Downloading {start} → {end} ...")
    df = fetch_yields(start, end, cache=True)
    print(f"  ✓ {len(df)} dates × {len(df.columns)} tenors")
    print(f"  Saved to: {CACHE_DIR / f'yields_{start}_{end}.csv'}")
    return df


def list_cached():
    print("\n  Cached data files:")
    csvs = sorted(CACHE_DIR.glob("yields_*.csv"))
    if not csvs:
        print("    (none)")
        return
    import pandas as pd
    for f in csvs:
        df = pd.read_csv(f, index_col=0, parse_dates=True)
        print(f"    {f.name}  ({len(df)} dates, "
              f"{df.index.min().strftime('%Y-%m-%d')} → "
              f"{df.index.max().strftime('%Y-%m-%d')})")


def main():
    ap = argparse.ArgumentParser(description="Download FRED Treasury yields")
    ap.add_argument("--start", type=str, help="Start date (YYYY-MM-DD)")
    ap.add_argument("--end", type=str, help="End date (YYYY-MM-DD)")
    ap.add_argument("--list", action="store_true", help="List cached data")
    ap.add_argument("--all", action="store_true",
                    help="Download all predefined windows")
    args = ap.parse_args()

    if args.list:
        list_cached()
    elif args.start and args.end:
        download_custom(args.start, args.end)
    else:
        download_all()


if __name__ == "__main__":
    main()
