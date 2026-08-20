"""
Download the Store Item Demand Forecasting Challenge dataset: daily sales of
50 items across 10 stores between 2013 and 2017, about 913,000 rows.

Original source: https://www.kaggle.com/c/demand-forecasting-kernels-only
Pulled from a public GitHub mirror so the pipeline runs without Kaggle
credentials.

Usage:
    python src/download_data.py
"""

import sys
from pathlib import Path

import requests

URL = ("https://raw.githubusercontent.com/allmeidaapedro/"
       "Store-Item-Demand-Forecasting/main/input/train.csv")


def download(destination: str = "data/sales.csv") -> None:
    """Fetch the dataset unless it is already on disk."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        print(f"{path} already present, skipped")
        return
    print("downloading daily sales (~17 MB)...")
    response = requests.get(URL, timeout=300)
    response.raise_for_status()
    path.write_bytes(response.content)
    print(f"Done: {path} ({len(response.content) / 1e6:.1f} MB)")


if __name__ == "__main__":
    try:
        download()
    except requests.RequestException as exc:
        sys.exit(f"Download failed: {exc}")
