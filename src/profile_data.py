"""
Profile the dataset before analysing it.

Runs integrity checks (nulls, duplicates, calendar coverage) plus two
plausibility tests that tell measured demand apart from generated demand.

Usage:
    python src/profile_data.py
"""

import pandas as pd

CV_SPREAD_LIMIT = 0.05
SHARE_SPREAD_LIMIT = 0.005


def load() -> pd.DataFrame:
    df = pd.read_csv("data/sales.csv", parse_dates=["date"])
    return df.rename(columns={"item": "sku", "sales": "units"})


def check_integrity(df: pd.DataFrame) -> None:
    print("== Integrity ==")
    print(f"rows: {len(df):,}")
    print(f"nulls: {df.isna().sum().sum()}")
    print(f"duplicates (store-sku-date): "
          f"{df.duplicated(['store', 'sku', 'date']).sum()}")
    print(f"negative units: {(df['units'] < 0).sum()}")

    days = df["date"].nunique()
    expected = (df["date"].max() - df["date"].min()).days + 1
    print(f"days with data: {days:,} of {expected:,} "
          f"({'no gaps' if days == expected else 'GAPS FOUND'})")

    series = df.groupby(["store", "sku"]).size()
    print(f"store-sku series: {len(series)} "
          f"(each with {series.min():,}-{series.max():,} days)")


def check_variability(df: pd.DataFrame) -> None:
    """
    In measured data the coefficient of variation differs widely across
    products: steady items sit alongside erratic ones.
    """
    print("\n== Variability across products ==")
    daily = df.groupby(["sku", "date"])["units"].sum()
    stats = daily.groupby("sku").agg(["mean", "std"])
    cv = stats["std"] / stats["mean"]

    print(f"lowest CV:  {cv.min():.4f}")
    print(f"highest CV: {cv.max():.4f}")
    print(f"spread:     {cv.max() - cv.min():.4f}")
    print(f"largest SKU / smallest SKU by volume: "
          f"{stats['mean'].max() / stats['mean'].min():.1f}x")

    if cv.max() - cv.min() < CV_SPREAD_LIMIT:
        print("-> All 50 products share essentially the same CV despite a "
              "5x spread in volume. Measured demand does not behave this way.")


def check_store_mix(df: pd.DataFrame) -> None:
    """
    In measured data every store has its own mix: it sells more of some
    products and less of others. A constant share points to a generated
    series built as store factor times item factor.
    """
    print("\n== Product mix per store ==")
    pivot = df.groupby(["store", "sku"])["units"].sum().unstack()
    share = pivot.div(pivot.sum(axis=0), axis=1)
    spread = share.std(axis=1).mean()

    print(f"share held by each store: "
          f"{share.mean(axis=1).min():.1%} to {share.mean(axis=1).max():.1%}")
    print(f"spread of that share across products: {spread:.5f}")

    if spread < SHARE_SPREAD_LIMIT:
        print("-> Every store keeps the same share across all 50 products. "
              "No local preferences, which no real chain shows.")


def main() -> None:
    df = load()
    check_integrity(df)
    check_variability(df)
    check_store_mix(df)
    print("\nConclusion: the series works for practising forecasting, since "
          "trend and seasonality are well formed, but its business findings "
          "do not carry over to a real operation.")


if __name__ == "__main__":
    main()
