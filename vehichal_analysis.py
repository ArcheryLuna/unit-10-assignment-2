"""
vehicle_analysis.py
-------------------
A self-contained script for preparing and analysing the UK new-car
registration dataset (`df_VEH0160_UK.xlsx`).

Features
--------
1. Loads the raw Excel file (VEH0160 – New vehicle registrations by make).
2. Cleans and filters the data to include **only new car registrations**.
3. Aggregates registrations by manufacturer (make) and calendar year.
4. Removes very-low-volume manufacturers (<100 total cars 2015-2023).
5. Computes routine statistics: mean, median, mode, range, variance,
   standard deviation, inter-quartile range.
6. Fits **linear regressions** (and optional quadratic) per manufacturer
   to capture sales trends.
7. Generates key visualisations:
      • Top-10 manufacturers bar chart
      • Market-share pie chart
      • Histogram of registrations (log-scale)
      • Example trend plots (e.g. Ford decline, Tesla growth)
8. Saves two textual summaries:
      • `directors_report.md` – detailed analysis
      • `managers_summary.md` – concise actionable insights
9. Outputs charts as PNG files in an `output/` folder.

Usage
-----
$ pip install pandas numpy matplotlib seaborn statsmodels
$ python vehicle_analysis.py
       --file  path/to/df_VEH0160_UK.xlsx   # optional
       --sheet "VEH0160"                    # optional

Author: Johan Sebastian Luna
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import seaborn as sns

sns.set_theme(style="whitegrid")

# ------------------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------------------
MIN_TOTAL_REGS = 100  # drop brands with fewer total cars 2015-23
POLY_ORDER = 2  # quadratic order for non-linear fits
OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# Map raw column names → canonical names.
COLUMN_MAP = {
    "Make": "make",
    "BodyType": "body_type",
    "Body Type": "body_type",
    "New_or_Used": "new_used",
    "New or Used": "new_used",
    "Year": "year",
    "Date": "date",
    "Quarter": "quarter",
    "Count": "count",
    "Count of vehicles": "count",
    "Number of vehicles": "count",
}


# ------------------------------------------------------------------------
# HELPER FUNCTIONS
# ------------------------------------------------------------------------
def load_and_clean(path: Path, sheet_name: str | int | None = 0) -> pd.DataFrame:
    """Load Excel file and return DataFrame of *new cars* only."""
    df = pd.read_excel(path, sheet_name=sheet_name)

    # Standardise column names
    df.columns = df.columns.str.strip().str.replace("\n", " ").str.replace("\r", " ")
    df = df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})

    # Required columns check
    if {"make", "body_type", "new_used"}.difference(df.columns):
        raise KeyError("Dataset must contain Make, BodyType and New_or_Used columns")

    # Derive 'year'
    if "year" not in df.columns and "date" in df.columns:
        df["year"] = pd.to_datetime(df["date"]).dt.year

    # Filter to new car registrations
    df = df[
        df["body_type"].str.lower().str.contains("car")
        & df["new_used"].str.lower().str.startswith("new")
    ]

    df = df.dropna(subset=["make", "year"])
    df = df[df["count"] > 0]

    # Keep analysis window
    df = df[(df["year"] >= 2015) & (df["year"] <= 2023)]

    return df


def aggregate_by_make_year(df: pd.DataFrame) -> pd.DataFrame:
    """Sum registrations by make and year."""
    return (
        df.groupby(["make", "year"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "registrations"})
    )


def filter_low_volume_makes(df_agg: pd.DataFrame, min_total: int) -> pd.DataFrame:
    totals = df_agg.groupby("make")["registrations"].sum()
    keep = totals[totals >= min_total].index
    return df_agg[df_agg["make"].isin(keep)].copy()


def pivot_make_year(df_agg: pd.DataFrame) -> pd.DataFrame:
    return (
        df_agg.pivot_table(
            index="make", columns="year", values="registrations", fill_value=0
        )
        .astype(int)
        .sort_index()
    )


def compute_summary_stats(pivot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for yr in pivot.columns:
        data = pivot[yr]
        rows.append(
            {
                "year": yr,
                "mean": data.mean(),
                "median": data.median(),
                "mode": data.mode().iloc[0] if not data.mode().empty else np.nan,
                "range": data.max() - data.min(),
                "variance": data.var(ddof=0),
                "std_dev": data.std(ddof=0),
                "iqr": data.quantile(0.75) - data.quantile(0.25),
            }
        )
    return pd.DataFrame(rows)


def fit_linear_trends(pivot: pd.DataFrame) -> pd.DataFrame:
    trends = []
    years = pivot.columns.to_numpy(dtype=float)
    for make, row in pivot.iterrows():
        y = row.to_numpy(dtype=float)
        X = sm.add_constant(years)
        model = sm.OLS(y, X).fit()
        trends.append(
            {
                "make": make,
                "slope": model.params[1],
                "intercept": model.params[0],
                "r_squared": model.rsquared,
            }
        )
    return pd.DataFrame(trends).set_index("make").sort_values("slope")


# -------------  Plotting helpers  ----------------
def plot_top10(pivot: pd.DataFrame, year: int):
    top10 = pivot[year].sort_values(ascending=False).head(10)
    plt.figure(figsize=(10, 6))
    plt.bar(top10.index, top10.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("New car registrations")
    plt.title(f"Top 10 manufacturers – {year}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"top10_{year}.png", dpi=300)
    plt.close()


def plot_market_share_pie(pivot: pd.DataFrame, year: int, top_n: int = 5):
    totals = pivot[year]
    top = totals.sort_values(ascending=False).head(top_n)
    labels = list(top.index) + ["Others"]
    sizes = list(top.values) + [totals.sum() - top.sum()]
    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    plt.title(f"Market share – {year}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"market_share_{year}.png", dpi=300)
    plt.close()


def plot_histogram(pivot: pd.DataFrame, year: int):
    data = pivot[year]
    plt.figure(figsize=(8, 6))
    plt.hist(data[data > 0], bins=30, log=True, edgecolor="black")
    plt.xlabel("Registrations")
    plt.ylabel("Frequency (log scale)")
    plt.title(f"Distribution of manufacturer registrations – {year}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"histogram_{year}.png", dpi=300)
    plt.close()


def plot_trend_example(pivot: pd.DataFrame, make: str):
    if make not in pivot.index:
        return
    years = pivot.columns.to_numpy(dtype=float)
    y = pivot.loc[make].to_numpy(dtype=float)
    slope, intercept = np.polyfit(years, y, 1)
    y_pred = intercept + slope * years
    plt.figure(figsize=(8, 6))
    plt.plot(years, y, "x", label="Actual")
    plt.plot(years, y_pred, "-", label=f"Linear fit (slope={slope:.0f})")
    plt.xlabel("Year")
    plt.ylabel("Registrations")
    plt.title(f"{make} – new car registrations")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / f"trend_{make.lower().replace(' ', '_')}.png", dpi=300)
    plt.close()


# -------------  Reporting helpers  ----------------
def save_reports(summary: pd.DataFrame, trends: pd.DataFrame, pivot: pd.DataFrame):
    directors = OUTPUT_DIR / "directors_report.md"
    with directors.open("w") as f:
        f.write("# Detailed Directors Report\n\n")
        f.write("## Summary statistics by year\n")
        f.write(summary.to_markdown(index=False))
        f.write("\n\n## Linear trends by manufacturer\n")
        f.write(
            trends.sort_values("slope", ascending=False).to_markdown(floatfmt=".2f")
        )
        f.write("\n")

    managers = OUTPUT_DIR / "managers_summary.md"
    winners = (
        trends[trends["slope"] > 0]
        .sort_values("slope", ascending=False)
        .head(10)
        .index.tolist()
    )
    losers = trends[trends["slope"] < 0].sort_values("slope").head(10).index.tolist()
    with managers.open("w") as f:
        f.write("# Managers Summary\n\n")
        f.write("**Growing brands:** " + ", ".join(winners) + "\n\n")
        f.write("**Declining brands:** " + ", ".join(losers) + "\n\n")
        f.write("See detailed report for full analysis and charts.\n")


# ------------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Analyse UK new-car registrations (VEH0160)."
    )
    parser.add_argument(
        "--file", default="./Data/df_VEH0160_UK.xlsx", help="Path to VEH0160 Excel file"
    )
    parser.add_argument("--sheet", default=0, help="Sheet index/name (default first)")
    args = parser.parse_args()

    excel_path = Path(args.file)
    if not excel_path.exists():
        raise FileNotFoundError(f"Excel file not found: {excel_path}")

    print("[1/6] Loading & cleaning …")
    df = load_and_clean(excel_path, args.sheet)
    print(f"    rows after filtering: {len(df):,}")

    print("[2/6] Aggregating …")
    df_agg = aggregate_by_make_year(df)
    df_agg = filter_low_volume_makes(df_agg, MIN_TOTAL_REGS)
    pivot = pivot_make_year(df_agg)
    print(f"    manufacturers retained: {pivot.shape[0]}")

    print("[3/6] Computing statistics …")
    summary = compute_summary_stats(pivot)

    print("[4/6] Fitting linear trends …")
    trends = fit_linear_trends(pivot)

    print("[5/6] Generating charts …")
    plot_top10(pivot, 2023)
    plot_market_share_pie(pivot, 2023)
    plot_histogram(pivot, 2023)
    for m in ["Ford", "Tesla"]:
        plot_trend_example(pivot, m)

    print("[6/6] Writing reports …")
    save_reports(summary, trends, pivot)
    print("Done. Results saved to 'output/' folder.")


if __name__ == "__main__":
    main()
