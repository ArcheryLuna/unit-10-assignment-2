"""
GCSE Results Analysis – Unit 10, Assignment 2  (v2 – handles 3 sheets)
Author : Johan Sebastian Luna
Date   : 25 Jun 2025
Python ≥3.9, pandas ≥2.0, scipy, matplotlib
---------------------------------------------------------------------
This version expects THREE separate sheets in the Excel file:

    * "Table 16 Boys"
    * "Table 16 Girls"
    * "Table 16 All"   (totals / combined)

Each sheet has identical structure, so the same cleaning function is
applied and the results are merged into a single dataframe containing
    LA_name | M_5AC | F_5AC | T_5AC
---------------------------------------------------------------------
"""

import pathlib

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

EXCEL_FILE = "./Data/Santized GCSE Data.xlsx"
SHEETS = {
    "M": "Table 16 Boys",
    "F": "Table 16 Girls",
    "T": "Table 16 All",
}
PLOT_DIR = pathlib.Path("plots")
PLOT_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1.  LOAD & TIDY  –– a helper we can reuse for all three sheets
# ---------------------------------------------------------------------------
def load_clean(sheet_name: str) -> pd.DataFrame:
    """Return a tidy dataframe for ONE sheet only."""
    raw = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=None)

    # First physical row holds the real headings
    header = raw.iloc[0]
    df = raw.iloc[1:].copy()
    df.columns = header

    # Remove blank lines created by merged-cell layout
    df = df[df["LA_name"].notna()].copy()

    # Remove Region & England summary rows
    REGIONS = [
        "North East",
        "North West",
        "Yorkshire and The Humber",
        "East Midlands",
        "West Midlands",
        "East of England",
        "London",
        "South East",
        "South West",
        "England",
    ]
    df = df[~df["LA_name"].isin(REGIONS)].reset_index(drop=True)

    # Force numeric conversion for every non-text column
    for col in df.columns:
        if col != "LA_name":
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ---------------------------------------------------------------------------
# 2.  READ all three sheets & MERGE on LA_name
# ---------------------------------------------------------------------------
df_boys = load_clean(SHEETS["M"])
df_girls = load_clean(SHEETS["F"])
df_all = load_clean(SHEETS["T"])

# Each sheet’s key indicator is the % achieving 5+ A*-C grades;
# the exact column header sometimes differs by year release
# (e.g. "5AC", "5ACinclEM"), but in this dataset it is consistent.
RESULT_COL = "5AC"

df_merged = (
    df_boys[["LA_name", RESULT_COL]]
    .rename(columns={RESULT_COL: "M_5AC"})
    .merge(
        df_girls[["LA_name", RESULT_COL]].rename(columns={RESULT_COL: "F_5AC"}),
        on="LA_name",
    )
    .merge(
        df_all[["LA_name", RESULT_COL]].rename(columns={RESULT_COL: "T_5AC"}),
        on="LA_name",
    )
)

# Quick sanity check: 3 numeric columns, no duplicates
assert df_merged.columns.tolist() == ["LA_name", "M_5AC", "F_5AC", "T_5AC"]
assert not df_merged.duplicated("LA_name").any()


# ---------------------------------------------------------------------------
# 3.  DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------
stats_cols = ["M_5AC", "F_5AC", "T_5AC"]
desc = (
    df_merged[stats_cols]
    .agg(
        ["count", "mean", "median", "mode", "var", "std", "min", "max", "skew", "kurt"]
    )
    .T
)
desc.to_csv("summary_stats.csv", index_label="Measure")


# ---------------------------------------------------------------------------
# 4.  ABOVE / BELOW NATIONAL MEAN  (using ‘All’ data)
# ---------------------------------------------------------------------------
national_mean = desc.loc["T_5AC", "mean"]
better = df_merged.loc[df_merged["T_5AC"] > national_mean, ["LA_name", "T_5AC"]]
worse = df_merged.loc[df_merged["T_5AC"] < national_mean, ["LA_name", "T_5AC"]]

better.to_csv("la_above_below_mean.csv", index=False, mode="w")
worse.to_csv("la_above_below_mean.csv", index=False, mode="a", header=False)


# ---------------------------------------------------------------------------
# 5.  PAIRED t-TEST  (Boys vs Girls, LA-by-LA)
# ---------------------------------------------------------------------------
t_stat, p_val = stats.ttest_rel(
    df_merged["M_5AC"], df_merged["F_5AC"], nan_policy="omit"
)
with open("male_female_ttest.txt", "w") as f:
    f.write(
        f"Paired t-test – Boys vs Girls (% 5+A*-C)\nt = {t_stat:.3f}\np = {p_val:.3e}\n"
    )


# ---------------------------------------------------------------------------
# 6.  PLOTS
# ---------------------------------------------------------------------------
def save_bar(series, title, ylabel, filename):
    series.sort_values(ascending=False).plot(kind="bar")
    plt.title(title)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename)
    plt.clf()


# 6a. Top 15 LAs by combined performance
save_bar(
    df_merged.set_index("LA_name")["T_5AC"].nlargest(15),
    "Top 15 LAs – % 5+ A*-C (All pupils)",
    "% of pupils",
    "top15_bar.png",
)

# 6b. Histogram of national distribution
plt.hist(df_merged["T_5AC"].dropna(), bins=20)
plt.axvline(national_mean, linestyle="--", linewidth=1)
plt.title("Distribution of % 5+ A*-C across LAs (All pupils)")
plt.xlabel("% of pupils")
plt.ylabel("Number of LAs")
plt.tight_layout()
plt.savefig(PLOT_DIR / "histogram_T_5AC.png")
plt.clf()

# 6c. Pie chart – national mean by gender
means = df_merged[["M_5AC", "F_5AC"]].mean()
plt.pie(means, labels=["Boys", "Girls"], autopct="%1.1f%%", startangle=90)
plt.title("National mean – % 5+ A*-C by gender")
plt.savefig(PLOT_DIR / "gender_pie.png")
plt.clf()

# 6d. Linear regression Boys vs Girls
x, y = df_merged["M_5AC"].values, df_merged["F_5AC"].values
slope, intercept, r, p, _ = stats.linregress(x, y)

plt.scatter(x, y, alpha=0.7)
plt.plot(
    x,
    intercept + slope * x,
    linestyle="--",
    label=f"y={slope:.2f}·x+{intercept:.1f}\n$r={r:.2f}$",
)
plt.xlabel("Boys % 5+ A*-C")
plt.ylabel("Girls % 5+ A*-C")
plt.title("Linear regression: Girls vs Boys performance")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "regression_FvsM.png")
plt.clf()
