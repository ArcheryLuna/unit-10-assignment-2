"""
GCSE Results Analysis – Unit 10, Assignment 2  (v3.1 – fixed mode aggregation)
Author : Johan Sebastian Luna
Date   : 25 Jun 2025
Python ≥3.9, pandas ≥2.0, scipy, matplotlib
---------------------------------------------------------------------
This script analyses the GCSE "Table 16" dataset provided in three
separate worksheets (Boys, Girls, All) and produces:
    •  summary_stats.csv              – descriptive statistics
    •  la_above_below_mean.csv        – LAs better/worse than national mean
    •  male_female_ttest.txt          – paired‑sample t‑test results
    •  plots/ … .png                  – bar, histogram, pie & regression plots

Revision 3.1 – why?
———————————————————
Pandas ≥2.2 raises **ValueError: cannot combine transform and aggregation operations**
when `pd.Series.mode` (returns a Series) is mixed with scalar reducers in
`DataFrame.agg`.  The statistics block has therefore been rewritten to
calculate the **mode** separately and append it to the aggregated table.
All other logic is unchanged.
"""

from __future__ import annotations

import pathlib
from typing import Dict, List

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------------
EXCEL_FILE: str = "./Data/Santized GCSE Data.xlsx"
SHEETS: Dict[str, str] = {
    "M": "Table 16 Boys",
    "F": "Table 16 Girls",
    "T": "Table 16 All",
}
PLOT_DIR: pathlib.Path = pathlib.Path("plots")
PLOT_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1.  DATA INGESTION & CLEANING
# ---------------------------------------------------------------------------


def load_clean(sheet_name: str, sex_tag: str) -> pd.DataFrame:
    """Return a tidy DataFrame with columns  LA_name | <sex_tag>_5AC.

    * Dynamically finds the header row by scanning for the word "Region".
    * Normalises the Local‑Authority column name to **LA_name**.
    * Removes region/England summary rows.
    * Picks the plain "5+A*-C" column (excluding English+Math variants).
    """
    raw = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=None)

    # — locate header row —
    header_idx = next(
        idx
        for idx, val in enumerate(raw.iloc[:, 0])
        if isinstance(val, str) and "Region" in val
    )
    header = raw.iloc[header_idx]
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = header

    # — Local Authority column —
    la_col = next(
        c for c in df.columns if isinstance(c, str) and "Local Authority" in c
    )
    df = df.rename(columns={la_col: "LA_name"})
    df["LA_name"] = df["LA_name"].astype(str).str.replace("\n", " ").str.strip()

    # — drop region summary rows —
    REGIONS = {
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
    }
    df = df[df["LA_name"].notna() & ~df["LA_name"].isin(REGIONS)]

    # — pick %5+A*-C column —
    res_col = next(
        c
        for c in df.columns
        if isinstance(c, str) and "5+A*-C" in c and "including" not in c.lower()
    )
    df[res_col] = pd.to_numeric(df[res_col], errors="coerce")

    return df[["LA_name", res_col]].rename(columns={res_col: f"{sex_tag}_5AC"})


# ---------------------------------------------------------------------------
# 2.  BUILD MASTER DATAFRAME
# ---------------------------------------------------------------------------

df_boys = load_clean(SHEETS["M"], "M")
df_girls = load_clean(SHEETS["F"], "F")
df_all = load_clean(SHEETS["T"], "T")

df_merged = df_boys.merge(df_girls, on="LA_name", how="outer").merge(
    df_all, on="LA_name", how="outer"
)

missing = df_merged[df_merged.isna().any(axis=1)]
if not missing.empty:
    print("⚠️  Some LAs lacked data in one or more sheets – they were dropped.")
    df_merged = df_merged.dropna()

assert df_merged.columns.tolist() == ["LA_name", "M_5AC", "F_5AC", "T_5AC"]
assert not df_merged.duplicated("LA_name").any()

# ---------------------------------------------------------------------------
# 3.  DESCRIPTIVE STATISTICS (fixed aggregation)
# ---------------------------------------------------------------------------

stats_cols = ["M_5AC", "F_5AC", "T_5AC"]

# Scalar reducers only — avoid mode here
agg_funcs = ["count", "mean", "median", "var", "std", "min", "max", "skew", "kurt"]
desc = df_merged[stats_cols].agg(agg_funcs).T

# Compute mode separately (first modal value)
mode_row = df_merged[stats_cols].mode().iloc[0]
desc["mode"] = mode_row

# Re‑order columns for readability
ordered_cols = [
    "count",
    "mean",
    "median",
    "mode",
    "var",
    "std",
    "min",
    "max",
    "skew",
    "kurt",
]
desc = desc[ordered_cols]

desc.to_csv("summary_stats.csv", index_label="Measure")

# ---------------------------------------------------------------------------
# 4.  LOCAL AUTHORITIES ABOVE / BELOW NATIONAL MEAN
# ---------------------------------------------------------------------------

a_mean = desc.loc["T_5AC", "mean"]
above_mean = df_merged[df_merged["T_5AC"] > a_mean][["LA_name", "T_5AC"]]
below_mean = df_merged[df_merged["T_5AC"] < a_mean][["LA_name", "T_5AC"]]

above_mean.to_csv("la_above_below_mean.csv", index=False, mode="w")
below_mean.to_csv("la_above_below_mean.csv", index=False, mode="a", header=False)

# ---------------------------------------------------------------------------
# 5.  PAIRED t‑TEST
# ---------------------------------------------------------------------------

t_stat, p_val = stats.ttest_rel(
    df_merged["M_5AC"], df_merged["F_5AC"], nan_policy="omit"
)
with open("male_female_ttest.txt", "w") as fh:
    fh.write(
        "Paired t‑test – Boys vs Girls (% 5+ A*-C)\n"
        f"t-statistic = {t_stat:.3f}\n"
        f"p-value     = {p_val:.3e}\n"
    )

# ---------------------------------------------------------------------------
# 6.  VISUALISATIONS
# ---------------------------------------------------------------------------


def save_bar(series: pd.Series, title: str, ylabel: str, filename: str):
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
plt.axvline(a_mean, linestyle="--", linewidth=1)
plt.title("Distribution of % 5+ A*-C across LAs (All pupils)")
plt.xlabel("% of pupils")
plt.ylabel("Number of LAs")
plt.tight_layout()
plt.savefig(PLOT_DIR / "histogram_T_5AC.png")
plt.clf()

# 6c. Pie – national mean by gender
means = df_merged[["M_5AC", "F_5AC"]].mean()
plt.pie(means, labels=["Boys", "Girls"], autopct="%1.1f%%", startangle=90)
plt.title("National mean – % 5+ A*-C by gender")
plt.savefig(PLOT_DIR / "gender_pie.png")
plt.clf()

# 6d. Regression: Girls vs Boys performance
x, y = df_merged["M_5AC"].values, df_merged["F_5AC"].values
slope, intercept, r, p, _ = stats.linregress(x, y)

plt.scatter(x, y, alpha=0.7)
plt.plot(
    x,
    intercept + slope * x,
    linestyle="--",
    label=f"y = {slope:.2f}·x + {intercept:.1f}\n$r = {r:.2f}$",
)
plt.xlabel("Boys % 5+ A*-C")
plt.ylabel("Girls % 5+ A*-C")
plt.title("Linear regression: Girls vs Boys performance")
plt.legend()
plt.tight_layout()
plt.savefig(PLOT_DIR / "regression_FvsM.png")
plt.clf()

print("✅ Analysis complete – outputs saved to current folder.")
