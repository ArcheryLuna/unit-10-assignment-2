"""
GCSE Results Analysis – Unit 10, Assignment 2  (v3 – robust header detection)
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

Major improvements over the earlier version
———————————————————————————
*  `load_clean()` now **dynamically locates** the real header row by
   scanning for the word "Region" in column 0. This avoids KeyErrors
   when blank rows are inserted above the header.
*  LA‑name and result columns are identified by **fuzzy matching** so
   the script is resilient to minor wording/line‑break changes.
*  Type‑hints and assertions provide early failure if the workbook
   structure is unexpected.
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

    The function is *header‑agnostic* – it finds the header row by
    scanning column 0 for the first cell containing the substring
    "Region".  It then:
       * renames the Local‑Authority column to LA_name
       * removes region‑level and England totals rows
       * coerces the chosen result column to numeric
    """
    raw = pd.read_excel(EXCEL_FILE, sheet_name=sheet_name, header=None)

    # --- locate header row --------------------------------------------------
    header_idx: int = next(
        idx
        for idx, val in enumerate(raw.iloc[:, 0])
        if isinstance(val, str) and "Region" in val
    )
    header = raw.iloc[header_idx]
    df = raw.iloc[header_idx + 1 :].copy()
    df.columns = header

    # --- identify the Local‑Authority column (robust to line breaks) --------
    la_col_candidates: List[str] = [
        c for c in df.columns if isinstance(c, str) and "Local Authority" in c
    ]
    if not la_col_candidates:
        raise RuntimeError(f"No 'Local Authority' column found in {sheet_name}")
    la_col: str = la_col_candidates[0]

    df = df.rename(columns={la_col: "LA_name"})
    df["LA_name"] = df["LA_name"].astype(str).str.replace("\n", " ").str.strip()

    # --- drop blank & region summary rows ----------------------------------
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

    # --- select result column (plain 5+A*-C, excluding English+Math note) ---
    res_col_candidates: List[str] = [
        c
        for c in df.columns
        if isinstance(c, str) and "5+A*-C" in c and "including" not in c.lower()
    ]
    if not res_col_candidates:
        raise RuntimeError(f"No suitable '5+A*-C' result column found in {sheet_name}")
    res_col: str = res_col_candidates[0]

    df[res_col] = pd.to_numeric(df[res_col], errors="coerce")

    tidy = df[["LA_name", res_col]].rename(columns={res_col: f"{sex_tag}_5AC"})
    return tidy


# ---------------------------------------------------------------------------
# 2.  BUILD MASTER DATAFRAME
# ---------------------------------------------------------------------------

df_boys = load_clean(SHEETS["M"], "M")
df_girls = load_clean(SHEETS["F"], "F")
df_all = load_clean(SHEETS["T"], "T")

# Merge on LA_name – outer join ensures we spot mismatches.
df_merged: pd.DataFrame = df_boys.merge(df_girls, on="LA_name", how="outer").merge(
    df_all, on="LA_name", how="outer"
)

# Sanity checks
missing_any: pd.DataFrame = df_merged[df_merged.isna().any(axis=1)]
if not missing_any.empty:
    print(
        "⚠️  Warning: some LAs missing data in one or more sheets – they will be dropped."
    )
    print(missing_any[["LA_name"]])
    df_merged = df_merged.dropna()

assert df_merged.columns.tolist() == [
    "LA_name",
    "M_5AC",
    "F_5AC",
    "T_5AC",
], "Unexpected columns after merge"
assert not df_merged.duplicated("LA_name").any(), "Duplicate LA_name rows found"

# ---------------------------------------------------------------------------
# 3.  DESCRIPTIVE STATISTICS
# ---------------------------------------------------------------------------
stats_cols = ["M_5AC", "F_5AC", "T_5AC"]

desc: pd.DataFrame = (
    df_merged[stats_cols]
    .agg(
        [
            "count",
            "mean",
            "median",
            pd.Series.mode,
            "var",
            "std",
            "min",
            "max",
            "skew",
            "kurt",
        ]
    )
    .T
)

desc.to_csv("summary_stats.csv", index_label="Measure")

# ---------------------------------------------------------------------------
# 4.  LOCAL AUTHORITIES ABOVE / BELOW NATIONAL MEAN
# ---------------------------------------------------------------------------

a_mean: float = desc.loc["T_5AC", "mean"]

above_mean = df_merged[df_merged["T_5AC"] > a_mean][["LA_name", "T_5AC"]]
below_mean = df_merged[df_merged["T_5AC"] < a_mean][["LA_name", "T_5AC"]]

above_mean.to_csv("la_above_below_mean.csv", index=False, mode="w")
below_mean.to_csv("la_above_below_mean.csv", index=False, mode="a", header=False)

# ---------------------------------------------------------------------------
# 5.  INFERENTIAL STATISTICS – PAIRED t‑TEST
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


def save_bar(series: pd.Series, title: str, ylabel: str, filename: str) -> None:
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
