# GCSE Results Analysis – Unit 10, Assignment 2

A fully‑automated, reproducible Python workflow for analysing the **Table 16** GCSE dataset (Boys, Girls, All worksheets). It cleans the raw Excel file, performs descriptive and inferential statistics, and generates publication‑ready plots.

---

## Project structure

```text
.
├── gcse_analysis.py   # main analysis script
├── install.sh         # POSIX installer (bash/sh)
├── install.cmd        # Windows CMD installer
├── Data/
│   └── Santized GCSE Data.xlsx  # ← place the dataset here
├── outputs/
└── plots/             # created on first run, holds PNGs
```

---

## Requirements

| Component  | Minimum version |
| ---------- | --------------- |
| Python     | 3.9             |
| pandas     | 2.0             |
| numpy      | –               |
| scipy      | –               |
| matplotlib | –               |

_No admin rights needed – everything installs into a local virtual environment (`./venv`)._

---

## Installation

### Unix / macOS / WSL

```sh
sh install.sh            # or chmod +x install.sh && ./install.sh
```

### Windows (Command Prompt)

```cmd
install.cmd              :: double‑click or run from cmd.exe
```

Both scripts:

1. Check for Python 3.9+
2. Create `./venv`
3. Upgrade **pip**/setuptools/wheel
4. Install `pandas numpy scipy matplotlib`
5. Snapshot exact versions to `requirements.txt`

---

## Usage

Activate the environment, then execute the analysis script:

```sh
# POSIX
. venv/bin/activate
python gcse_analysis.py

# Windows CMD
call venv\Scripts\activate.bat
python gcse_analysis.py
```

Expected console output:

```
✅ Analysis complete – outputs saved to current folder.
```

### Outputs

| File / folder             | Description                            |
| ------------------------- | -------------------------------------- |
| `summary_stats.csv`       | Central‑tendency & dispersion measures |
| `la_above_below_mean.csv` | LAs better / worse than national mean  |
| `male_female_ttest.txt`   | Paired t‑test (Boys vs Girls)          |
| `plots/*.png`             | Bar, histogram, pie, regression charts |

---

## Updating dependencies

```
. venv/bin/activate          # or call venv\Scripts\activate.bat
pip install --upgrade pandas numpy scipy matplotlib
pip freeze > requirements.txt
```

---

## Troubleshooting

| Symptom                                 | Fix                                                   |
| --------------------------------------- | ----------------------------------------------------- |
| `python3: command not found`            | Install Python ≥ 3.9 and ensure it’s in **PATH**      |
| `ModuleNotFoundError: pandas`           | Reactivate venv or rerun installer                    |
| `ValueError: cannot combine transform…` | Use gcse_analysis **v3.1** or later (mode calc fixed) |

---
