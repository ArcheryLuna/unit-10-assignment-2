#!/usr/bin/env sh
# ----------------------------------------------------------------------------
# POSIX‑compliant installer for the GCSE Analysis project (Unit 10 – Assignment 2)
# -----------------------------------------------------------------------------
# Creates an isolated Python virtual environment in ./venv and installs the
# minimal package set required by gcse_analysis.py.
#
# Usage:
#   sh install.sh        # or ./install.sh if executable bit is set
# -----------------------------------------------------------------------------
set -eu

# -------- SETTINGS -----------------------------------------------------------
VENV_DIR="venv"        # folder name for the virtual environment
REQUIRED_PY="3.9"      # minimum Python version
PACKAGES="pandas numpy scipy matplotlib seaborn"  # space‑separated list

# -------- HELPER: compare versions ------------------------------------------
ver_ge() {
    # Returns 0 if $1 >= $2  (both dotted numeric versions)
    awk -v v1="$1" -v v2="$2" '
        BEGIN {
            n = split(v1,a,"."); m = split(v2,b,".");
            max = (n>m?n:m);
            for(i=1;i<=max;i++) {
                x = (i in a) ? a[i] : 0;
                y = (i in b) ? b[i] : 0;
                if (x+0 > y+0) exit 0;
                if (x+0 < y+0) exit 1;
            }
            exit 0;
        }'
}

# -------- 1. Locate python3 --------------------------------------------------
if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
else
    echo "Error: python3 not found in PATH. Please install Python $REQUIRED_PY+ and retry." >&2
    exit 1
fi

PY_VER="$($PYTHON_BIN -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
if ! ver_ge "$PY_VER" "$REQUIRED_PY"; then
    echo "Error: Python $REQUIRED_PY or later required (found $PY_VER)." >&2
    exit 1
fi

# -------- 2. Create virtual environment -------------------------------------
if [ -d "$VENV_DIR" ]; then
    echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
    echo "Creating virtual environment in ./$VENV_DIR …"
    "$PYTHON_BIN" -m venv "$VENV_DIR" || {
        echo "Failed to create venv. Ensure the 'venv' module is available." >&2
        exit 1
    }
fi

# -------- 3. Activate & upgrade pip -----------------------------------------
# shellcheck source=/dev/null
. "$VENV_DIR/bin/activate"

printf '\nUpgrading pip, setuptools, wheel …\n'
python -m pip install --upgrade --quiet pip setuptools wheel

# -------- 4. Install project dependencies -----------------------------------
echo "Installing project packages: $PACKAGES …"
python -m pip install --quiet $PACKAGES

# -------- 5. Freeze (+ local dev convenience) -------------------------------
python -m pip freeze > requirements.txt

cat <<EOF
\n✔ Installation complete.
To activate the environment in future sessions, run:
    . $VENV_DIR/bin/activate
Then execute the analysis script with:
    python gcse_analysis.py
EOF

