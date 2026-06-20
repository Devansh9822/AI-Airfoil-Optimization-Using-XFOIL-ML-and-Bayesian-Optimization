import os
import subprocess
import pandas as pd

# =====================================================
# PATHS
# =====================================================

XFOIL_PATH = r"C:\XFOIL\xfoil.exe"
AIRFOIL_FOLDER = r"C:\DAT_FILES"
RESULTS_FOLDER = r"C:\40\RESULTS_40"
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# =====================================================
# SETTINGS
# =====================================================

RE = 500000
MACH = 0.1

# =====================================================
# LOAD ALL AIRFOILS
# =====================================================

airfoil_files = sorted(
    [
        f
        for f in os.listdir(AIRFOIL_FOLDER)
        if f.endswith(".dat")
    ]
)

# =====================================================
# BATCH SELECTION
# =====================================================

START_INDEX = 401
END_INDEX = 500

airfoil_files = airfoil_files[START_INDEX:END_INDEX]

print(f"Found {len(airfoil_files)} airfoils")

# =====================================================
# STORAGE
# =====================================================

results = []

# =====================================================
# MAIN LOOP
# =====================================================

for i, airfoil in enumerate(airfoil_files):

    print(
        f"\nProcessing {airfoil} "
        f"({i+1}/{len(airfoil_files)})"
    )

    airfoil_path = os.path.join(
        AIRFOIL_FOLDER,
        airfoil
    )

    polar_file = os.path.join(
        RESULTS_FOLDER,
        airfoil.replace(
            ".dat",
            "_polar.txt"
        )
    )

    commands = f"""
LOAD {airfoil_path}
PANE
OPER
VISC {RE}
MACH {MACH}
ITER 200
PACC
{polar_file}

ASEQ 0 10 1

PACC

QUIT
"""

    try:

        subprocess.run(
            XFOIL_PATH,
            input=commands,
            text=True,
            capture_output=True,
            timeout=180
        )

        results.append(
            {
                "Airfoil": airfoil,
                "Status": "DONE"
            }
        )

    except Exception as e:

        print(f"FAILED: {airfoil}")

        results.append(
            {
                "Airfoil": airfoil,
                "Status": "FAILED",
                "Error": str(e)
            }
        )

# =====================================================
# SAVE SUMMARY
# =====================================================

summary_file = os.path.join(
    RESULTS_FOLDER,
    f"batch_{START_INDEX}_{END_INDEX}.csv"
)

pd.DataFrame(results).to_csv(
    summary_file,
    index=False
)

print("\n================================")
print("FINISHED")
print("================================")
print(f"Summary saved to:\n{summary_file}")
