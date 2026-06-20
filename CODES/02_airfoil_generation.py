import numpy as np
import pandas as pd
import os
from math import comb

# =====================================================
# SETTINGS
# =====================================================

NUM_AIRFOILS = 500
MAX_ATTEMPTS = 50000

PERTURBATION = 0.90      # ±30%

SAVE_FOLDER = "generated_airfoils40"

os.makedirs(SAVE_FOLDER, exist_ok=True)

# =====================================================
# BASELINE CST COEFFICIENTS (NACA4412)
# =====================================================

Au_base = np.array([
    0.213228,
    0.269124,
    0.271072,
    0.235784,
    0.312352
])

Al_base = np.array([
   -0.136971,
   -0.026861,
   -0.057538,
   -0.000738,
   -0.031269
])

# =====================================================
# CST FUNCTIONS
# =====================================================

def class_function(x):
    return np.sqrt(x) * (1 - x)

def bernstein(n, i, x):
    return comb(n, i) * (x ** i) * ((1 - x) ** (n - i))

def cst_surface(x, A):

    n = len(A) - 1

    y = np.zeros_like(x)

    for i in range(n + 1):
        y += A[i] * bernstein(n, i, x)

    return class_function(x) * y

# =====================================================
# COSINE SPACING
# =====================================================

N_POINTS = 201

beta = np.linspace(0, np.pi, N_POINTS)

x = 0.5 * (1 - np.cos(beta))

# =====================================================
# VERIFY COSINE SPACING
# =====================================================

print("\nFirst 10 x locations:")
print(x[:10])

print("\nLast 10 x locations:")
print(x[-10:])

# =====================================================
# STORAGE
# =====================================================

database = []

generated = 0
attempts = 0

# =====================================================
# GENERATION LOOP
# =====================================================

while generated < NUM_AIRFOILS and attempts < MAX_ATTEMPTS:

    attempts += 1

    # ---------------------------------------------
    # RANDOM CST PERTURBATION
    # ---------------------------------------------

    Au_new = Au_base * (
        1 + np.random.uniform(
            -PERTURBATION,
             PERTURBATION,
             size=5
        )
    )

    Al_new = Al_base * (
        1 + np.random.uniform(
            -PERTURBATION,
             PERTURBATION,
             size=5
        )
    )

    # ---------------------------------------------
    # GENERATE AIRFOIL
    # ---------------------------------------------

    yu = cst_surface(x, Au_new)
    yl = cst_surface(x, Al_new)

    # ---------------------------------------------
    # THICKNESS DISTRIBUTION
    # ---------------------------------------------

    thickness_distribution = yu - yl

    thickness = np.max(thickness_distribution)

    if thickness < 0.10:
        continue

    # ---------------------------------------------
    # SURFACE CROSSING CHECK
    # ---------------------------------------------

    min_gap = np.min(thickness_distribution)

    if min_gap < -1e-5:
        continue

    # ---------------------------------------------
    # CAMBER
    # ---------------------------------------------

    camber_line = (yu + yl) / 2

    max_camber = np.max(camber_line)

    camber_location = x[
        np.argmax(camber_line)
    ]

    # ---------------------------------------------
    # THICKNESS LOCATION
    # ---------------------------------------------

    thickness_location = x[
        np.argmax(thickness_distribution)
    ]

    # ---------------------------------------------
    # SAVE DAT FILE
    # ---------------------------------------------

    filename = os.path.join(
        SAVE_FOLDER,
        f"airfoil_{generated:04d}.dat"
    )

    with open(filename, "w") as f:

        f.write(
            f"airfoil_{generated:04d}\n"
        )

        # Upper Surface

        for xx, yy in zip(
            x[::-1],
            yu[::-1]
        ):
            f.write(
                f"{xx:.6f} {yy:.6f}\n"
            )

        # Lower Surface

        for xx, yy in zip(
            x[1:],
            yl[1:]
        ):
            f.write(
                f"{xx:.6f} {yy:.6f}\n"
            )

    # ---------------------------------------------
    # SAVE DATABASE ENTRY
    # ---------------------------------------------

    row = {

        "ID": generated,

        "Thickness": thickness,

        "Thickness_Location":
        thickness_location,

        "Max_Camber":
        max_camber,

        "Camber_Location":
        camber_location
    }

    for i, val in enumerate(Au_new):
        row[f"a{i}"] = val

    for i, val in enumerate(Al_new):
        row[f"b{i}"] = val

    database.append(row)

    generated += 1

    if generated % 25 == 0:

        print(
            f"Generated {generated} airfoils"
        )

# =====================================================
# SAVE CSV
# =====================================================

df = pd.DataFrame(database)

df.to_csv(
    "airfoil_database90.csv",
    index=False
)

# =====================================================
# SUMMARY
# =====================================================

print("\n========================")
print("GENERATION COMPLETE")
print("========================")

print(f"Attempts  : {attempts}")
print(f"Generated : {generated}")

print("\nCSV Saved:")
print("airfoil_database90.csv")

print("\nDAT Files Saved In:")
print(SAVE_FOLDER)
