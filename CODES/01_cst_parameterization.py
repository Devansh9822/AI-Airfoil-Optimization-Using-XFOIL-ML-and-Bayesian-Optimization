import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from math import comb

# =====================================================
# READ AIRFOIL FILE
# =====================================================

filename = r"C:\naca4412.dat"

data = np.loadtxt(filename, skiprows=1)

x = data[:, 0]
y = data[:, 1]

# =====================================================
# SPLIT UPPER / LOWER
# =====================================================

le_index = np.argmin(x)

xu = x[:le_index + 1][::-1]
yu = y[:le_index + 1][::-1]

xl = x[le_index:]
yl = y[le_index:]

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
# FIT UPPER SURFACE
# =====================================================

def upper_residual(A):
    return cst_surface(xu, A) - yu

Au0 = np.zeros(5)

upper_fit = least_squares(
    upper_residual,
    Au0
)

Au = upper_fit.x

# =====================================================
# FIT LOWER SURFACE
# =====================================================

def lower_residual(A):
    return cst_surface(xl, A) - yl

Al0 = np.zeros(5)

lower_fit = least_squares(
    lower_residual,
    Al0
)

Al = lower_fit.x

# =====================================================
# PRINT CST VARIABLES
# =====================================================

print("\n========================")
print("CST VARIABLES")
print("========================")

for i, a in enumerate(Au):
    print(f"a{i} = {a:.6f}")

for i, b in enumerate(Al):
    print(f"b{i} = {b:.6f}")

# =====================================================
# RECONSTRUCT AIRFOIL
# =====================================================

x_plot = np.linspace(0, 1, 300)

yu_fit = cst_surface(x_plot, Au)
yl_fit = cst_surface(x_plot, Al)

# =====================================================
# FIT QUALITY CHECK
# =====================================================

yu_reconstructed = cst_surface(xu, Au)
yl_reconstructed = cst_surface(xl, Al)

rmse_upper = np.sqrt(
    np.mean(
        (yu - yu_reconstructed)**2
    )
)

rmse_lower = np.sqrt(
    np.mean(
        (yl - yl_reconstructed)**2
    )
)

max_error_upper = np.max(
    np.abs(yu - yu_reconstructed)
)

max_error_lower = np.max(
    np.abs(yl - yl_reconstructed)
)

print("\n========================")
print("FIT QUALITY")
print("========================")

print(f"Upper RMSE      = {rmse_upper:.8f}")
print(f"Lower RMSE      = {rmse_lower:.8f}")

print(f"Upper Max Error = {max_error_upper:.8f}")
print(f"Lower Max Error = {max_error_lower:.8f}")

# =====================================================
# THICKNESS CHECK
# =====================================================

thickness = np.max(yu_fit - yl_fit)

print("\n========================")
print("THICKNESS RATIO")
print("========================")
print(f"t/c = {thickness:.4f}")

if thickness >= 0.10:
    print("PASS : Thickness >= 10%")
else:
    print("FAIL : Thickness < 10%")

# =====================================================
# PLOT
# =====================================================

plt.figure(figsize=(12,5))

plt.plot(xu, yu, label='Original Upper')
plt.plot(xl, yl, label='Original Lower')

plt.plot(
    x_plot,
    yu_fit,
    '--',
    linewidth=2,
    label='CST Upper'
)

plt.plot(
    x_plot,
    yl_fit,
    '--',
    linewidth=2,
    label='CST Lower'
)

plt.axis('equal')
plt.grid(True)
plt.legend()

plt.title("NACA4412 : 10 Variable CST Fit")

plt.xlabel("x/c")
plt.ylabel("y/c")

plt.show()
