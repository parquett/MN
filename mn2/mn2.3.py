import numpy as np
from scipy.interpolate import lagrange, interp1d, BarycentricInterpolator
from scipy.interpolate import CubicSpline
from scipy.integrate import romberg
import matplotlib.pyplot as plt

# Read the dataset from a text file
data = np.genfromtxt('dataset_3.txt', delimiter=',', skip_header=1)

x_known = data[:, 0]
y_known = data[:, 1]

# Remove NaN values from the known data
known_mask = ~np.isnan(y_known)
x_known = x_known[known_mask]
y_known = y_known[known_mask]

# Generate interpolated x-values
x_interp = np.linspace(min(x_known), max(x_known), num=100)

# Lagrange interpolation
poly_lagrange = lagrange(x_known, y_known)
y_lagrange = poly_lagrange(x_interp)

# Piecewise linear interpolation
f_linear = interp1d(x_known, y_known, kind='linear')
y_linear = f_linear(x_interp)

# Newton interpolation
poly_newton = BarycentricInterpolator(x_known, y_known)
y_newton = poly_newton(x_interp)

# Cubic spline interpolation
spline = CubicSpline(x_known, y_known)
y_spline = spline(x_interp)

# Calculate area under the curve using Romberg integration
area_romberg = romberg(spline, min(x_known), max(x_known))


# Plot the original data and the interpolated data
plt.figure(figsize=(10, 6))
plt.plot(x_known, y_known, 'bo', label='Original Data')
#plt.plot(x_interp, y_lagrange, 'r-', label='Lagrange Interpolation')
#plt.plot(x_interp, y_linear, 'g--', label='Piecewise Linear Interpolation', alpha=0.7)
#plt.plot(x_interp, y_newton, 'm-.', label='Newton Interpolation', alpha=0.5)
plt.plot(x_interp, y_spline, 'c-', label='Cubic Spline Interpolation', alpha=0.6)
plt.xlabel('Number of Items Purchased')
plt.ylabel('Time Spent (minutes)')
plt.title('Interpolation Comparison')
plt.legend()
plt.grid(True)
plt.show()

print("Area under the curve using Romberg integration:", area_romberg)
