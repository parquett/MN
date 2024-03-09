import numpy as np
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
def lagrange_interpolation(x, y, x_interp):
    n = len(x)
    m = len(x_interp)
    y_interp = np.zeros(m)
    for j in range(m):
        p = np.ones(n)
        for k in range(n):
            for i in range(n):
                if i != k:
                    p[k] *= (x_interp[j] - x[i]) / (x[k] - x[i])
        y_interp[j] = np.dot(y, p)
    return y_interp


y_lagrange = lagrange_interpolation(x_known, y_known, x_interp)


# Piecewise linear interpolation
def piecewise_linear_interpolation(x, y, x_interp):
    n = len(x)
    m = len(x_interp)
    y_interp = np.zeros(m)
    for j in range(m):
        for i in range(1, n):
            if x[i - 1] <= x_interp[j] <= x[i]:
                y_interp[j] = y[i - 1] + (y[i] - y[i - 1]) * (x_interp[j] - x[i - 1]) / (x[i] - x[i - 1])
                break
    return y_interp


y_linear = piecewise_linear_interpolation(x_known, y_known, x_interp)


# Newton interpolation
def divided_difference(x, y):
    n = len(x)
    coefficients = np.copy(y)
    for j in range(1, n):
        for i in range(n - 1, j - 1, -1):
            coefficients[i] = (coefficients[i] - coefficients[i - 1]) / (x[i] - x[i - j])
    return coefficients


def newton_interpolation(x, y, x_interp):
    n = len(x)
    m = len(x_interp)
    coefficients = divided_difference(x, y)
    y_interp = np.zeros(m)
    for j in range(m):
        p = 1
        for i in range(n):
            y_interp[j] += coefficients[i] * p
            p *= (x_interp[j] - x[i])
    return y_interp


y_newton = newton_interpolation(x_known, y_known, x_interp)


# Cubic spline interpolation
def cubic_spline_interpolation(x, y, x_interp):
    n = len(x)
    m = len(x_interp)
    h = x[1:] - x[:-1]
    a = np.copy(y)
    l = np.zeros(n - 1)
    u = np.zeros(n - 1)
    z = np.zeros(n)
    c = np.zeros(n)
    b = np.zeros(n - 1)
    d = np.zeros(n - 1)

    for i in range(1, n - 1):
        l[i] = 2 * (x[i + 1] - x[i - 1]) - h[i - 1] * u[i - 1]
        u[i] = h[i] / l[i]
        z[i] = (3 * (a[i + 1] - a[i]) / h[i] - 3 * (a[i] - a[i - 1]) / h[i - 1]) / l[i]

    for i in range(n - 2, 0, -1):
        c[i] = z[i] - u[i] * c[i + 1]
        b[i] = (a[i + 1] - a[i]) / h[i] - h[i] * (c[i + 1] + 2 * c[i]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h[i])

    y_interp = np.zeros(m)

    for j in range(m):
        for i in range(1, n):
            if x_interp[j] <= x[i]:
                y_interp[j] = a[i - 1] + b[i - 1] * (x_interp[j] - x[i - 1]) + c[i - 1] * (
                            x_interp[j] - x[i - 1]) ** 2 + d[i - 1] * (x_interp[j] - x[i - 1]) ** 3
                break
    return y_interp

y_spline = cubic_spline_interpolation(x_known, y_known, x_interp)

# Calculate area under the curve using Romberg integration
def romberg_integration(x, y):
    n = len(x)
    h = x[1:] - x[:-1]
    area = np.sum((y[:-1] + y[1:]) / 2 * h)
    return area

area_romberg = romberg_integration(x_known, y_known)

# Plot the original data and the interpolated data
plt.figure(figsize=(10, 6))
plt.plot(x_known, y_known, 'bo', label='Original Data')
plt.plot(x_interp, y_lagrange, 'r-', label='Lagrange Interpolation')
plt.plot(x_interp, y_linear, 'g--', label='Piecewise Linear Interpolation')
plt.plot(x_interp, y_newton, 'm-.', label='Newton Interpolation')
plt.plot(x_interp, y_spline, 'c-', label='Cubic Spline Interpolation')
plt.xlabel('Number of Items Purchased')
plt.ylabel('Time Spent (minutes)')
plt.title('Interpolation Comparison')
plt.legend()
plt.grid(True)
plt.show()

print("Area under the curve using Romberg integration:", area_romberg)

