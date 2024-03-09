import math

def bisection_method(func, a, b, tol, max_iterations=100):
    if func(a) * func(b) > 0:
        raise ValueError("Function must have opposite signs at the endpoints of the interval.")
    if tol <= 0:
        raise ValueError("Tolerance must be positive.")

    iterations = 0
    while iterations < max_iterations:
        c = (a + b) / 2
        if abs(func(c)) < tol:
            return c
        if func(c) * func(a) < 0:
            b = c
        else:
            a = c
        iterations += 1

    raise RuntimeError("Bisection method did not converge within the specified number of iterations.")

def func(x):
    return math.exp(x) - x**2

# Initial interval [-2, 0]
a = -2
b = 0
tolerance = 1e-8

# Call the bisection method to find the root
root = bisection_method(func, a, b, tolerance)

# Print the root with 8 decimal places of accuracy
print("Approximate root: {:.8f}".format(root))


