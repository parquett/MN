import numpy as np

# Define the function
def f(x):
    return x**3 + 2*x**2 + 10*x - 20

# Define Muller's method
def muller(f, x0, x1, x2, tol=1e-8, max_iter=100):
    for i in range(max_iter):
        h0 = x1 - x0
        h1 = x2 - x1
        delta0 = (f(x1) - f(x0))/h0
        delta1 = (f(x2) - f(x1))/h1
        a = (delta1 - delta0)/(h1 + h0)
        b = a*h1 + delta1
        c = f(x2)
        x3 = x2 - 2*c/(b + np.sign(b)*np.sqrt(b**2 - 4*a*c))
        if abs(x3 - x2) < tol:
            return x3
        x0, x1, x2 = x1, x2, x3
    raise ValueError(f"Failed to converge after {max_iter} iterations")

# Call Muller's method with initial guesses
root = muller(f, 0, 1, 2)

# Print the result
print(f"The root is {root:.8f}")
