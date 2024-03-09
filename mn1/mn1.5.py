import numpy as np

def secant_bisection_method(f, a, b, tol=1e-8, max_iter=100):
    """
    Hybrid secant-bisection method for finding a root of f(x) = 0 in the interval [a, b].
    """
    # Check that the function changes sign in the interval
    if np.sign(f(a)) == np.sign(f(b)):
        raise ValueError("Function has the same sign at both endpoints of the interval.")

    # Initialize the method with the secant method
    x0 = a
    x1 = b
    fx0 = f(x0)
    fx1 = f(x1)

    for i in range(max_iter):
        # Compute the next approximation using the secant method
        x2 = x1 - fx1 * (x1 - x0) / (fx1 - fx0)
        fx2 = f(x2)

        # Check if the secant method is making progress
        if np.abs(fx2) < tol:
            return x2

        # If the secant method fails, use the bisection method
        if np.sign(fx2) == np.sign(fx0):
            a = x2
            b = x1
        else:
            a = x0
            b = x2

        # Compute the next approximation using the bisection method
        x3 = (a + b) / 2
        fx3 = f(x3)

        # Check if the bisection method is making progress
        if np.abs(fx3) < tol:
            return x3

        # Update the interval for the next iteration
        if np.sign(fx3) == np.sign(fx0):
            x0 = x3
            fx0 = fx3
        else:
            x1 = x3
            fx1 = fx3

    raise RuntimeError("Maximum number of iterations exceeded.")

def f(x):
    return x**3 - 2*x - 5

# Set the interval [a, b]
a = 2
b = 3

# Call the secant_bisection_method function with the function f and interval [a, b]
root = secant_bisection_method(f, a, b)

# Print the root
print("The root of the equation is:", root)