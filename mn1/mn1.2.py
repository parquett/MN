import numpy as np

# Define the function g(x)
def g(x, c):
    return 2*x - c*x**2

# Define the fixed-point iteration function
def fixed_point_iteration(p0, c, tol=1e-8, max_iter=100):
    # Initialize variables
    p = p0
    i = 0
    # Perform iteration
    while i < max_iter:
        p_new = g(p, c)
        if abs(p_new - p) < tol:
            # Convergence reached
            return p_new
        p = p_new
        i += 1
    # Max number of iterations reached without convergence
    raise ValueError("Maximum number of iterations reached without convergence.")

# Test the fixed-point iteration method for different values of c
c_values = [0.5, 1.0, 2.0]
for c in c_values:
    # Find the limit using the fixed-point iteration method
    p_limit = fixed_point_iteration(0.5, c)
    # Check if the limit is equal to 1/c within a tolerance
    assert np.isclose(p_limit, 1/c, rtol=1e-6)
    print(f"The limit for c={c} is {p_limit}.")

# For each value of c, the fixed_point_iteration function returns
# the limit of the fixed-point iteration method. The code then checks
# if the returned limit is close enough to 1/c, using the np.isclose()
# function with a relative tolerance of 1e-6.
#
# If the test passes for all values of c, the code prints the limit
# for each value of c. If the test fails for any value of c, the
# assert statement raises an error and the code stops executing.






# import numpy as np
# import matplotlib.pyplot as plt
#
# # Define the function g(x)
# def g(x, c):
#     return 2*x - c*x**2
#
# # Define the derivative of g(x)
# def g_prime(x, c):
#     return 2 - 2*c*x
#
# # Define the function f(x) = g(x) - x
# def f(x, c):
#     return g(x, c) - x
#
# # Define the derivative of f(x)
# def f_prime(x, c):
#     return g_prime(x, c) - 1
#
# # Define the Newton's method function
# def newton_method(x0, c, tol=1e-8, max_iter=100):
#     # Initialize variables
#     x = x0
#     i = 0
#     # Perform iteration
#     while i < max_iter:
#         x_new = x + (c*x**2 - 3*x) / (2*c*x - 3)
#         if abs(x_new - x) < tol:
#             # Convergence reached
#             return x_new
#         x = x_new
#         i += 1
#     # Max number of iterations reached without convergence
#     raise ValueError("Maximum number of iterations reached without convergence.")
#
# # Test the Newton's method for different values of c
# c_values = [0.5, 1.0, 2.0]
# for c in c_values:
#     # Plot the function f(x) and its derivative f
#     x_values = np.linspace(0, 2 / c, 1000)
#     f_values = f(x_values, c)
#     f_prime_values = f_prime(x_values, c)
#     plt.plot(x_values, f_values, label=r"$f(x)$")
#     plt.plot(x_values, f_prime_values, label=r"$f'(x)$")
#     plt.axhline(y=0, color='k', linestyle='--')
#     plt.axvline(x=1 / c, color='r', linestyle='--')
#     plt.title(f"Newton's method for c = {c}")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.legend()
#     plt.show()
#     # Test the Newton's method
#     x0 = 1 / c
#     x = newton_method(x0, c)
#     assert np.isclose(x, 1 / c, rtol=1e-8)
#     print(f"Limit for c = {c}: {x}")
