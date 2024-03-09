# import numpy as np
# import sys
# def newton_raphson(f, J, x0, tol=1e-6, max_iter=100):
#     x = x0
#     for i in range(max_iter):
#         f_x = f(x)
#         J_x = J(x)
#         if np.linalg.cond(J_x) < 1/sys.float_info.epsilon:
#             # Add a small perturbation to the Jacobian matrix to handle singularities
#             J_x = J_x + 1e-6*np.eye(len(x))
#         delta_x = np.linalg.solve(J_x, -f_x)
#         x = x + delta_x
#         if np.linalg.norm(delta_x) < tol:
#             return x
#     raise Exception("Failed to converge after {} iterations".format(max_iter))
#
#
# # Example usage
# if __name__ == "__main__":
#     # Define the system of equations and its Jacobian
#     f = lambda x: np.array([x[0]**2 + x[1]**2 - 1, x[0]*x[1] - 1])
#     J = lambda x: np.array([[2*x[0], 2*x[1]], [x[1], x[0]]])
#
#     # Choose an initial guess
#     x0 = np.array([1.0, 1.0])
#
#     # Find the root of the system of equations
#     root = newton_raphson(f, J, x0, tol=1e-8)
#
#     # Print the root
#     print(f"The root is {root}")

import numpy as np


def newton_raphson_system(f, J, x0, tol=1e-6, max_iter=100):
    """
    Finds the roots of a system of nonlinear equations using the Newton-Raphson method.

    Parameters:
        f (callable): A function that takes a vector x as input and returns a vector f(x).
        J (callable): A function that takes a vector x as input and returns the Jacobian matrix J(x) of f(x).
        x0 (array-like): The initial guess for the root(s).
        tol (float, optional): The desired tolerance. The default is 1e-6.
        max_iter (int, optional): The maximum number of iterations. The default is 100.

    Returns:
        x (numpy.ndarray): An array containing the roots of the system of nonlinear equations.
    """
    x = np.asarray(x0)  # Convert the initial guess to a numpy array

    for i in range(max_iter):
        f_val = f(x)
        J_val = J(x)
        delta_x = np.linalg.solve(J_val, -f_val)
        x = x + delta_x
        if np.linalg.norm(delta_x) < tol:
            return x

    raise RuntimeError("The Newton-Raphson method did not converge in {} iterations".format(max_iter))

# Define the system of equations
def f(x):
    return np.array([
        x[0]**2 + x[1]**2 - 1,
        x[0] - x[1]**2
    ])

# Define the Jacobian of the system of equations
def J(x):
    return np.array([
        [2*x[0], 2*x[1]],
        [1, -2*x[1]]
    ])

# Find the roots of the system of equations
x0 = [1.0, 1.0]
x = newton_raphson_system(f, J, x0)

print("The roots are:", x)
