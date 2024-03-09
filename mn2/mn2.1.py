import numpy as np

def gauss_legendre_integration(f, a, b, tol):
    # Gauss-Legendre quadrature weights and abscissae for n = 4
    weights = np.array([0.3478548451, 0.6521451549, 0.6521451549, 0.3478548451])
    abscissae = np.array([-0.8611363116, -0.3399810436, 0.3399810436, 0.8611363116])

    # Initialize variables
    h = (b - a) / 2  # Step size
    integral_old = float('inf')
    n = 1  # Number of function evaluations

    while True:
        integral = 0.0
        for i in range(len(weights)):
            x = (a + b) / 2 + h * abscissae[i]  # Map abscissae to the integration interval
            integral += weights[i] * f(x)  # Evaluate function at each abscissa

        integral *= h  # Scale by step size

        if abs(integral - integral_old) < tol:
            break

        integral_old = integral
        n *= 2
        h /= 2

    return integral, n

# Example usage
def equations_of_motion(x):
    # Replace this with your actual equations of motion
    return x**2 + 2 * x - 1

a = 0.0  # Lower limit of integration
b = 2.0  # Upper limit of integration
tolerance = 1e-6  # Desired tolerance

integral_approx, num_evaluations = gauss_legendre_integration(equations_of_motion, a, b, tolerance)

print("Approximate value of integral:", integral_approx)
print("Number of function evaluations:", num_evaluations)
