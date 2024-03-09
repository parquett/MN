import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Define the ODEs for resource utilisation and resource allocation
def resource_odes(t, y, A, D):
    U, R = y
    dUdt = A(t) - D(t)
    dRdt = A(t) - D(t)
    return [dUdt, dRdt]

# Define the arrival rate and departure rate functions
def arrival_rate(t):
    return 2 * np.sin(t)

def departure_rate(t):
    return np.cos(t)

# Define the time span for integration
t_start = 0
t_end = 10
num_points = 100
t_span = np.linspace(t_start, t_end, num_points)

# Set initial conditions for resource utilisation and resource allocation
U0 = 0
R0 = 100

# Solve the ODEs numerically
solution = solve_ivp(resource_odes, [t_start, t_end], [U0, R0], t_eval=t_span, args=(arrival_rate, departure_rate))

# Extract the solution
U = solution.y[0]
R = solution.y[1]

# Plot the resource utilisation over time
plt.plot(t_span, U, label='Resource Utilisation')
plt.xlabel('Time')
plt.ylabel('Resource Utilisation')
plt.legend()
plt.title('Resource Utilisation over Time')
plt.grid(True)
plt.show()
