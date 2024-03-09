import numpy as np

# Define the coefficients of the variables
coefficients = np.array([[2, -1], [3, 2], [1, 1], [4, -1], [1, -3]])
constants = np.array([4, 7, 3, 1, -2])

# Solve the system of equations
solution = np.linalg.lstsq(coefficients, constants, rcond=None)[0]

# Determine the order of stepping stones
order = np.argsort(solution) + 1

# Create a dictionary to map the order to the stone labels
stone_labels = {1: "Stone 1", 2: "Stone 2", 3: "Stone 3", 4: "Stone 4", 5: "Stone 5"}

# Print the order of stepping stones
print("The order of stepping stones is:")
for stone in order:
    print(stone_labels[stone])
