import numpy as np

# Read the matrix data from the text file
with open('matrix.txt', 'r') as file:
    lines = file.readlines()

# Extract the individuals' names from the first row
individuals = lines[0][6:-2].split("', '")

# Remove the first row from the data
lines = lines[1:]

# Create an empty connectivity matrix
A = np.zeros((len(individuals), len(individuals)))

# Fill in the connectivity matrix based on the data
for i, line in enumerate(lines):
    values = line.strip().split(", ")[1:]
    for j, value in enumerate(values):
        A[i, j] = int(value.strip('[],'))

# Compute eigenvalues and eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(A)

# Find the dominant eigenvalue and its corresponding eigenvector
dominant_index = np.argmax(eigenvalues)
dominant_eigenvalue = eigenvalues[dominant_index]
dominant_eigenvector = eigenvectors[:, dominant_index]

# Print the dominant eigenvalue
print("Dominant Eigenvalue:", dominant_eigenvalue)

# Print the corresponding eigenvector and its associated individuals
print("Dominant Eigenvector (Opinion Leaders):")
for i, value in enumerate(dominant_eigenvector):
    print(individuals[i], ":", value*-1)

