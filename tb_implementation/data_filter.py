#%%
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def create_diagonal_matrix(rows=91, cols=63, decay_rate=0.5):
    # Create a zero matrix
    matrix = np.zeros((rows, cols))

    # Fill the actual diagonal (from (0, 0) to (min(rows, cols), min(rows, cols)))
    for i in range(min(rows, cols)):
        matrix[i, i] = 1

    # Apply exponentially decreasing values to off-diagonal elements based on their distance
    for i in range(rows):
        for j in range(cols):
            if i != j:
                # Calculate the distance from the actual diagonal
                distance = abs(i - j)
                matrix[i, j] = np.exp(-decay_rate * distance)

    return matrix

def plot_matrix_3d(matrix):
    rows, cols = matrix.shape
    X, Y = np.meshgrid(range(cols), range(rows))
    Z = matrix

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z, cmap='viridis')

    ax.set_xlabel('Column Index')
    ax.set_ylabel('Row Index')
    ax.set_zlabel('Matrix Value')
    ax.set_title('3D Visualization of the Matrix')

    plt.show()

# Generate and print the matrix
matrix = create_diagonal_matrix()
print(matrix)

# Plot the matrix
plot_matrix_3d(matrix)

# %%
