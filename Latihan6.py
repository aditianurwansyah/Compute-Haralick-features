import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
 
# Define larger (8x8) example matrices
matrix1 = np.array([
    [0, 0, 0, 0, 10, 10, 10, 10],
    [0, 0, 0, 0, 10, 10, 10, 10],
    [0, 0, 0, 0, 10, 10, 10, 10],
    [0, 0, 0, 0, 10, 10, 10, 10],
    [100, 100, 100, 100, 200, 200, 200, 200],
    [100, 100, 100, 100, 200, 200, 200, 200],
    [100, 100, 100, 100, 200, 200, 200, 200],
    [100, 100, 100, 100, 200, 200, 200, 200]
], dtype=np.uint8)
 
matrix2 = np.linspace(0, 255, 64, dtype=np.uint8).reshape((8, 8))
 
# Compute Haralick features
def compute_haralick(mat, distances=[1], angles=[0]):
    glcm = graycomatrix(mat, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)
    return {
        'contrast': graycoprops(glcm, 'contrast')[0,0],
        'homogeneity': graycoprops(glcm, 'homogeneity')[0,0],
        'energy': graycoprops(glcm, 'energy')[0,0],
        'correlation': graycoprops(glcm, 'correlation')[0,0],
        'dissimilarity': graycoprops(glcm, 'dissimilarity')[0,0]
    }
 
feat1 = compute_haralick(matrix1)
feat2 = compute_haralick(matrix2)
 
# Plot heatmaps
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(matrix1, interpolation='nearest')
axs[0].set_title("Matrix 1 (8×8)")
axs[0].axis('off')
 
axs[1].imshow(matrix2, interpolation='nearest')
axs[1].set_title("Matrix 2 (8×8)")
axs[1].axis('off')
 
plt.tight_layout()
plt.show()
 
# Compute features
feat1 = compute_haralick(matrix1)
feat2 = compute_haralick(matrix2)
 
# Print results in a table format
print(f"{'Feature':<15} {'Matrix1':>10} {'Matrix2':>10}")
print('-' * 37)
for key in feat1:
    print(f"{key:<15} {feat1[key]:>10.4f} {feat2[key]:>10.4f}")