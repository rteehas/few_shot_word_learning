import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# load the 4 numpy arrays from the directory temp_embeddings
A = np.load('temp_embeddings/embeddings_A.npy') # (1, 1024)
B = np.load('temp_embeddings/embeddings_B.npy') # (1, 1024)
C = np.load('temp_embeddings/embeddings_C.npy') # (49, 1, 1024)
D = np.load('temp_embeddings/embeddings_D.npy') # (49, 1, 1024)

A_minus_B = A - B
B_minus_A = B - A
D_plus_A_minus_B = D + (A - B)
C_plus_B_minus_A = C + (B - A)

# print the shape of the arrays
print("Shape of A_minus_B:", A_minus_B.shape)
print("Shape of B_minus_A:", B_minus_A.shape)
print("Shape of D_plus_A_minus_B:", D_plus_A_minus_B.shape)
print("Shape of C_plus_B_minus_A:", C_plus_B_minus_A.shape)

all_embeddings = np.concatenate((np.vstack(C), np.vstack(D), np.vstack(D_plus_A_minus_B), np.vstack(C_plus_B_minus_A)), axis=0)
tsne_results = TSNE(n_components=2, perplexity=10, random_state=0).fit_transform(all_embeddings)

# Plotting
plt.figure(figsize=(10, 6))
num_C = len(C)
num_D = len(D)
plt.scatter(tsne_results[:num_C, 0], tsne_results[:num_C, 1], c='blue', label='C')
plt.scatter(tsne_results[num_C:num_C+num_D, 0], tsne_results[num_C:num_C+num_D, 1], c='red', label='D')
plt.scatter(tsne_results[num_C+num_D:num_C+num_D+num_D, 0], tsne_results[num_C+num_D:num_C+num_D+num_D, 1], c='green', label='A - B + D (C approx)')
plt.scatter(tsne_results[num_C+num_D+num_D:, 0], tsne_results[num_C+num_D+num_D:, 1], c='purple', label='B - A + C (D approx)')
plt.legend()
plt.title('T-SNE of Embeddings')

# Save the plot
plt.savefig('embeddings_diff_tsne3.png')
