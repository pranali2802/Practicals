import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Create dataset
data = {
    'Age': [20, 25, 30, 35, 40, 45, 50, 55],
    'Salary': [20000, 25000, 30000, 35000, 40000, 45000, 50000, 55000],
    'Experience': [1, 2, 3, 4, 5, 6, 7, 8]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Standardize the data
scaler = StandardScaler()
X = scaler.fit_transform(df)

# Apply PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# Display reduced data
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])

print("\nReduced Dataset:")
print(pca_df)

# Display explained variance
print("\nExplained Variance Ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal Variance Preserved:")
print(sum(pca.explained_variance_ratio_) * 100, "%")

# Visualize PCA
plt.scatter(pca_df['PC1'], pca_df['PC2'])
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA Visualization')
plt.show()