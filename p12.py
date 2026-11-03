import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage

# Create dataset
data = {
    'Age': [20, 22, 25, 30, 35, 40, 45, 50],
    'Salary': [20000, 22000, 25000, 30000, 35000, 40000, 45000, 50000]
}

df = pd.DataFrame(data)
print(df)

# Create dendrogram
X = df[['Age', 'Salary']]

plt.figure(figsize=(8, 5))
dendrogram(linkage(X, method='ward'))
plt.title('Dendrogram')
plt.xlabel('Data Points')
plt.ylabel('Distance')
plt.show()

# Apply Agglomerative Clustering
model = AgglomerativeClustering(n_clusters=3, linkage='ward')

df['Cluster'] = model.fit_predict(X)

print("\nClustered Data:")
print(df)

# Visualize clusters
plt.scatter(df['Age'], df['Salary'], c=df['Cluster'])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('Hierarchical Clustering')
plt.show()