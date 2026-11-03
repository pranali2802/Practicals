import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Create dataset
data = {
    'Age': [20, 22, 25, 30, 35, 40, 45, 50, 55, 60],
    'Salary': [20000, 22000, 25000, 30000, 35000,
               40000, 45000, 50000, 55000, 60000]
}

df = pd.DataFrame(data)
print(df)

# Select features
X = df[['Age', 'Salary']]

# Elbow Method
wcss = []

for k in range(1, 6):
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    model.fit(X)
    wcss.append(model.inertia_)

plt.plot(range(1, 6), wcss, marker='o')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')
plt.title('Elbow Method')
plt.show()

# Apply K-Means
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X)

print("\nClustered Data:")
print(df)

# Visualize clusters
plt.scatter(df['Age'], df['Salary'], c=df['Cluster'])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('K-Means Clustering')
plt.show()