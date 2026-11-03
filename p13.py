import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# Create dataset
data = {
    'Age': [20, 21, 22, 23, 24, 40, 41, 42, 43, 44, 80],
    'Salary': [20000, 21000, 22000, 23000, 24000,
               40000, 41000, 42000, 43000, 44000, 90000]
}

df = pd.DataFrame(data)

print("Original Dataset:")
print(df)

# Scale the data
X = StandardScaler().fit_transform(df)

# Apply DBSCAN
model = DBSCAN(eps=0.8, min_samples=2)

df['Cluster'] = model.fit_predict(X)

print("\nClustered Data:")
print(df)

# Display outliers
print("\nOutliers:")
print(df[df['Cluster'] == -1])

# Visualize clusters
plt.scatter(df['Age'], df['Salary'], c=df['Cluster'])
plt.xlabel('Age')
plt.ylabel('Salary')
plt.title('DBSCAN Clustering')
plt.show()