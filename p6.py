import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error

# Input data
X = np.array([
    [1, 2],
    [2, 3],
    [3, 4],
    [4, 5],
    [5, 6]
])

# Output data
y = np.array([10, 20, 30, 40, 50])

# Ridge Regression
ridge = Ridge()
ridge.fit(X, y)
ridge_pred = ridge.predict(X)

# Lasso Regression
lasso = Lasso()
lasso.fit(X, y)
lasso_pred = lasso.predict(X)

# Elastic Net Regression
elastic = ElasticNet()
elastic.fit(X, y)
elastic_pred = elastic.predict(X)

# Display results
print("Ridge Prediction:", ridge_pred)
print("Ridge MSE:", mean_squared_error(y, ridge_pred))

print("\nLasso Prediction:", lasso_pred)
print("Lasso MSE:", mean_squared_error(y, lasso_pred))

print("\nElastic Net Prediction:", elastic_pred)
print("Elastic Net MSE:", mean_squared_error(y, elastic_pred))