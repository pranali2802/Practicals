import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([2, 4, 5, 4, 6])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Prediction
y_pred = model.predict(X)

# Display output
print("Actual Values:", y)
print("Predicted Values:", y_pred)

# Calculate errors
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))

# Graphical Representation
plt.scatter(X, y, color='blue', label='Actual Values')
plt.plot(X, y_pred, color='red', label='Predicted Line')

plt.xlabel("X Values")
plt.ylabel("Y Values")
plt.title("Linear Regression Graph")
plt.legend()
plt.grid()
plt.show()