import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Input data (Hours and Attendance)
X = np.array([
    [2, 60],
    [3, 65],
    [4, 70],
    [5, 80],
    [6, 90]
])

# Output data (Marks)
y = np.array([40, 50, 60, 70, 85])

# Create model
model = LinearRegression()

# Train model
model.fit(X, y)

# Predict values
y_pred = model.predict(X)

# Display results
print("Actual Values:", y)
print("Predicted Values:", y_pred)

# Calculate errors
print("MAE:", mean_absolute_error(y, y_pred))
print("MSE:", mean_squared_error(y, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y, y_pred)))
print("R2 Score:", r2_score(y, y_pred))