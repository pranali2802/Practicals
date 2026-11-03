import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline

# Dataset
X = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
y = np.array([2, 4, 6, 8, 10, 12])

# Different polynomial degrees
for d in [1, 2, 5]:

    model = make_pipeline(
        PolynomialFeatures(degree=d),
        LinearRegression()
    )

    model.fit(X, y)

    prediction = model.predict(X)

    print("Degree:", d)
    print("Prediction:", prediction)

    plt.plot(X, prediction, label="Degree " + str(d))

plt.scatter(X, y)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Polynomial Regression")
plt.legend()
plt.show()