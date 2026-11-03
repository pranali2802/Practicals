from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Apply Decision Tree
for depth in [1, 2, 3, 5]:
    model = DecisionTreeClassifier(max_depth=depth, random_state=42)

    # Train model
    model.fit(X_train, y_train)

    # Predict output
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("Maximum Depth:", depth)
    print("Accuracy:", accuracy)
    print()