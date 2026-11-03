import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create dataset
data = {
    'Age': [20, 25, 30, 35, 40, 45, 50, 55, 60, 65],
    'Salary': [20000, 25000, 30000, 35000, 40000,
               45000, 50000, 55000, 60000, 65000],
    'Purchased': [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Split data into input and output
X = df[['Age', 'Salary']]
y = df['Purchased']

# Split into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Decision Tree Classifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

dt_pred = dt.predict(X_test)

dt_accuracy = accuracy_score(y_test, dt_pred)

print("\nDecision Tree Results")
print("Predicted values:", dt_pred)
print("Actual values:", y_test.values)
print("Accuracy:", dt_accuracy)

# Random Forest Classifier
rf = RandomForestClassifier(n_estimators=10, random_state=42)
rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_pred)

print("\nRandom Forest Results")
print("Predicted values:", rf_pred)
print("Actual values:", y_test.values)
print("Accuracy:", rf_accuracy)

# Confusion Matrix
print("\nDecision Tree Confusion Matrix:")
print(confusion_matrix(y_test, dt_pred))

print("\nRandom Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_pred))

# Classification Report
print("\nDecision Tree Classification Report:")
print(classification_report(y_test, dt_pred, zero_division=0))

print("\nRandom Forest Classification Report:")
print(classification_report(y_test, rf_pred, zero_division=0))

# Compare both algorithms
print("\nFinal Comparison")

print("Decision Tree Accuracy:", dt_accuracy * 100, "%")
print("Random Forest Accuracy:", rf_accuracy * 100, "%")

if rf_accuracy > dt_accuracy:
    print("Random Forest performs better")
elif dt_accuracy > rf_accuracy:
    print("Decision Tree performs better")
else:
    print("Both algorithms have equal accuracy")