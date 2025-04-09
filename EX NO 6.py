import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Generate random data
np.random.seed(42)  # For reproducibility
n_samples = 100
n_features = 5  # Number of features

# Random data for features and target variable
X = np.random.rand(n_samples, n_features) * 100  # Random data for features

# Convert target to a classification task (categorical target)
y = (X[:, 0] + X[:, 1] + np.random.randn(n_samples) * 10 > 100).astype(int)  # Convert to binary target

# Create a DataFrame
data = pd.DataFrame(X, columns=[f'feature{i+1}' for i in range(n_features)])
data['target'] = y

# Split data into training and test sets
X = data.drop(['target'], axis=1)  # Feature columns
y = data['target']  # Target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build decision tree classifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# Predict on test set
y_pred_dt = dt.predict(X_test)

# Evaluate performance for decision tree classifier
accuracy_dt = accuracy_score(y_test, y_pred_dt)
print(f"Decision Tree Classifier Accuracy: {accuracy_dt:.4f}")

# Build random forest classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf.predict(X_test)

# Evaluate performance for random forest classifier
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Random Forest Classifier Accuracy: {accuracy_rf:.4f}")
