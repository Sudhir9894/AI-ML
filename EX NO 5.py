import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Generate random data
np.random.seed(42)  # For reproducibility
n_samples = 100

# Random data for two features and one target variable
feature1 = np.random.rand(n_samples) * 100  # Random data between 0 and 100
feature2 = np.random.rand(n_samples) * 100
target = 5 + 2 * feature1 + 3 * feature2 + np.random.randn(n_samples) * 10  # Linear relationship with noise

# Create a DataFrame
df = pd.DataFrame({
    'feature1': feature1,
    'feature2': feature2,
    'target': target
})

# Split the dataset into training and testing sets
X = df[['feature1', 'feature2']]  # Feature columns
y = df['target']  # Target column
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the regression model
reg = LinearRegression()
reg.fit(X_train, y_train)

# Make predictions on the test set
y_pred = reg.predict(X_test)

# Evaluate the model
print('Mean squared error: %.2f' % mean_squared_error(y_test, y_pred))
print('Coefficient of determination: %.2f' % r2_score(y_test, y_pred))

# Plot the results
plt.scatter(X_test['feature1'], y_test, color='black')  # Plot actual values
plt.plot(X_test['feature1'], y_pred, color='blue', linewidth=3)  # Plot predicted values

plt.xticks([])  # Optionally remove x-axis ticks
plt.yticks([])  # Optionally remove y-axis ticks
plt.show()
