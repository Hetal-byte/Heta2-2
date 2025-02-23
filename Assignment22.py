import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = '/mnt/data/Civil_Engineering_Regression_Dataset.csv'
df = pd.read_csv(file_path)

# Display initial info
df.info()
print("\nMissing values in the dataset:")
print(df.isna().sum())

# Generate summary statistics
print("\nSummary statistics:")
print(df.describe())

# Correlation heatmap
plt.figure(figsize=(10,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Simple Linear Regression
X = df[['Building Height']]
y = df['Construction Cost']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get regression equation
intercept = model.intercept_
coefficient = model.coef_[0]
print(f"Regression Equation: Construction Cost = {intercept:.2f} + {coefficient:.2f} * Building Height")

# Make predictions
y_pred = model.predict(X_test)

# Evaluate model performance
r_squared = r2_score(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
print(f"R-squared: {r_squared:.4f}")
print(f"Mean Squared Error: {mse:.2f}")

# Scatter plot with regression line
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual Data')
plt.plot(X_test, y_pred, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Building Height")
plt.ylabel("Construction Cost")
plt.title("Simple Linear Regression: Building Height vs. Construction Cost")
plt.legend()
plt.show()
