import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm
from scipy import stats

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
X_simple = df[['Building Height']]
y = df['Construction Cost']

# Split data into training and testing sets
X_train_simple, X_test_simple, y_train, y_test = train_test_split(X_simple, y, test_size=0.2, random_state=42)

# Train the model
model_simple = LinearRegression()
model_simple.fit(X_train_simple, y_train)

# Get regression equation
intercept_simple = model_simple.intercept_
coefficient_simple = model_simple.coef_[0]
print(f"Simple Regression Equation: Construction Cost = {intercept_simple:.2f} + {coefficient_simple:.2f} * Building Height")

# Make predictions
y_pred_simple = model_simple.predict(X_test_simple)

# Evaluate model performance
r_squared_simple = r2_score(y_test, y_pred_simple)
mse_simple = mean_squared_error(y_test, y_pred_simple)
print(f"Simple R-squared: {r_squared_simple:.4f}")
print(f"Simple Mean Squared Error: {mse_simple:.2f}")

# Scatter plot with regression line
plt.figure(figsize=(8,6))
plt.scatter(X_test_simple, y_test, color='blue', label='Actual Data')
plt.plot(X_test_simple, y_pred_simple, color='red', linewidth=2, label='Regression Line')
plt.xlabel("Building Height")
plt.ylabel("Construction Cost")
plt.title("Simple Linear Regression: Building Height vs. Construction Cost")
plt.legend()
plt.show()

# Multiple Linear Regression
X_multiple = df[['Building Height', 'Material Quality', 'Labor Cost', 'Concrete Strength', 'Foundation Depth']]

# Split data into training and testing sets
X_train_multiple, X_test_multiple, y_train, y_test = train_test_split(X_multiple, y, test_size=0.2, random_state=42)

# Train the model
model_multiple = LinearRegression()
model_multiple.fit(X_train_multiple, y_train)

# Get regression coefficients
intercept_multiple = model_multiple.intercept_
coefficients_multiple = model_multiple.coef_
print("\nMultiple Regression Equation: Construction Cost = ")
print(f"{intercept_multiple:.2f} + ")
for feature, coef in zip(X_multiple.columns, coefficients_multiple):
    print(f"{coef:.2f} * {feature} + ")

# Identify most impactful factor
impactful_feature = X_multiple.columns[np.argmax(np.abs(coefficients_multiple))]
print(f"\nMost impactful factor on Construction Cost: {impactful_feature}")

# Make predictions
y_pred_multiple = model_multiple.predict(X_test_multiple)

# Evaluate model performance
r_squared_multiple = r2_score(y_test, y_pred_multiple)
mse_multiple = mean_squared_error(y_test, y_pred_multiple)
print(f"Multiple R-squared: {r_squared_multiple:.4f}")
print(f"Multiple Mean Squared Error: {mse_multiple:.2f}")

# Adjusted R-squared calculation
n = X_test_multiple.shape[0]  # Number of observations
p = X_test_multiple.shape[1]  # Number of predictors
adj_r_squared = 1 - (1 - r_squared_multiple) * (n - 1) / (n - p - 1)
print(f"Adjusted R-squared: {adj_r_squared:.4f}")

# Variance Inflation Factor (VIF) for multicollinearity detection
X_with_const = sm.add_constant(X_multiple)
vif_data = pd.DataFrame()
vif_data['Feature'] = X_with_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_with_const.values, i) for i in range(X_with_const.shape[1])]
print("\nVariance Inflation Factor (VIF):")
print(vif_data)

# Feature Selection using Lasso Regression
lasso = Lasso(alpha=0.1)
lasso.fit(X_train_multiple, y_train)
print("\nLasso Regression Coefficients:")
for feature, coef in zip(X_multiple.columns, lasso.coef_):
    print(f"{feature}: {coef:.4f}")

# Residual Analysis
residuals = y_test - y_pred_multiple
plt.figure(figsize=(8,6))
sns.histplot(residuals, bins=20, kde=True)
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.title("Residual Distribution")
plt.show()

# Outlier Detection using Z-score
z_scores = np.abs(stats.zscore(df[['Construction Cost']]))
outliers = df[z_scores > 3]
print("\nOutliers detected:")
print(outliers)

# Business Impact Considerations
print("\nEthical Considerations & Business Impact:")
print("Errors in cost prediction can lead to financial losses and safety risks. Underestimation may cause budget overruns, while overestimation may deter investors.")
print("Future enhancements: Incorporating real-time material costs, weather impact, and advanced machine learning models.")
