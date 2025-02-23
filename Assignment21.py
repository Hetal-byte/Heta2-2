import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/Civil_Engineering_Regression_Dataset.csv'
df = pd.read_csv(file_path)

# Display the first few rows of the dataset
print("First 5 rows of the dataset:")
print(df.head())

# Identify independent and dependent variables
# Assuming 'Construction Cost' is the dependent variable
dependent_var = 'Construction Cost'
independent_vars = [col for col in df.columns if col != dependent_var]
print("\nDependent Variable:", dependent_var)
print("Independent Variables:", independent_vars)

# Check for missing values
print("\nMissing Values:")
print(df.isna().sum())

# Handle missing values (Example: Filling numerical values with median)
df.fillna(df.median(numeric_only=True), inplace=True)

# Generate summary statistics for numerical columns
print("\nSummary Statistics:")
print(df.describe())

# Create a correlation heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title("Correlation Heatmap")
plt.show()

# Interpretation: Identify strong correlations
correlation_matrix = df.corr()
construction_cost_corr = correlation_matrix[dependent_var].sort_values(ascending=False)
print("\nCorrelation of factors with Construction Cost:")
print(construction_cost_corr)
