import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = '/mnt/data/Day 19_E-Commerce_Data.csv'
df = pd.read_csv(file_path)

# Display initial info
df.info()
print("\nMissing values in the dataset:")
print(df.isna().sum())

# Compute percentage of missing values
missing_percentage = (df.isna().sum() / len(df)) * 100
print("\nPercentage of missing values per column:")
print(missing_percentage)

# Visualizing missing data
plt.figure(figsize=(10,6))
sns.heatmap(df.isna(), cmap='viridis', cbar=False, yticklabels=False)
plt.title("Missing Data Heatmap")
plt.show()

# Handling missing values
# Impute numerical columns with median
num_cols = ['Product_Price']
for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Impute categorical columns with mode
cat_cols = ['Product_Category']
for col in cat_cols:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Forward fill for date-related fields
date_cols = ['Order_Date']
for col in date_cols:
    df[col].fillna(method='ffill', inplace=True)

# KNN Imputation for complex cases
knn_imputer = KNNImputer(n_neighbors=5)
df[['Product_Price']] = knn_imputer.fit_transform(df[['Product_Price']])

# Validate if missing values are handled
print("\nMissing values after imputation:")
print(df.isna().sum())

# Visualizing data after imputation
plt.figure(figsize=(10,5))
sns.boxplot(x=df['Product_Price'])
plt.title('Boxplot of Product Price After Imputation')
plt.show()

# Save cleaned dataset
df.to_csv('/mnt/data/Cleaned_E-Commerce_Data.csv', index=False)
print("Cleaned dataset saved successfully!")
