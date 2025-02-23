import pandas as pd

# Load the dataset
file_path = '/mnt/data/Day_9_banking_data.csv'
df = pd.read_csv(file_path)

# Display the first 5 rows
print("First 5 rows of the dataset:")
print(df.head())

# Generate basic statistics of numerical columns
print("\nBasic statistics of numerical columns:")
print(df.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Group by Account_Type and calculate required statistics
account_group = df.groupby('Account_Type')
print("\nTotal sum of Transaction_Amount per Account_Type:")
print(account_group['Transaction_Amount'].sum())

print("\nAverage Account_Balance per Account_Type:")
print(account_group['Account_Balance'].mean())

# Group by Branch and calculate required statistics
branch_group = df.groupby('Branch')
print("\nTotal number of transactions per Branch:")
print(branch_group.size())

print("\nAverage Transaction_Amount per Branch:")
print(branch_group['Transaction_Amount'].mean())
