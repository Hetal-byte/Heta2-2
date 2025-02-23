import pandas as pd

# Load the dataset
file_path = '/mnt/data/Day_11_banking_data.csv'
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

# Filtering Data Based on Conditions
filtered_df1 = df[df['Transaction_Amount'] <= 2000]
print("\nTransactions with Transaction_Amount <= 2000:")
print(filtered_df1)

filtered_df2 = df[(df['Transaction_Type'] == 'Loan Payment') & (df['Account_Balance'] > 5000)]
print("\nTransactions where Transaction_Type is 'Loan Payment' and Account_Balance > 5000:")
print(filtered_df2)

filtered_df3 = df[df['Branch'] == 'Uptown']
print("\nTransactions made in the 'Uptown' branch:")
print(filtered_df3)

# Data Transformation
# Add a new column Transaction_Fee (2% of Transaction_Amount)
df['Transaction_Fee'] = df['Transaction_Amount'] * 0.02

# Create a new column Balance_Status based on Account_Balance
df['Balance_Status'] = df['Account_Balance'].apply(lambda x: 'High Balance' if x > 5000 else 'Low Balance')

# Sorting and Ranking Data
# Sort dataset by Account_Balance in descending order
sorted_df = df.sort_values(by='Account_Balance', ascending=False)
print("\nTop 10 rows sorted by Account_Balance:")
print(sorted_df.head(10))

# Rank Transaction_Amount within each Branch
df['Transaction_Rank'] = df.groupby('Branch')['Transaction_Amount'].rank(method='dense', ascending=False)

print("\nDataset after sorting and ranking:")
print(df.head())
