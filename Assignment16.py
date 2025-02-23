import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/Day_14_Pharma_data.csv'
df = pd.read_csv(file_path)

# Data Cleaning
print("Missing values in the dataset:")
print(df.isnull().sum())
df.drop_duplicates(inplace=True)

# Bar plot comparing average Effectiveness for each drug across different regions
plt.figure(figsize=(10,5))
sns.barplot(x='Region', y='Effectiveness', hue='Product', data=df)
plt.title('Average Drug Effectiveness by Region')
plt.xlabel('Region')
plt.ylabel('Effectiveness')
plt.legend(title='Product')
plt.show()

# Violin plot showing distribution of Effectiveness and Side_Effects for each product
plt.figure(figsize=(10,5))
sns.violinplot(x='Product', y='Effectiveness', data=df, inner='quartile')
plt.title('Distribution of Drug Effectiveness by Product')
plt.xlabel('Product')
plt.ylabel('Effectiveness')
plt.show()

plt.figure(figsize=(10,5))
sns.violinplot(x='Product', y='Side_Effects', data=df, inner='quartile')
plt.title('Distribution of Side Effects by Product')
plt.xlabel('Product')
plt.ylabel('Side Effects')
plt.show()

# Pairplot to explore relationships between Effectiveness, Side_Effects, and Marketing_Spend
sns.pairplot(df[['Effectiveness', 'Side_Effects', 'Marketing_Spend']])
plt.show()

# Boxplot comparing Effectiveness for different trial periods
plt.figure(figsize=(10,5))
sns.boxplot(x='Trial_Period', y='Effectiveness', data=df)
plt.title('Effectiveness by Trial Period')
plt.xlabel('Trial Period')
plt.ylabel('Effectiveness')
plt.show()

# Regression plot to analyze how Marketing_Spend affects Effectiveness
plt.figure(figsize=(8,5))
sns.regplot(x='Marketing_Spend', y='Effectiveness', data=df)
plt.title('Marketing Spend vs Drug Effectiveness')
plt.xlabel('Marketing Spend')
plt.ylabel('Effectiveness')
plt.show()
