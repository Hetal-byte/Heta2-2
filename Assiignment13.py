import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
file_path = '/mnt/data/Day_13_Pharma_data.csv'
df = pd.read_csv(file_path)

# Data Cleaning
print("Missing values in the dataset:")
print(df.isnull().sum())

df.drop_duplicates(inplace=True)

# Bar plot for total sales per region
plt.figure(figsize=(10,5))
df.groupby('Region')['Sales'].sum().plot(kind='bar', color='skyblue')
plt.title('Total Sales per Region')
plt.xlabel('Region')
plt.ylabel('Total Sales')
plt.show()

# Scatter plot for Marketing_Spend vs Sales
plt.figure(figsize=(8,5))
sns.scatterplot(x=df['Marketing_Spend'], y=df['Sales'])
plt.title('Marketing Spend vs Sales')
plt.xlabel('Marketing Spend')
plt.ylabel('Sales')
plt.show()

# Boxplot for Drug Effectiveness across Age Groups
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Age_Group'], y=df['Effectiveness'])
plt.title('Drug Effectiveness across Age Groups')
plt.xlabel('Age Group')
plt.ylabel('Effectiveness')
plt.show()

# Line plot for sales trend per product over different trial periods
plt.figure(figsize=(10,5))
sns.lineplot(x=df['Trial_Period'], y=df['Sales'], hue=df['Product'])
plt.title('Sales Trend per Product')
plt.xlabel('Trial Period')
plt.ylabel('Sales')
plt.legend(title='Product')
plt.show()

# Heatmap of correlation between Sales, Marketing_Spend, and Effectiveness
plt.figure(figsize=(6,5))
sns.heatmap(df[['Sales', 'Marketing_Spend', 'Effectiveness']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()
