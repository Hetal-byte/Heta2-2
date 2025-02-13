# Importing necessary libraries
import pandas as pd

# Loading the CSV file into a Pandas DataFrame
file_path = '/mnt/data/Day_7_sales_data.csv'
sales_data = pd.read_csv('C:/Users/Win 10/Downloads/Day_7_sales_data.csv')

# Display the first 5 rows of the dataset
print("First 5 rows of the dataset:")
print(sales_data.head())

# Display basic statistics of the numerical columns
print("\nBasic statistics of numerical columns:")
print(sales_data.describe())

# Calculate total sales for each region
total_sales_by_region = sales_data.groupby('Region')['Sales'].sum()
print("\nTotal sales for each region:")
print(total_sales_by_region)

# Find the most sold product based on quantity
most_sold_product = sales_data.groupby('Product')['Quantity'].sum().idxmax()
print(f"\nMost sold product: {most_sold_product}")

# Compute the average profit margin for each product
sales_data['Profit Margin (%)'] = (sales_data['Profit'] / sales_data['Sales']) * 100
average_profit_margin_by_product = sales_data.groupby('Product')['Profit Margin (%)'].mean()
print("\nAverage profit margin for each product:")
print(average_profit_margin_by_product)
