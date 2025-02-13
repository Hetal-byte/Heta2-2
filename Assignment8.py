import pandas as pd

# Load the CSV file
df = pd.read_csv("Day_8_sales_data.csv")

# Data Filtering
# Extract all rows where sales are greater than 1000
high_sales_df = df[df["Sales"] > 1000]

# Find all sales records for a specific region (e.g., "East")
east_sales_df = df[df["Region"] == "East"]

# Data Processing
# Add a new column, Profit_Per_Unit, calculated as Profit / Quantity
df["Profit_Per_Unit"] = df["Profit"] / df["Quantity"]

# Create another column, High_Sales, which labels rows as Yes if Sales > 1000, else No
df["High_Sales"] = df["Sales"].apply(lambda x: "Yes" if x > 1000 else "No")

# Display results
print(df.head())
