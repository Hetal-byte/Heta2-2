import pandas as pd

# Creating the DataFrame
data = {
    "Name": ["John", "Alice", "Bob", "Diana"],
    "Age": [28, 34, 23, 29],
    "Department": ["HR", "IT", "Marketing", "Finance"],
    "Salary": [45000, 60000, 35000, 50000]
}

df = pd.DataFrame(data)

# Display the first 2 rows
print("First 2 rows of the DataFrame:")
print(df.head(2))

# Add a new column 'Bonus' (10% of Salary)
df["Bonus"] = df["Salary"] * 0.10

# Calculate the average salary
average_salary = df["Salary"].mean()
print("\nAverage Salary:", average_salary)

# Filter and display employees older than 25
older_than_25 = df[df["Age"] > 25]
print("\nEmployees older than 25:")
print(older_than_25)
