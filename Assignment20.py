import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import KNNImputer
from textblob import TextBlob

# Load the dataset
file_path = '/mnt/data/Day 20_E-Commerce_Data.csv'
df = pd.read_csv(file_path)

# Display initial info
df.info()
print("\nMissing values in the dataset:")
print(df.isna().sum())

# Handling missing values
# Fill missing numerical values with median
df['Customer_Age'].fillna(df['Customer_Age'].median(), inplace=True)
df['Rating'].fillna(df['Rating'].median(), inplace=True)

# Handle missing textual data using NLP (fill with sentiment-based placeholder)
def handle_missing_reviews(text):
    return "No Review" if pd.isna(text) else text
df['Review_Text'] = df['Review_Text'].apply(handle_missing_reviews)

# Detect and remove duplicates
df.drop_duplicates(inplace=True)

# Standardizing Rating values (ensuring range 1-5)
df['Rating'] = df['Rating'].clip(1, 5)

# Correct spelling inconsistencies in Product_Category using TextBlob
def correct_spelling(text):
    return str(TextBlob(text).correct())
df['Product_Category'] = df['Product_Category'].apply(correct_spelling)

# Identifying and handling outliers
plt.figure(figsize=(10,5))
sns.boxplot(x=df['Product_Price'])
plt.title('Boxplot of Product Price Before Outlier Treatment')
plt.show()

# Cap extreme outliers
q1, q3 = np.percentile(df['Product_Price'].dropna(), [25, 75])
iqr = q3 - q1
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr
df['Product_Price'] = np.where(df['Product_Price'] > upper_bound, upper_bound, df['Product_Price'])
df['Product_Price'] = np.where(df['Product_Price'] < lower_bound, lower_bound, df['Product_Price'])

# Convert categorical data into numerical format (Label Encoding)
df['Product_Category'] = df['Product_Category'].astype('category').cat.codes

# Save cleaned dataset
df.to_csv('/mnt/data/Cleaned_E-Commerce_Reviews.csv', index=False)
print("Cleaned dataset saved successfully!")
