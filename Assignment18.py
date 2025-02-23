import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from textblob import TextBlob
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

# Load the dataset
file_path = '/mnt/data/Day_18_Tours_and_Travels.csv'
df = pd.read_csv(file_path)

# Display initial information
df.info()
print("\nMissing values in the dataset:")
print(df.isnull().sum())

# Handling Missing Values
num_imputer = SimpleImputer(strategy='median')
df['Customer_Age'] = num_imputer.fit_transform(df[['Customer_Age']])
df['Rating'].fillna(df['Rating'].median(), inplace=True)
df['Review_Text'].fillna("No Review", inplace=True)

# NLP-based handling for missing textual data
def correct_spelling(text):
    return str(TextBlob(text).correct())
df['Review_Text'] = df['Review_Text'].apply(correct_spelling)

# Removing Duplicates
df.drop_duplicates(inplace=True)

# Handling Inconsistent Data
df['Rating'] = df['Rating'].clip(1, 5)  # Ensure ratings are within range
df['Tour_Package'] = df['Tour_Package'].str.strip().str.title()  # Standardize tour names

# Detecting and Handling Outliers
plt.figure(figsize=(10,5))
sns.boxplot(x=df['Package_Price'])
plt.title('Boxplot of Package Price')
plt.show()

df['Package_Price'] = np.where(df['Package_Price'] > df['Package_Price'].quantile(0.95), df['Package_Price'].quantile(0.95), df['Package_Price'])

# Standardization and Normalization
scaler = MinMaxScaler()
df[['Package_Price', 'Customer_Age']] = scaler.fit_transform(df[['Package_Price', 'Customer_Age']])

# Convert categorical data into numerical format
encoder = LabelEncoder()
df['Tour_Package'] = encoder.fit_transform(df['Tour_Package'])

# Final Data Export
df.to_csv('/mnt/data/Cleaned_Travel_Reviews.csv', index=False)
print("Cleaned dataset saved successfully!")
