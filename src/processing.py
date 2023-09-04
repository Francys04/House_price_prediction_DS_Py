from sklearn.datasets import fetch_california_housing  # This line imports the fetch_california_housing function from
# scikit-learn's datasets module. It's used to load the California housing dataset.
import numpy as np  # This line imports the NumPy library, typically used for numerical and mathematical operations.
import pandas as pd  # This line imports the Pandas library, which is used for data manipulation and analysis.

# Load the Boston Housing dataset

housing = fetch_california_housing()

# Access the data, target, and feature names
data = housing.data
target = housing.target
feature_names = housing.feature_names

# Convert data and target to a DataFrame for display


boston_df = pd.DataFrame(data, columns=feature_names)
boston_df['target'] = target

# Display the DataFrame
print(boston_df.head())

'''load the datasets to the pandas dataframe'''
house_price_dataframe = pd.DataFrame(housing.data, columns=housing.feature_names)

'''print first 5 rows of our dataframe'''
print(house_price_dataframe.head())

'''add the target (price) column to the dataframe'''
house_price_dataframe['price'] = housing.target
print(house_price_dataframe.head())

print(house_price_dataframe.shape)

'''check for missing values'''
print(house_price_dataframe.isnull().sum())

'''statistical measures of the datasets'''
print(house_price_dataframe.describe())

'''Positive and negative correlation'''
correlation = house_price_dataframe.corr()

'''splitting the data and target'''
X = house_price_dataframe.drop(['price'], axis=1)
Y = house_price_dataframe['price']

print(X)
print(Y)

'''splitting the data intro training and test data'''
