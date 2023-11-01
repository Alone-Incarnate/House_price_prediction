import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Load the dataset
data = pd.read_csv(r'House_Dataset.csv')

# Select important features and target variable
features = ['bedrooms', 'bathrooms', 'sqft_living', 'floors', 'condition', 'sqft_above', 'sqft_basement', 'yr_built', 'yr_renovated']
target = 'price'
data = data[features + [target]]

# Drop rows with missing values
data = data.dropna()

# Split the data into independent variables (X) and the target variable (y)
X = data[features]
y = data[target]

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and fit the Linear Regression model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions on the test set
y_test_predict = lr.predict(X_test)

# Print the coefficients of the model
print("Coefficients:", lr.coef_)

# Evaluate the model (you can add more evaluation metrics here)
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_test_predict)
r2 = r2_score(y_test, y_test_predict)
print("Mean Squared Error:", mse)
print("R-squared:", r2)