# simple_linear_regression.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Load the dataset
data = pd.read_csv(r"../data/delivery_time.csv")

# Display basic statistics
data_stats = data.describe()
print(data_stats)

# Check for missing values
data_info = data.info()

# Plot histograms of the features
data.hist(bins=30, figsize=(10, 5))
plt.show()

# Scatter plot to visualize the relationship between sorting time and delivery time
plt.scatter(data['Sorting Time'], data['Delivery Time'])
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Sorting Time vs Delivery Time')
plt.show()

# Correlation matrix
sns.heatmap(data.corr(), annot=True, cmap="YlGnBu")
plt.show()

# Rename columns to avoid spaces
data.columns = ['Delivery_Time', 'Sorting_Time']

# Simple Linear Regression
model = smf.ols('Delivery_Time ~ Sorting_Time', data=data).fit()
model_summary = model.summary()
print(model_summary)

# Predictions
pred1 = model.predict(data['Sorting_Time'])

# Plotting the regression line
plt.scatter(data['Sorting_Time'], data['Delivery_Time'])
plt.plot(data['Sorting_Time'], pred1, "r")
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Simple Linear Regression')
plt.show()

# Calculate RMSE
res1 = data['Delivery_Time'] - pred1
mse1 = (res1**2).mean()
rmse1 = mse1**0.5
print(f"RMSE: {rmse1}")

# Log Transformation
model_log = smf.ols('Delivery_Time ~ np.log(Sorting_Time)', data=data).fit()
model_log_summary = model_log.summary()
print(model_log_summary)

# Predictions
pred_log = model_log.predict(data['Sorting_Time'])

# Plotting the regression line
plt.scatter(np.log(data['Sorting_Time']), data['Delivery_Time'])
plt.plot(np.log(data['Sorting_Time']), pred_log, "r")
plt.xlabel('Log of Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Log Transformation')
plt.show()

# Calculate RMSE
res_log = data['Delivery_Time'] - pred_log
mse_log = (res_log**2).mean()
rmse_log = mse_log**0.5
print(f"RMSE (Log Transformation): {rmse_log}")

# Exponential Transformation
model_exp = smf.ols('np.log(Delivery_Time) ~ Sorting_Time', data=data).fit()
model_exp_summary = model_exp.summary()
print(model_exp_summary)

# Predictions
pred_exp = model_exp.predict(data['Sorting_Time'])
pred_exp = np.exp(pred_exp)

# Plotting the regression line
plt.scatter(data['Sorting_Time'], np.log(data['Delivery_Time']))
plt.plot(data['Sorting_Time'], np.log(pred_exp), "r")
plt.xlabel('Sorting Time')
plt.ylabel('Log of Delivery Time')
plt.title('Exponential Transformation')
plt.show()

# Calculate RMSE
res_exp = data['Delivery_Time'] - pred_exp
mse_exp = (res_exp**2).mean()
rmse_exp = mse_exp**0.5
print(f"RMSE (Exponential Transformation): {rmse_exp}")

# Polynomial Transformation
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(data[['Sorting_Time']])

poly_model = LinearRegression()
poly_model.fit(X_poly, data['Delivery_Time'])
pred_poly = poly_model.predict(X_poly)

# Plotting the regression line
plt.scatter(data['Sorting_Time'], data['Delivery_Time'])
plt.plot(data['Sorting_Time'], pred_poly, "r")
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.title('Polynomial Transformation')
plt.show()

# Calculate RMSE
res_poly = data['Delivery_Time'] - pred_poly
mse_poly = (res_poly**2).mean()
rmse_poly = mse_poly**0.5
print(f"RMSE (Polynomial Transformation): {rmse_poly}")
