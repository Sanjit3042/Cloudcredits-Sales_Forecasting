# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

# Load the dataset
data = pd.read_csv(r'C:\Users\user\Downloads\CLOUDCREDITS\Sales_forecasting\data\retail_sales_dataset.csv')

# Convert the 'Date' column to datetime and set it as the index
data['Date'] = pd.to_datetime(data['Date'], errors='coerce')  # Convert Date to datetime
data.set_index('Date', inplace=True)

# Check for missing dates and forward fill them
data.ffill(inplace=True)

# Use 'Total Amount' as the sales data and ensure it's numeric
data['Total Amount'] = pd.to_numeric(data['Total Amount'], errors='coerce')  # Coerce any non-numeric values

# Visualize the 'Total Amount' as sales
plt.figure(figsize=(10,6))
plt.plot(data.index, data['Total Amount'], label='Sales')  # Using 'Total Amount' as the sales column
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales (Total Amount)')
plt.legend()
plt.show()

# Train-test split
train_size = int(len(data) * 0.8)
train, test = data['Total Amount'][:train_size], data['Total Amount'][train_size:]

# Ensure no missing values in the training data
train.ffill(inplace=True)  # Forward fill missing values in the training data if any
test.ffill(inplace=True)   # Forward fill missing values in the testing data if any

# -----------------------------
# ARIMA Model
# -----------------------------
# Fit ARIMA model
arima_model = ARIMA(train, order=(5,1,0))
arima_model_fit = arima_model.fit()

# Forecast using ARIMA
arima_forecast = arima_model_fit.forecast(steps=len(test))

# Evaluate ARIMA model
arima_rmse = np.sqrt(mean_squared_error(test, arima_forecast))
print(f'ARIMA RMSE: {arima_rmse}')

# -----------------------------
# Visualization of the forecasts
# -----------------------------
plt.figure(figsize=(12,8))

# Actual sales
plt.plot(test.index, test, label='Actual Sales', color='black')

# ARIMA forecast
plt.plot(test.index, arima_forecast, label='ARIMA Forecast', color='blue')

plt.title('ARIMA Sales Forecast vs Actual Sales')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.legend()
plt.show()
