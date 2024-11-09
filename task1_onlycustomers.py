import pandas as pd
import numpy as np
from statsmodels.tsa.seasonal import STL
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Load the historical data
df = pd.read_excel(r'C:\Users\U348469\Desktop\dli notifications .xlsx', header=0, sheet_name='Data')
df = df.dropna(subset=['Billed MWh'])

# Convert 'Date' column to datetime and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Decompose the seasonal variable
stl = STL(df['Billed MWh'], seasonal=13)
result = stl.fit()
seasonal = result.seasonal
trend = result.trend

# Linear regression on the trend component
X = df['Customers'].values.reshape(-1, 1)
y = trend.values
model = LinearRegression().fit(X, y)

# Load the future values for the linear variable
future_df = pd.read_excel(r'C:\Users\U348469\Desktop\dli notifications .xlsx', header=0, sheet_name='FutureData')
future_df['Date'] = pd.to_datetime(future_df['Date'])
future_df.set_index('Date', inplace=True)
future_customers = future_df['Customers'].values.reshape(-1, 1)

# Predict future trend values based on future customers
trend_forecast = model.predict(future_customers)

# Repeat the seasonal pattern for the forecast period
seasonal_pattern = seasonal[-12:]  # Last year's seasonal pattern
seasonal_forecast = np.tile(seasonal_pattern, len(future_customers) // 12 + 1)[:len(future_customers)]

# Combine the seasonal and trend forecasts
forecast = seasonal_forecast[:len(trend_forecast)] + trend_forecast

# Create a DataFrame for the forecasted values
forecast_df = pd.DataFrame({'Date': future_df.index, 'Forecasted Billed MWh': forecast})
forecast_df.set_index('Date', inplace=True)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['Billed MWh'], label='Original')
plt.plot(forecast_df.index, forecast_df['Forecasted Billed MWh'], label='Forecast')
plt.legend()
plt.show()

forecast_df.to_excel(r'C:\Users\U348469\Desktop\answer.xlsx')