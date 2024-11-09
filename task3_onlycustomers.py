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

# Create seasonal dummy variables
df['month'] = df.index.month
seasonal_dummies = pd.get_dummies(df['month'], prefix='month', drop_first=True)

# Prepare the features for linear regression including seasonal dummies
X = pd.concat([df[['Customers', 'HDD', 'CDD']], seasonal_dummies], axis=1)
y = trend.values
model = LinearRegression().fit(X, y)

# Load the future values for the linear variables
future_df = pd.read_excel(r'C:\Users\U348469\Desktop\dli notifications .xlsx', header=0, sheet_name='FutureData')
future_df['Date'] = pd.to_datetime(future_df['Date'])
future_df.set_index('Date', inplace=True)

# Create seasonal dummy variables for future data
future_df['month'] = future_df.index.month
future_seasonal_dummies = pd.get_dummies(future_df['month'], prefix='month', drop_first=True)

# Prepare the future features for linear regression including seasonal dummies
future_features = pd.concat([future_df[['Customers', 'HDD', 'CDD']], future_seasonal_dummies], axis=1)

# Predict future trend values based on future features
trend_forecast = model.predict(future_features)

# Repeat the seasonal pattern for the forecast period
seasonal_pattern = seasonal[-12:]  # Last year's seasonal pattern
seasonal_forecast = np.tile(seasonal_pattern, len(future_features) // 12 + 1)[:len(future_features)]

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

# Rationale
# Seasonal Dummy Variables: By including dummy variables for each month, we allow the model to account for the seasonal effects that repeat every year. This helps in capturing the impact of HDD and CDD more accurately.
# Combining Features: Including ‘Customers’, ‘HDD’, ‘CDD’, and seasonal dummies in the regression model helps in understanding how these variables collectively influence the trend component of ‘Billed MWh’.
# STL Decomposition: Separating the seasonal and trend components allows us to model the underlying trend separately from the seasonal effects, leading to more accurate forecasts.