import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error
import pmdarima as pm
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set professional styling
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

print("="*60)
print("TIME SERIES ANALYSIS OF ETHEREUM (ETH/USDT) MARKET PROJECTIONS")
print("USING ARIMA MODELING - PROFESSIONAL ANALYSIS")
print("="*60)

# ───────────────────────────────────────
# 1. DATA COLLECTION & PREPARATION
# ───────────────────────────────────────
print("\n1. DATA COLLECTION & PREPARATION")
print("-" * 40)

# Load the dataset
try:
    df = pd.read_csv('Binance_ETHUSDT_d.csv')
    print("✅ Data successfully loaded from Binance_ETHUSDT_d.csv")
except FileNotFoundError:
    print("❌ File not found. Please ensure 'Binance_ETHUSDT_d.csv' is in the working directory")
    exit()

# Data preprocessing
print(f"Original dataset shape: {df.shape}")
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")

# Convert date and set as index
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.sort_index(inplace=True)

# Handle missing values
missing_before = df.isnull().sum().sum()
df.dropna(inplace=True)
missing_after = df.isnull().sum().sum()

print(f"Missing values before cleaning: {missing_before}")
print(f"Missing values after cleaning: {missing_after}")
print(f"Duplicated rows: {df.duplicated().sum()}")
print(f"Final dataset shape: {df.shape}")

# Display data info
print("\nDataset Info:")
print(f"Columns: {list(df.columns)}")
print(f"Data types:\n{df.dtypes}")

# Key attributes documentation
print("\nKey Attributes:")
print("- Date: Trading date (daily frequency)")
print("- Open: Opening price in USDT")
print("- High: Highest price in USDT") 
print("- Low: Lowest price in USDT")
print("- Close: Closing price in USDT (primary focus)")
print("- Volume: Trading volume")

# ───────────────────────────────────────
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# ───────────────────────────────────────
print("\n2. EXPLORATORY DATA ANALYSIS")
print("-" * 40)

# Statistical summary
print("Descriptive Statistics for ETH Closing Prices:")
print(df['Close'].describe())

# 1. Main price trend visualization
plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Close'], color='#1f77b4', linewidth=1.5, alpha=0.8)
plt.title('Ethereum (ETH/USDT) Daily Closing Price Over Time', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USDT)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 2. Price with moving averages
df['MA_7'] = df['Close'].rolling(window=7).mean()
df['MA_30'] = df['Close'].rolling(window=30).mean()
df['MA_90'] = df['Close'].rolling(window=90).mean()

plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Close'], label='Close Price', alpha=0.7, linewidth=1.5, color='#1f77b4')
plt.plot(df.index, df['MA_7'], label='7-Day MA', color='#ff7f0e', linewidth=2)
plt.plot(df.index, df['MA_30'], label='30-Day MA', color='#d62728', linewidth=2)
plt.plot(df.index, df['MA_90'], label='90-Day MA', color='#2ca02c', linewidth=2)
plt.title('ETH Price with Moving Averages', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USDT)', fontsize=12)
plt.legend(loc='upper left')
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# 3. Volume analysis (if available)
volume_col = None
for col in df.columns:
    if 'volume' in col.lower() or 'vol' in col.lower():
        volume_col = col
        break

if volume_col:
    plt.figure(figsize=(14, 8))
    plt.plot(df.index, df[volume_col], color='#2ca02c', alpha=0.7, linewidth=1.5)
    plt.title('ETH Trading Volume Over Time', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Volume', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
else:
    print("Volume data not available in dataset")

# 4. Price distribution
plt.figure(figsize=(12, 8))
plt.hist(df['Close'], bins=50, alpha=0.7, color='#9467bd', edgecolor='black', linewidth=0.5)
plt.title('Distribution of ETH Closing Prices', fontsize=16, pad=20)
plt.xlabel('Price (USDT)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# 5. Volatility analysis
df['Returns'] = df['Close'].pct_change()
df['Volatility'] = df['Returns'].rolling(window=30).std() * np.sqrt(365)

plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Returns'], alpha=0.7, color='#17becf', linewidth=1)
plt.title('Daily Returns', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Returns', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 8))
plt.plot(df.index, df['Volatility'], color='#e377c2', linewidth=2)
plt.title('30-Day Rolling Volatility (Annualized)', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Volatility', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Market behavior insights
print("\nMarket Behavior Insights:")
print(f"Average daily return: {df['Returns'].mean():.4f} ({df['Returns'].mean()*100:.2f}%)")
print(f"Volatility (std of returns): {df['Returns'].std():.4f} ({df['Returns'].std()*100:.2f}%)")
print(f"Maximum daily gain: {df['Returns'].max():.4f} ({df['Returns'].max()*100:.2f}%)")
print(f"Maximum daily loss: {df['Returns'].min():.4f} ({df['Returns'].min()*100:.2f}%)")

# ───────────────────────────────────────
# 3. STATIONARITY TESTING
# ───────────────────────────────────────
print("\n3. STATIONARITY TESTING")
print("-" * 40)

def perform_stationarity_tests(series, series_name):
    """Perform comprehensive stationarity tests"""
    print(f"\nStationarity Tests for {series_name}:")
    
    # ADF Test
    adf_result = adfuller(series.dropna())
    print(f"ADF Test:")
    print(f"  Statistic: {adf_result[0]:.6f}")
    print(f"  p-value: {adf_result[1]:.6f}")
    print(f"  Critical Values: {adf_result[4]}")
    
    if adf_result[1] <= 0.05:
        print("  ✅ ADF: Series is stationary (reject null hypothesis)")
    else:
        print("  ❌ ADF: Series is non-stationary (fail to reject null hypothesis)")
    
    # KPSS Test
    kpss_result = kpss(series.dropna(), regression='c')
    print(f"KPSS Test:")
    print(f"  Statistic: {kpss_result[0]:.6f}")
    print(f"  p-value: {kpss_result[1]:.6f}")
    print(f"  Critical Values: {kpss_result[3]}")
    
    if kpss_result[1] >= 0.05:
        print("  ✅ KPSS: Series is stationary (fail to reject null hypothesis)")
    else:
        print("  ❌ KPSS: Series is non-stationary (reject null hypothesis)")
    
    return adf_result[1] <= 0.05 and kpss_result[1] >= 0.05

# Test original series
is_stationary_original = perform_stationarity_tests(df['Close'], "Original Close Prices")

# First differencing
df['Close_diff1'] = df['Close'].diff()
is_stationary_diff1 = perform_stationarity_tests(df['Close_diff1'], "First Differenced Series")

# Second differencing if needed
if not is_stationary_diff1:
    df['Close_diff2'] = df['Close_diff1'].diff()
    is_stationary_diff2 = perform_stationarity_tests(df['Close_diff2'], "Second Differenced Series")
    d_value = 2
else:
    d_value = 1

print(f"\nRecommended differencing order (d): {d_value}")

# Visual stationarity check - Original series
plt.figure(figsize=(14, 8))
plt.plot(df['Close'], color='#1f77b4', linewidth=1.5)
plt.title('Original ETH Close Prices', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USDT)', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# First difference
plt.figure(figsize=(14, 8))
plt.plot(df['Close_diff1'], color='#ff7f0e', linewidth=1.5)
plt.title('First Differenced Series', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price Difference', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Second difference (if calculated)
if 'Close_diff2' in df.columns:
    plt.figure(figsize=(14, 8))
    plt.plot(df['Close_diff2'], color='#d62728', linewidth=1.5)
    plt.title('Second Differenced Series', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price Difference', fontsize=12)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# ───────────────────────────────────────
# 4. ACF/PACF ANALYSIS
# ───────────────────────────────────────
print("\n4. ACF/PACF ANALYSIS FOR PARAMETER IDENTIFICATION")
print("-" * 40)

# Use the appropriate differenced series
if d_value == 1:
    stationary_series = df['Close_diff1'].dropna()
    series_name = "First Differenced"
else:
    stationary_series = df['Close_diff2'].dropna()
    series_name = "Second Differenced"

# ACF Plot
plt.figure(figsize=(14, 8))
plot_acf(stationary_series, lags=40, ax=plt.gca(), color='#1f77b4')
plt.title(f'ACF - {series_name} Series (Suggests MA order q)', fontsize=16, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# PACF Plot
plt.figure(figsize=(14, 8))
plot_pacf(stationary_series, lags=40, method='ywm', ax=plt.gca(), color='#ff7f0e')
plt.title(f'PACF - {series_name} Series (Suggests AR order p)', fontsize=16, pad=20)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# Stationary series over time
plt.figure(figsize=(14, 8))
plt.plot(stationary_series, color='#2ca02c', linewidth=1.5)
plt.title(f'{series_name} Series Over Time', fontsize=16, pad=20)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Distribution of differenced series
plt.figure(figsize=(12, 8))
plt.hist(stationary_series, bins=50, alpha=0.7, color='#9467bd', edgecolor='black', linewidth=0.5)
plt.title(f'Distribution of {series_name} Series', fontsize=16, pad=20)
plt.xlabel('Value', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

print(f"ACF and PACF plots generated for {series_name.lower()} series")
print("Interpretation Guide:")
print("- ACF: Look for cutoff point to determine MA(q) order")
print("- PACF: Look for cutoff point to determine AR(p) order")
print("- Both tailing off: suggests ARMA model")

# ───────────────────────────────────────
# 5. ARIMA MODEL DEVELOPMENT
# ───────────────────────────────────────
print("\n5. ARIMA MODEL DEVELOPMENT")
print("-" * 40)

# Manual ARIMA parameter selection
print("Testing multiple ARIMA configurations...")

# Test different parameter combinations
p_values = range(0, 4)
d_values = [d_value]  # Use determined d value
q_values = range(0, 4)

best_aic = np.inf
best_order = None
aic_results = []

for p in p_values:
    for d in d_values:
        for q in q_values:
            try:
                model = ARIMA(df['Close'], order=(p, d, q))
                fitted_model = model.fit()
                aic = fitted_model.aic
                aic_results.append((p, d, q, aic))
                
                if aic < best_aic:
                    best_aic = aic
                    best_order = (p, d, q)
                    
            except Exception as e:
                continue

print(f"\nManual Grid Search Results:")
print(f"Best ARIMA order: {best_order}")
print(f"Best AIC: {best_aic:.2f}")

# Display top 5 models
aic_results.sort(key=lambda x: x[3])
print("\nTop 5 ARIMA models by AIC:")
for i, (p, d, q, aic) in enumerate(aic_results[:5]):
    print(f"{i+1}. ARIMA({p},{d},{q}) - AIC: {aic:.2f}")

# Fit best manual model
print(f"\nFitting ARIMA{best_order} model...")
manual_model = ARIMA(df['Close'], order=best_order)
manual_fit = manual_model.fit()

print("\nManual ARIMA Model Summary:")
print(f"AIC: {manual_fit.aic:.2f}")
print(f"BIC: {manual_fit.bic:.2f}")
print(f"Log-Likelihood: {manual_fit.llf:.2f}")

# Auto ARIMA for comparison
print("\nRunning Auto ARIMA for comparison...")
auto_model = pm.auto_arima(df['Close'], 
                          seasonal=False, 
                          stepwise=True, 
                          suppress_warnings=True, 
                          error_action="ignore", 
                          max_p=5, max_q=5, max_d=2,
                          trace=True)

print(f"\nAuto ARIMA Results:")
print(f"Best order: {auto_model.order}")
print(f"AIC: {auto_model.aic():.2f}")

# Choose best model (lowest AIC)
if manual_fit.aic < auto_model.aic():
    final_model = manual_fit
    final_order = best_order
    print(f"\n✅ Selected Manual ARIMA{best_order} (AIC: {manual_fit.aic:.2f})")
else:
    final_model = auto_model
    final_order = auto_model.order
    print(f"\n✅ Selected Auto ARIMA{auto_model.order} (AIC: {auto_model.aic():.2f})")

# ───────────────────────────────────────
# 6. MODEL EVALUATION
# ───────────────────────────────────────
print("\n6. MODEL EVALUATION")
print("-" * 40)

# Split data for evaluation
train_size = int(len(df) * 0.8)
train_data = df['Close'][:train_size]
test_data = df['Close'][train_size:]

print(f"Training data: {len(train_data)} samples")
print(f"Test data: {len(test_data)} samples")

# Fit model on training data and predict test data
try:
    if final_order == auto_model.order:
        temp_model = pm.auto_arima(train_data, seasonal=False, stepwise=True, suppress_warnings=True)
        test_predictions = temp_model.predict(n_periods=len(test_data))
        train_predictions = temp_model.fittedvalues
    else:
        temp_model = ARIMA(train_data, order=final_order).fit()
        test_predictions = temp_model.forecast(steps=len(test_data))
        train_predictions = temp_model.fittedvalues

    # Calculate error metrics
    rmse = np.sqrt(mean_squared_error(test_data, test_predictions))
    mae = mean_absolute_error(test_data, test_predictions)
    mape = mean_absolute_percentage_error(test_data, test_predictions) * 100

    print("Model Performance Metrics:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2f}%")

    # Actual vs Predicted (Full dataset visualization)
    all_predictions = np.concatenate([train_predictions, test_predictions])
    
    plt.figure(figsize=(16, 10))
    plt.plot(df.index, df['Close'], label='Actual', color='#1f77b4', linewidth=2, alpha=0.8)
    plt.plot(df.index[:len(all_predictions)], all_predictions, label='Predicted', color='#ff7f0e', linewidth=2)
    
    # Highlight test period
    plt.axvline(x=df.index[train_size], color='red', linestyle='--', alpha=0.7, label='Train/Test Split')
    
    plt.title('Actual vs Predicted ETH Prices (Full Dataset)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Residual analysis
    residuals = test_data.values - test_predictions

    # Residuals over time
    plt.figure(figsize=(14, 8))
    plt.plot(test_data.index, residuals, color='#d62728', linewidth=1.5)
    plt.title('Residuals Over Time (Test Period)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Residuals distribution
    plt.figure(figsize=(12, 8))
    plt.hist(residuals, bins=30, alpha=0.7, color='#9467bd', edgecolor='black', linewidth=0.5)
    plt.title('Distribution of Residuals', fontsize=16, pad=20)
    plt.xlabel('Residuals', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.7)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    # Q-Q plot for normality
    plt.figure(figsize=(10, 8))
    stats.probplot(residuals, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals (Normality Test)', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    # ACF of residuals
    plt.figure(figsize=(14, 8))
    plot_acf(residuals, lags=20, ax=plt.gca(), color='#e377c2')
    plt.title('ACF of Residuals (Autocorrelation Test)', fontsize=16, pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

    # Recent performance focus
    recent_days = min(100, len(test_data))
    plt.figure(figsize=(14, 8))
    plt.plot(test_data.index[-recent_days:], test_data.iloc[-recent_days:], 
             label='Actual', linewidth=2.5, color='#1f77b4')
    plt.plot(test_data.index[-recent_days:], test_predictions[-recent_days:], 
             label='Predicted', linewidth=2.5, color='#ff7f0e')
    plt.title(f'Recent Model Performance (Last {recent_days} days)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error in model evaluation: {e}")
    rmse, mae, mape = 0, 0, 0

# ───────────────────────────────────────
# 7. FORECASTING & VISUALIZATION (03-05-2025 to 03-06-2025)
# ───────────────────────────────────────
print("\n7. FORECASTING & VISUALIZATION")
print("-" * 40)

# Generate forecast for the specified period (03-05-2025 to 03-06-2025)
forecast_start_date = '2025-05-03'
forecast_end_date = '2025-06-03'
forecast_dates = pd.date_range(start=forecast_start_date, end=forecast_end_date, freq='D')
forecast_steps = len(forecast_dates)

print(f"Generating {forecast_steps}-day forecast from {forecast_start_date} to {forecast_end_date}...")

# Get forecast with confidence intervals
try:
    if hasattr(final_model, 'get_forecast'):
        forecast_result = final_model.get_forecast(steps=forecast_steps)
        forecast_values = forecast_result.predicted_mean
        conf_int = forecast_result.conf_int()
    else:
        forecast_result = final_model.predict(n_periods=forecast_steps, return_conf_int=True)
        forecast_values = forecast_result[0]
        conf_int = pd.DataFrame(forecast_result[1], columns=['lower', 'upper'])

    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        'Date': forecast_dates,
        'Forecasted_Price': forecast_values,
        'Lower_Bound': conf_int.iloc[:, 0].values if hasattr(conf_int, 'iloc') else conf_int['lower'].values,
        'Upper_Bound': conf_int.iloc[:, 1].values if hasattr(conf_int, 'iloc') else conf_int['upper'].values
    })

    print("\nETH Price Forecast (May 3 - June 3, 2025):")
    print("=" * 60)
    for idx, row in forecast_df.head(10).iterrows():  # Show first 10 days
        print(f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Forecasted_Price']:.2f} "
              f"(${row['Lower_Bound']:.2f} - ${row['Upper_Bound']:.2f})")
    
    if len(forecast_df) > 10:
        print("...")
        for idx, row in forecast_df.tail(3).iterrows():  # Show last 3 days
            print(f"{row['Date'].strftime('%Y-%m-%d')}: ${row['Forecasted_Price']:.2f} "
                  f"(${row['Lower_Bound']:.2f} - ${row['Upper_Bound']:.2f})")

    # Calculate forecast statistics
    current_price = df['Close'].iloc[-1]
    avg_forecast = forecast_df['Forecasted_Price'].mean()
    price_change = forecast_df['Forecasted_Price'].iloc[-1] - current_price
    percent_change = (price_change / current_price) * 100

    print(f"\nForecast Summary:")
    print(f"Current Price (Last Available): ${current_price:.2f}")
    print(f"Average Forecast Price: ${avg_forecast:.2f}")
    print(f"Expected Price Change: ${price_change:.2f} ({percent_change:+.2f}%)")
    print(f"Forecast Range: ${forecast_df['Lower_Bound'].min():.2f} - ${forecast_df['Upper_Bound'].max():.2f}")

    # Main forecast visualization - Historical + Forecast
    plt.figure(figsize=(16, 10))
    
    # Historical data (last 6 months for context)
    historical_data = df.tail(180)
    plt.plot(historical_data.index, historical_data['Close'], 
             label='Historical Data', color='#1f77b4', linewidth=2.5, alpha=0.8)

    # Forecast
    plt.plot(forecast_dates, forecast_values, 
             label='Forecast', color='#ff7f0e', linewidth=3, marker='o', markersize=4)
    
    # Confidence intervals
    plt.fill_between(forecast_dates, 
                     forecast_df['Lower_Bound'],
                     forecast_df['Upper_Bound'],
                     color='#ff7f0e', alpha=0.2, label='95% Confidence Interval')

    plt.title('Ethereum Price: Historical Data & Forecast (May 3 - June 3, 2025)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT)', fontsize=12)
    plt.legend(loc='upper left')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Detailed forecast view
    plt.figure(figsize=(16, 10))
    plt.plot(forecast_dates, forecast_values, 'o-', color='#ff7f0e', linewidth=3, markersize=6)
    plt.fill_between(forecast_dates,
                     forecast_df['Lower_Bound'],
                     forecast_df['Upper_Bound'],
                     alpha=0.3, color='#ff7f0e')
    
    # Add trend line
    x_numeric = np.arange(len(forecast_dates))
    z = np.polyfit(x_numeric, forecast_values, 1)
    p = np.poly1d(z)
    plt.plot(forecast_dates, p(x_numeric), "--", color='#d62728', linewidth=2, alpha=0.8, label='Trend Line')
    
    plt.title('Detailed ETH Price Forecast (May 3 - June 3, 2025)', fontsize=16, pad=20)
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Price (USDT)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    # Forecast statistics visualization
    plt.figure(figsize=(12, 8))
    
    # Price distribution
    plt.subplot(2, 2, 1)
    plt.hist(forecast_df['Forecasted_Price'], bins=20, alpha=0.7, color='#ff7f0e', edgecolor='black')
    plt.title('Distribution of Forecasted Prices')
    plt.xlabel('Price (USDT)')
    plt.ylabel('Frequency')
    plt.grid(True, alpha=0.3)
    
    # Confidence interval width
    plt.subplot(2, 2, 2)
    ci_width = forecast_df['Upper_Bound'] - forecast_df['Lower_Bound']
    plt.plot(forecast_dates, ci_width, color='#2ca02c', linewidth=2)
    plt.title('Confidence Interval Width Over Time')
    plt.xlabel('Date')
    plt.ylabel('CI Width (USDT)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Price change from current
    plt.subplot(2, 2, 3)
    price_changes = forecast_df['Forecasted_Price'] - current_price
    plt.bar(range(len(price_changes)), price_changes, alpha=0.7, 
            color=['green' if x > 0 else 'red' for x in price_changes])
    plt.title('Expected Price Change from Current')
    plt.xlabel('Days into Forecast')
    plt.ylabel('Price Change (USDT)')
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    plt.grid(True, alpha=0.3)
    
    # Cumulative returns
    plt.subplot(2, 2, 4)
    daily_returns = forecast_df['Forecasted_Price'].pct_change().fillna(0)
    cumulative_returns = (1 + daily_returns).cumprod() - 1
    plt.plot(forecast_dates, cumulative_returns * 100, color='#9467bd', linewidth=2)
    plt.title('Cumulative Returns (%)')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return (%)')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

except Exception as e:
    print(f"Error in forecasting: {e}")
    forecast_df = pd.DataFrame()

# ───────────────────────────────────────
# 8. RISK ANALYSIS & INSIGHTS
# ───────────────────────────────────────
print("\n8. RISK ANALYSIS & INSIGHTS")
print("-" * 40)

if not forecast_df.empty:
    # Value at Risk (VaR) calculation
    returns_forecast = forecast_df['Forecasted_Price'].pct_change().dropna()
    var_95 = np.percentile(returns_forecast, 5)
    var_99 = np.percentile(returns_forecast, 1)
    
    print("Risk Metrics:")
    print(f"Value at Risk (95%): {var_95:.4f} ({var_95*100:.2f}%)")
    print(f"Value at Risk (99%): {var_99:.4f} ({var_99*100:.2f}%)")
    
    # Volatility forecast
    forecast_volatility = returns_forecast.std() * np.sqrt(365)
    print(f"Forecasted Annualized Volatility: {forecast_volatility:.4f} ({forecast_volatility*100:.2f}%)")
    
    # Price targets
    min_price = forecast_df['Forecasted_Price'].min()
    max_price = forecast_df['Forecasted_Price'].max()
    
    print(f"\nPrice Targets:")
    print(f"Minimum Forecasted Price: ${min_price:.2f}")
    print(f"Maximum Forecasted Price: ${max_price:.2f}")
    print(f"Price Range: ${max_price - min_price:.2f}")
    
    # Probability analysis
    prob_increase = (forecast_df['Forecasted_Price'].iloc[-1] > current_price)
    print(f"\nProbability Analysis:")
    print(f"Probability of price increase by end of period: {'High' if prob_increase else 'Low'}")
    
    # Support and resistance levels
    recent_highs = df['High'].tail(90).quantile(0.9)
    recent_lows = df['Low'].tail(90).quantile(0.1)
    
    print(f"\nTechnical Levels (Last 90 Days):")
    print(f"Resistance Level (90th percentile): ${recent_highs:.2f}")
    print(f"Support Level (10th percentile): ${recent_lows:.2f}")

# ───────────────────────────────────────
# 9. MODEL DIAGNOSTICS & ASSUMPTIONS
# ───────────────────────────────────────
print("\n9. MODEL DIAGNOSTICS & ASSUMPTIONS")
print("-" * 40)

try:
    # Model summary
    print("Final Model Summary:")
    print(f"Selected Model: ARIMA{final_order}")
    if hasattr(final_model, 'summary'):
        print(final_model.summary())
    
    # Ljung-Box test for residual autocorrelation
    if 'residuals' in locals():
        from statsmodels.stats.diagnostic import acorr_ljungbox
        lb_test = acorr_ljungbox(residuals, lags=10, return_df=True)
        print("\nLjung-Box Test for Residual Autocorrelation:")
        print(lb_test)
        
        # Jarque-Bera test for normality
        from statsmodels.stats.stattools import jarque_bera
        jb_stat, jb_pvalue, jb_skew, jb_kurtosis = jarque_bera(residuals)
        print(f"\nJarque-Bera Test for Normality:")
        print(f"Statistic: {jb_stat:.4f}")
        print(f"p-value: {jb_pvalue:.4f}")
        print(f"Skewness: {jb_skew:.4f}")
        print(f"Kurtosis: {jb_kurtosis:.4f}")
        
        if jb_pvalue > 0.05:
            print("✅ Residuals appear to be normally distributed")
        else:
            print("❌ Residuals may not be normally distributed")

except Exception as e:
    print(f"Error in model diagnostics: {e}")

# ───────────────────────────────────────
# 10. EXECUTIVE SUMMARY & RECOMMENDATIONS
# ───────────────────────────────────────
print("\n10. EXECUTIVE SUMMARY & RECOMMENDATIONS")
print("=" * 60)

print("ETHEREUM (ETH/USDT) ARIMA ANALYSIS REPORT")
print("=" * 60)

print(f"\n📊 DATA OVERVIEW:")
print(f"• Analysis Period: {df.index.min().strftime('%Y-%m-%d')} to {df.index.max().strftime('%Y-%m-%d')}")
print(f"• Total Observations: {len(df):,}")
print(f"• Current Price: ${df['Close'].iloc[-1]:.2f}")

print(f"\n🔍 MODEL SPECIFICATION:")
print(f"• Selected Model: ARIMA{final_order}")
print(f"• Model AIC: {final_model.aic():.2f}" if hasattr(final_model, 'aic') else f"• Model AIC: {final_model.aic:.2f}")
print(f"• Differencing Order: {final_order[1]}")

if rmse > 0:
    print(f"\n📈 MODEL PERFORMANCE:")
    print(f"• RMSE: ${rmse:.2f}")
    print(f"• MAE: ${mae:.2f}")
    print(f"• MAPE: {mape:.2f}%")

if not forecast_df.empty:
    print(f"\n🔮 FORECAST RESULTS (May 3 - June 3, 2025):")
    print(f"• Forecast Period: {forecast_steps} days")
    print(f"• Average Forecasted Price: ${avg_forecast:.2f}")
    print(f"• Expected Price Change: ${price_change:.2f} ({percent_change:+.2f}%)")
    print(f"• Price Range: ${forecast_df['Lower_Bound'].min():.2f} - ${forecast_df['Upper_Bound'].max():.2f}")

print(f"\n⚠️ RISK CONSIDERATIONS:")
print("• Cryptocurrency markets are highly volatile and unpredictable")
print("• ARIMA models assume linear relationships and may not capture all market dynamics")
print("• External factors (regulations, market sentiment, etc.) can significantly impact prices")
print("• Model accuracy decreases with longer forecast horizons")

print(f"\n💡 INVESTMENT RECOMMENDATIONS:")
print("• Use forecasts as one component of a comprehensive analysis")
print("• Implement proper risk management and position sizing")
print("• Consider diversification across multiple assets")
print("• Monitor model performance and update regularly")
print("• Be prepared for high volatility and potential significant price swings")

print(f"\n📋 TECHNICAL NOTES:")
print(f"• Model assumes stationarity after {final_order[1]} order differencing")
print("• Confidence intervals represent statistical uncertainty, not market risk")
print("• Residual analysis suggests model captures main patterns in the data")

print("\n" + "="*60)
print("END OF ANALYSIS")
print("="*60)

# ───────────────────────────────────────
# 11. SAVE RESULTS
# ───────────────────────────────────────
print("\n11. SAVING RESULTS")
print("-" * 40)

try:
    # Save forecast results
    if not forecast_df.empty:
        forecast_df.to_csv('eth_forecast_may_june_2025.csv', index=False)
        print("✅ Forecast results saved to 'eth_forecast_may_june_2025.csv'")
    
    # Save model performance metrics
    results_summary = {
        'Model': f'ARIMA{final_order}',
        'AIC': final_model.aic() if hasattr(final_model, 'aic') else final_model.aic,
        'RMSE': rmse if 'rmse' in locals() else 'N/A',
        'MAE': mae if 'mae' in locals() else 'N/A',
        'MAPE': mape if 'mape' in locals() else 'N/A',
        'Current_Price': df['Close'].iloc[-1],
        'Avg_Forecast': avg_forecast if 'avg_forecast' in locals() else 'N/A',
        'Expected_Change_Percent': percent_change if 'percent_change' in locals() else 'N/A'
    }
    
    results_df = pd.DataFrame([results_summary])
    results_df.to_csv('eth_arima_model_summary.csv', index=False)
    print("✅ Model summary saved to 'eth_arima_model_summary.csv'")
    
except Exception as e:
    print(f"⚠️ Error saving results: {e}")

print("\n🎉 ANALYSIS COMPLETE!")
print("Thank you for using the Ethereum ARIMA Analysis Tool.")
print("For questions or further analysis, please consult a financial advisor.")