# Ethereum ARIMA Forecasting

A comprehensive time series analysis and forecasting tool for Ethereum prices using ARIMA modeling. This repository provides automated analysis, model selection, and price predictions for ETH/USDT trading pairs.

## Features

- Automated data preprocessing and exploratory data analysis
- Stationarity testing with ADF and KPSS tests
- ACF/PACF analysis for parameter identification
- Manual and automatic ARIMA model selection
- Model validation with train/test split
- 32-day price forecasting with confidence intervals
- Comprehensive risk analysis and performance metrics
- Residual analysis and model diagnostics

## Requirements

```txt
pandas>=1.5.0
numpy>=1.24.0
matplotlib>=3.6.0
seaborn>=0.12.0
statsmodels>=0.14.0
scikit-learn>=1.2.0
pmdarima>=2.0.0
scipy>=1.10.0
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/ethereum-arima-forecasting.git
cd ethereum-arima-forecasting
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your Ethereum price data in CSV format as `Binance_ETHUSDT_d.csv`
   - Ensure the file contains the following columns: Unix, Date, Symbol, Open, High, Low, Close, Volume ETH, Volume USDT, tradecount, market_cap, total_volume

## Data Format

The CSV file should follow this structure:

```
Unix,Date,Symbol,Open,High,Low,Close,Volume ETH,Volume USDT,tradecount,market_cap,total_volume
1.75E+12,01/05/2025,ETHUSDT,1793.62,1873.17,1792.52,1838.11,537198.7291,988206595.8,2398765,14097451632,7103187151
```

## Usage

Run the analysis script:

```bash
python eth_arima_analysis.py
```

The script will automatically:
1. Load and preprocess the data
2. Perform exploratory data analysis
3. Test for stationarity
4. Generate ACF/PACF plots
5. Select optimal ARIMA parameters
6. Train and evaluate the model
7. Generate 32-day forecasts
8. Save results to CSV files

## Output Files

- `eth_forecast_may_june_2025.csv` - Detailed forecast results with confidence intervals
- `eth_arima_model_summary.csv` - Model performance metrics and summary statistics
- Multiple visualization plots displayed during execution

## Model Evaluation Metrics

- **RMSE (Root Mean Square Error)**: Measures prediction accuracy
- **MAE (Mean Absolute Error)**: Average absolute prediction error
- **MAPE (Mean Absolute Percentage Error)**: Percentage-based error metric
- **AIC (Akaike Information Criterion)**: Model selection criterion

## Forecast Output

The forecast includes:
- Daily price predictions for the specified period
- 95% confidence intervals
- Expected price changes and percentage movements
- Risk metrics including Value at Risk (VaR)
- Technical support and resistance levels

## Risk Considerations

- Cryptocurrency markets are highly volatile and unpredictable
- ARIMA models assume linear relationships and may not capture all market dynamics
- External factors can significantly impact prices beyond model predictions
- Model accuracy decreases with longer forecast horizons
- Use forecasts as one component of comprehensive analysis, not as sole investment advice

## Technical Details

The analysis performs:
- Augmented Dickey-Fuller (ADF) and KPSS stationarity tests
- First and second-order differencing as needed
- Grid search for optimal (p,d,q) parameters
- Comparison between manual and auto-ARIMA model selection
- Ljung-Box test for residual autocorrelation
- Jarque-Bera test for residual normality

## Repository Structure

```
ethereum-arima-forecasting/
├── README.md
├── requirements.txt
├── eth_arima_analysis.py
├── Binance_ETHUSDT_d.csv
├── results/
└── plots/
```

## License

This project is for educational and research purposes. Not intended as financial advice.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## Support

For issues related to:
- Data format problems: Ensure CSV follows the specified structure
- Missing dependencies: Install all requirements from requirements.txt
- Model convergence: Check data quality and parameter ranges
- Performance issues: Consider reducing data size or forecast horizon
