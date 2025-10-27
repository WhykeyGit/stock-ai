# stock-ai
A personal project where AI predicts stock trends based on historic stock data.
Used to practice using various models on stock data.

## yfinance Historical Data Columns

When fetching historical data with `Ticker("AAPL").history()`, the returned DataFrame typically contains the following columns:

- **Open**: The stock price at the beginning of the trading day (first trade after market opens).
- **High**: The highest price reached during the trading day.
- **Low**: The lowest price reached during the trading day.
- **Close**: The stock price at the end of the trading day (last trade before market close).
- **Adj Close (Adjusted Close)**: The closing price adjusted for corporate actions such as dividends and stock splits.  
  This is the recommended column for comparing prices across time periods.
- **Volume**: The total number of shares traded during the trading day.

## Prerequisites

- Python 3.8 or higher
- pip package manager

## Installation
 **Clone the repository**

### Create a virtual environment (recommended)
  Windows: 
   ```python python -m venv venv
   venv\Scripts\activate
  ```
### Install required packages
```
pip install -r requirements.txt
```
#### or install manually:
 ```
 pip install pandas numpy yfinance plotly matplotlib scikit-learn tensorflow pyarrow
 ```
