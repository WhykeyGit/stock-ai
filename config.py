# ==============================
#  Data Settings
# ==============================

# List of stock tickers to analyze
TICKERS = [
    "NVDA",
    "ORCL",
    "MSFT",
    "AMZN",
]

# Date range for historical stock data
START_DATE = "2015-01-01"
END_DATE = "2025-01-01"


# ==============================
# Machine Learning Settings
# ==============================

# Train-test split ratio for supervised learning
TRAIN_TEST_SPLIT = 0.8

# Random seed for reproducibility
RANDOM_SEED = 42

# ==============================
# Plotting Settings
# ==============================

# Matplotlib figure size
FIGURE_SIZE = (14, 7)

# Plotly template style (e.g., "plotly_dark", "ggplot2", "seaborn")
PLOTLY_TEMPLATE = "plotly_white"

# ==============================
# File Paths
# ==============================

# Where to cache downloaded data (optional)
DATA_CACHE_PATH = "D:/VSCode/stocksProject/stock-ai/data/cache.parquet"

# Where to save trained models (optional)
MODEL_SAVE_PATH = "models/"

# ==============================
# Feature Engineering Settings
# ==============================

# Moving average windows to compute
MOVING_AVERAGE_WINDOWS = [50, 200]

# Rolling window for volatility calculation (days)
VOLATILITY_WINDOW = 21

# Annualization factor for volatility
TRADING_DAYS_PER_YEAR = 252