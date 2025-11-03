"""
Handles downloading and caching of stock data using yfinance.
"""
import pandas as pd
import yfinance as yf
import os

import config

class StockData:
    def __init__(self):
        self.tickers = config.TICKERS
        self.start =config.START_DATE
        self.end = config.END_DATE
        self.data = None

    def download(self, use_cache=True, force_refresh=False, cache_path=None):
        """
        Downloads historical stock data for the specified tickers and date range.
        
        Args:
            use_cache: Whether to use caching (load and save)
            force_refresh: Force fresh download even if cache exists
            cache_path: Custom cache file path (defaults to config.DATA_CACHE_PATH)
        
        Returns:
            pd.DataFrame: The downloaded stock data
        """
        cache_file = cache_path or config.DATA_CACHE_PATH
        
        # Try loading from cache
        if use_cache and not force_refresh and os.path.exists(cache_file):
            try:
                print(f"Loading cached data from {cache_file}...")
                df = pd.read_parquet(cache_file)
                self.data = df
                self.copy_data = df.copy()
                print("Loaded from cache")
                return df  # Return here for cache hit
            except Exception as e:
                print(f"Cache load failed ({e}). Downloading fresh data...")
        
        # Download fresh data
        print(f"Downloading data for {self.tickers} from {self.start} to {self.end}...")
        try:
            df = yf.download(
                self.tickers, 
                start=self.start, 
                end=self.end, 
                group_by="ticker",
                auto_adjust=False
            )
            
            # Normalize structure for single tickers
            if isinstance(self.tickers, str) or len(self.tickers) == 1:
                ticker = self.tickers if isinstance(self.tickers, str) else self.tickers[0]
                # Check if columns are already a MultiIndex
                if not isinstance(df.columns, pd.MultiIndex):
                    df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
                # If it's already a MultiIndex but not properly structured, fix it
                elif df.columns.nlevels == 2 and df.columns.get_level_values(0)[0] != ticker:
                    # Rename the first level to the ticker name
                    df.columns = df.columns.set_levels([ticker], level=0)
            
            df.reset_index(inplace=True)
            self.data = df
            self.copy_data = df.copy()
            print(f"Downloaded {len(df)} rows")
            
        except Exception as e:
            print(f"Download failed: {e}")
            raise
        
        # Save to cache
        if use_cache:
            try:
                os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                df.to_parquet(cache_file)
                print(f"Data cached at {cache_file}")
            except Exception as e:
                print(f"Cache save failed: {e}")
        
        return df  # Return here after download
