
import config
from stock_data import StockData
import stock_transformation as st
def main():

    # Download stock data
    sd = StockData()
    sd.download()
    data = sd.data
    data_nvda = data['NVDA']
    macd_df = st.macd_single_ticker(data_nvda)
    print(macd_df.head())

if __name__ == "__main__":
    main()