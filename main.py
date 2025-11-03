
import config
from stock_data import StockData
import stock_transformation as st
def main():

    # Download stock data
    sd = StockData()
    sd.download()
    data = sd.data
    print(data['ORCL'].head())
    normalized_data, scaler = st.normalize(data,"ORCL")
    print(normalized_data['ORCL'].nlargest(10,"Adj Close"))
    print(normalized_data['ORCL'].nsmallest(10,"Adj Close"))

if __name__ == "__main__":
    main()