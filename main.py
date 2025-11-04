
import config
from stock_data import StockData
import stock_transformation as st
def main():

    # Download stock data
    sd = StockData()
    sd.download()
    data = sd.data
    moving_average_50 = st.moving_average_50(data)
    moving_average_200 = st.moving_averages_200(data)
    print(type(moving_average_50.head()))
    print(type(moving_average_200.head()))
if __name__ == "__main__":
    main()