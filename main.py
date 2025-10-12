
import config
from stock_data import StockData
from linear_regression import LRModel


# The tickers to fetch data for
tickers = config.TICKERS
start_date = config.START_DATE
end_date = config.END_DATE

print(f"Fetching data for {tickers} from {start_date} to {end_date}...")


def main():
    # sd = StockData()
    # sd.download()
    lr = LRModel()
    print(lr.data_with_target.head())
    # print(lr.data.head())
    # print(lr.data_with_target)
    # print(sd.data.head())
    # print(sd.data.columns)
    # print(sd.data[("ORCL", "Close")].head())

    # sd.add_indicators()

    # sd.plot_moving_averages()

if __name__ == "__main__":
    main()