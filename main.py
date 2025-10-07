
import config
from stock_data import StockData


# The tickers to fetch data for
tickers = config.TICKERS
start_date = config.START_DATE
end_date = config.END_DATE

print(f"Fetching data for {tickers} from {start_date} to {end_date}...")


def main():
    sd = StockData()
    sd.download()
    sd.add_indicators()
    sd.plot_moving_averages()

if __name__ == "__main__":
    main()