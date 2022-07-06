import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt


# data = stockDataFetch(['AAPL'], 'max', '1d')
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#
# fetch data by interval (including intraday if period < 60 days)
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
def stock_data_fetch(tickers_list, time_duration, time_interval):
    return yf.download(tickers_list, period=time_duration, interval=time_interval, group_by='ticker', threads='True')


# plotStockData(['AAPL'], 'max', '1d')
# valid periods: 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
#
# fetch data by interval (including intraday if period < 60 days)
# valid intervals: 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
def plot_stock_data(tickers_list, time_duration, time_interval):
    data = stock_data_fetch(tickers_list, time_duration, time_interval)
    # Plot all the close prices
    ((data.pct_change() + 1).cumprod()).plot(figsize=(10, 7))

    # Show the legend
    plt.legend()

    # Define the label for the title of the figure
    plt.title("Returns", fontsize=16)

    # Define the labels for x-axis and y-axis
    plt.ylabel('Cumulative Returns', fontsize=14)
    plt.xlabel('Year', fontsize=14)

    # Plot the grid lines
    plt.grid(which="major", color='k', linestyle='-.', linewidth=0.5)
    plt.show()


def get_sp500():
    tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    return tickers.Symbol.to_list()


def stock_save(tickers_list):
    tod = datetime.datetime.now()
    d59 = datetime.timedelta(days=59)
    past = tod - d59
    day = datetime.timedelta(days=1)
    f = open('last_download.txt', 'r')
    try:
        last_download = f.read()
        f.close()
        last_date = datetime.datetime.strptime(last_download, '%Y-%m-%d %H:%M:%S.%f')
    except ValueError:
        f.close()
        last_date = tod - day
    if tod - last_date >= day:
        for t in tickers_list:
            # GRAB DATA FROM YAHOO FINANCE
            data = yf.download([t],
                               start=past.date(),
                               end=tod.date(),
                               interval='5m',
                               group_by='ticker',
                               threads='True')

            # EXPORT DATA AND GRAB DATA TO BETTER MANIPULATE
            data.to_csv('data/' + t + '.csv')

        with open('last_download.txt', 'w') as f:
            write_str = tod.strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write(write_str)


if __name__ == '__main__':
    stock_save(get_sp500())
