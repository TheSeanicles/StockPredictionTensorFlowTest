import pandas as pd
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
from os.path import exists
import os
import yaml


with open('config.yml', 'r') as file:
    config = yaml.safe_load(file)


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
    if exists(config['path'] + '/last_download.txt'):
        f = open(config['path'] + '/last_download.txt', 'r')
        last_download = f.read()
        f.close()
        last_date = datetime.datetime.strptime(last_download, '%Y-%m-%d %H:%M:%S.%f')
    else:
        last_date = tod - day
    if tod - last_date >= day:
        for t in tickers_list:
            # GRAB DATA FROM YAHOO FINANCE
            new_data = yf.download([t],
                               start=past.date(),
                               # end=tod.date(),
                               interval='5m',
                               group_by='ticker',
                               threads='True')

            # EXPORT DATA AND GRAB DATA TO BETTER MANIPULATE
            if exists(config['path'] + '/' + t + '.csv'):
                original_data = pd.read_csv(config['path'] + '/' + t + '.csv')
                new_data.merge(original_data)
                new_data.to_csv(config['path'] + '/' + t + '.csv')
            else:
                new_data.to_csv(config['path'] + '/' + t + '.csv')

        with open(config['path'] + '/last_download.txt', 'w') as f:
            write_str = tod.strftime('%Y-%m-%d %H:%M:%S.%f')
            f.write(write_str)


def delete_last_download():
    if exists(config['path'] + '/last_download.txt'):
        os.remove(config['path'] + '/last_download.txt')


def delete_all_data():
    files_in_path = os.listdir(config['path'])
    for f in files_in_path:
        os.remove(config['path'] + '/' + f)


if __name__ == '__main__':
    if config['delete_data']:
        delete_all_data()
    if config['delete_last_download']:
        delete_last_download()
    stock_save(get_sp500())
