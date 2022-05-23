import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists


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


def stock_predict(t):
    tod = datetime.datetime.now()
    d59 = datetime.timedelta(days=59)
    past = tod - d59

    # GRAB DATA FROM YAHOO FINANCE
    data = yf.download(t,
                       start=past.date(),
                       end=tod.date(),
                       interval='5m',
                       group_by='ticker',
                       threads='True')

    # EXPORT DATA AND GRAB DATA TO BETTER MANIPULATE
    data.to_csv('data/' + t + '.csv')

    print(t)
    if exists('data/' + t + '.csv'):
        df = pd.read_csv(r'data/' + t + '.csv')
        if len(df) > 2000:
            # MAKE DATE TIME MORE RELEVANT FOR RNN
            date_time = pd.to_datetime(df.pop('Datetime'), format='%Y-%m-%d %H:%M:%S%z')
            timestamp_s = date_time.map(pd.Timestamp.timestamp)
            day = 24 * 60 * 60
            year = 365.2425 * day

            df['Day sin'] = np.sin(timestamp_s * (2 * np.pi / day))
            df['Day cos'] = np.cos(timestamp_s * (2 * np.pi / day))
            df['Year sin'] = np.sin(timestamp_s * (2 * np.pi / year))
            df['Year cos'] = np.cos(timestamp_s * (2 * np.pi / year))

            # DIVIDE DATA FOR TRAINING VALIDATION AND TEST SET
            column_indices = {name: i for i, name in enumerate(df.columns)}

            n = len(df)
            in_df = df[0:int(n * 0.5)]
            out_df = df[int(n * 0.5):]

            num_features = df.shape[1]

            # DATA NORMALIZATION
            in_mean = in_df.mean()
            in_std = in_df.std()

            in_df = (in_df - in_mean) / in_std
            out_df = (out_df - in_mean) / in_std

            # PLOT NORMALIZED DATA
            # df_std = (df - in_mean) / in_std
            # df_std = df_std.melt(var_name='Column', value_name='Normalized')
            # plt.figure(figsize=(12, 6))
            # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
            # _ = ax.set_xticklabels(df.keys(), rotation=90)
            #
            # plt.show()

            class WindowGenerator():
                def __init__(self, input_width, label_width, shift,
                             in_df=in_df, out_df=out_df,
                             label_columns=None):
                    # Store the raw data.
                    self.in_df = in_df
                    self.out_df = out_df

                    # Work out the label column indices.
                    self.label_columns = label_columns
                    if label_columns is not None:
                        self.label_columns_indices = {name: i for i, name in
                                                      enumerate(label_columns)}
                    self.column_indices = {name: i for i, name in
                                           enumerate(in_df.columns)}

                    # Work out the window parameters.
                    self.input_width = input_width
                    self.label_width = label_width
                    self.shift = shift

                    self.total_window_size = input_width + shift

                    self.input_slice = slice(0, input_width)
                    self.input_indices = np.arange(self.total_window_size)[self.input_slice]

                    self.label_start = self.total_window_size - self.label_width
                    self.labels_slice = slice(self.label_start, None)
                    self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

                def __repr__(self):
                    return '\n'.join([
                        f'Total window size: {self.total_window_size}',
                        f'Input indices: {self.input_indices}',
                        f'Label indices: {self.label_indices}',
                        f'Label column name(s): {self.label_columns}'])

            def split_window(self, features):
                inputs = features[:, self.input_slice, :]
                labels = features[:, self.labels_slice, :]
                if self.label_columns is not None:
                    labels = tf.stack(
                        [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                        axis=-1)

                # Slicing doesn't preserve static shape information, so set the shapes
                # manually. This way the `tf.data.Datasets` are easier to inspect.
                inputs.set_shape([None, self.input_width, None])
                labels.set_shape([None, self.label_width, None])

                return inputs, labels

            WindowGenerator.split_window = split_window

            def plot(self, model=None, plot_col='Close', max_subplots=3):
                inputs, labels = self.example
                plt.figure(figsize=(12, 8))
                plot_col_index = self.column_indices[plot_col]
                max_n = min(max_subplots, len(inputs))
                for n in range(max_n):
                    plt.subplot(max_n, 1, n + 1)
                    plt.ylabel(f'{plot_col} [normed]')
                    plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                             label='Inputs', marker='.', zorder=-10)

                    if self.label_columns:
                        label_col_index = self.label_columns_indices.get(plot_col, None)
                    else:
                        label_col_index = plot_col_index

                    if label_col_index is None:
                        continue

                    plt.scatter(self.label_indices, labels[n, :, label_col_index],
                                edgecolors='k', label='Labels', c='#2ca02c', s=64)
                    if model is not None:
                        predictions = model(inputs)
                        plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                                    marker='X', edgecolors='k', label='Predictions',
                                    c='#ff7f0e', s=64)

                    if n == 0:
                        plt.legend()

                plt.xlabel('Time [h]')
                plt.show()

            WindowGenerator.plot = plot

            ################################################################################################
            # Model Test
            ################################################################################################
            lin = tf.keras.models.load_model('models/linear_stock_model')
            dnn = tf.keras.models.load_model('models/dense_stock_model')
            conv = tf.keras.models.load_model('models/conv_stock_model')
            rnn = tf.keras.models.load_model('models/rnn_stock_model')

            input_frame = WindowGenerator(input_width=100, label_width=100, shift=100, label_columns=['Close'])
            input_window = tf.stack([np.array(in_df[:input_frame.total_window_size]),
                           np.array(in_df[100:100+input_frame.total_window_size]),
                           np.array(in_df[200:200+input_frame.total_window_size])])
            stock_inputs, stock_labels = input_frame.split_window(input_window)

            input_frame.example = stock_inputs, stock_labels

            print('All shapes are: (batch, time, features)')
            print(f'Window shape: {input_window.shape}')
            print(f'Inputs shape: {stock_inputs.shape}')
            print(f'Labels shape: {stock_labels.shape}')

            input_frame.plot(model=lin)
            input_frame.plot(model=dnn)
            # input_frame.plot(model=conv)
            input_frame.plot(model=rnn)


if __name__ == '__main__':
    stock_predict('RBLX')

