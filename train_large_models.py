import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as np
import datetime
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
from os.path import exists

global multi_linear_model
global multi_dense_model
global multi_conv_model
global multi_lstm_model


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


def stock_predict(tickers_list):
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

    for t in tickers_list:
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
                train_df = df
                val_df = df

                num_features = df.shape[1]

                # DATA NORMALIZATION
                train_mean = train_df.mean()
                train_std = train_df.std()

                train_df = (train_df - train_mean) / train_std
                val_df = (val_df - train_mean) / train_std

                train_df.to_csv('train.csv')
                val_df.to_csv('val.csv')

                # PLOT NORMALIZED DATA
                # df_std = (df - train_mean) / train_std
                # df_std = df_std.melt(var_name='Column', value_name='Normalized')
                # plt.figure(figsize=(12, 6))
                # ax = sns.violinplot(x='Column', y='Normalized', data=df_std)
                # _ = ax.set_xticklabels(df.keys(), rotation=90)
                #
                # plt.show()

                class WindowGenerator():
                    def __init__(self, input_width, label_width, shift,
                                 train_df=train_df, val_df=val_df,
                                 label_columns=None):
                        # Store the raw data.
                        self.train_df = train_df
                        self.val_df = val_df

                        # Work out the label column indices.
                        self.label_columns = label_columns
                        if label_columns is not None:
                            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
                        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}

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
                        return '\n'.join([f'Total window size: {self.total_window_size}',
                                          f'Input indices: {self.input_indices}',
                                          f'Label indices: {self.label_indices}',
                                          f'Label column name(s): {self.label_columns}'])

                in_w = 1500
                OUT_STEPS = 1500
                MAX_EPOCHS = 30
                b_size = 1

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

                def make_dataset(self, dataset):
                    dataset = np.array(dataset, dtype=np.float32)
                    ds = tf.keras.utils.timeseries_dataset_from_array(
                        data=dataset,
                        targets=None,
                        sequence_length=self.total_window_size,
                        sequence_stride=1,
                        shuffle=False,
                        batch_size=b_size, )

                    ds = ds.map(self.split_window)

                    return ds

                WindowGenerator.make_dataset = make_dataset

                @property
                def train(self):
                    return self.make_dataset(self.train_df)

                @property
                def val(self):
                    return self.make_dataset(self.val_df)

                @property
                def example(self):
                    """Get and cache an example batch of `inputs, labels` for plotting."""
                    result = getattr(self, '_example', None)
                    if result is None:
                        # No example batch was found, so get one from the `.train` dataset
                        result = next(iter(self.train))
                        # And cache it for next time
                        self._example = result
                    return result

                WindowGenerator.train = train
                WindowGenerator.val = val
                WindowGenerator.example = example

                def compile_and_fit(model, window, patience=2):
                    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                                      patience=patience,
                                                                      mode='min')

                    model.compile(loss=tf.losses.MeanSquaredError(),
                                  optimizer=tf.optimizers.Adam(),
                                  metrics=[tf.metrics.MeanAbsoluteError()])

                    history = model.fit(window.train, epochs=MAX_EPOCHS,
                                        validation_data=window.val,
                                        callbacks=[early_stopping])
                    return history

                ########################################################################
                # Baseline
                ########################################################################
                multi_window = WindowGenerator(input_width=in_w,
                                               label_width=OUT_STEPS,
                                               shift=OUT_STEPS)

                # multi_window.plot()

                class MultiStepLastBaseline(tf.keras.Model):
                    def call(self, inputs):
                        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

                last_baseline = MultiStepLastBaseline()
                last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                                      metrics=[tf.metrics.MeanAbsoluteError()])

                multi_val_performance = {}

                multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val, verbose=0)
                # multi_window.plot(last_baseline)

                class RepeatBaseline(tf.keras.Model):
                    def call(self, inputs):
                        return inputs

                repeat_baseline = RepeatBaseline()
                repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                                        metrics=[tf.metrics.MeanAbsoluteError()])

                multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val, verbose=0)
                # multi_window.plot(repeat_baseline)

                ########################################################################
                # Linear
                ########################################################################
                # print('LINEAR')
                # multi_linear_model = tf.keras.Sequential([
                #     # Take the last time-step.
                #     # Shape [batch, time, features] => [batch, 1, features]
                #     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                #     # Shape => [batch, 1, out_steps*features]
                #     tf.keras.layers.Dense(OUT_STEPS * num_features,
                #                           kernel_initializer=tf.initializers.zeros()),
                #     # Shape => [batch, out_steps, features]
                #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
                # ])
                #
                # history = compile_and_fit(multi_linear_model, multi_window)
                #
                # multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val, verbose=0)
                # multi_window.plot(multi_linear_model)

                ########################################################################
                # DNN
                ########################################################################
                print('DNN')
                multi_dense_model = tf.keras.Sequential([
                    # Take the last time step.
                    # Shape [batch, time, features] => [batch, 1, features]
                    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
                    # Shape => [batch, 1, dense_units]
                    tf.keras.layers.Dense(13500, activation='sigmoid'),
                    # Shape => [batch, 1, dense_units]
                    tf.keras.layers.Dense(9000, activation='sigmoid'),
                    # Shape => [batch, out_steps*features]
                    tf.keras.layers.Dense(OUT_STEPS * num_features,
                                          kernel_initializer=tf.initializers.zeros()),
                    # Shape => [batch, out_steps, features]
                    tf.keras.layers.Reshape([OUT_STEPS, num_features])
                ])

                history = compile_and_fit(multi_dense_model, multi_window)

                multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val, verbose=0)
                multi_window.plot(multi_dense_model)

                ########################################################################
                # CNN
                ########################################################################
                # print('CNN')
                # CONV_WIDTH = 3
                # multi_conv_model = tf.keras.Sequential([
                #     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
                #     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
                #     # Shape => [batch, 1, conv_units]
                #     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
                #     # Shape => [batch, 1,  out_steps*features]
                #     tf.keras.layers.Dense(OUT_STEPS * num_features,
                #                           kernel_initializer=tf.initializers.zeros()),
                #     # Shape => [batch, out_steps, features]
                #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
                # ])
                #
                # history = compile_and_fit(multi_conv_model, multi_window)
                #
                # multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val, verbose=0)
                # multi_window.plot(multi_conv_model)

                ########################################################################
                # RNN
                ########################################################################
                # print('RNN')
                # multi_lstm_model = tf.keras.Sequential([
                #     # Shape [batch, time, features] => [batch, lstm_units].
                #     # Adding more `lstm_units` just overfits more quickly.
                #     tf.keras.layers.LSTM(32, return_sequences=False),
                #     # Shape => [batch, out_steps*features].
                #     tf.keras.layers.Dense(OUT_STEPS * num_features,
                #                           kernel_initializer=tf.initializers.zeros()),
                #     # Shape => [batch, out_steps, features].
                #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
                # ])
                #
                # history = compile_and_fit(multi_lstm_model, multi_window)
                #
                # multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val, verbose=0)
                # multi_window.plot(multi_lstm_model)

                ########################################################################
                # Bench Comparison
                ########################################################################
                # x = np.arange(len(multi_val_performance))
                # width = 0.3
                #
                # metric_index = multi_linear_model.metrics_names.index('mean_absolute_error')
                # val_mae = [v[metric_index] for v in multi_val_performance.values()]
                #
                # plt.bar(x - 0.17, val_mae, width, label='Validation')
                # plt.xticks(ticks=x, labels=multi_val_performance.keys(),
                #            rotation=45)
                # plt.ylabel(f'MAE (average over all times and outputs)')
                # _ = plt.legend()
                #
                # plt.show()
                # for name, value in multi_val_performance.items():
                #     print(f'{name:8s}: {value[1]:0.4f}')
    # multi_linear_model.save('models/linear_stock_model')
    multi_dense_model.save('models/dense_stock_model')
    # multi_conv_model.save('models/conv_stock_model')
    # multi_lstm_model.save('models/rnn_stock_model')


if __name__ == '__main__':
    stock_predict(['AAPL'])
    # stock_predict(get_sp500())

