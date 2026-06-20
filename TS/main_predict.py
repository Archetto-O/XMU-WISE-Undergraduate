import os.path
from itertools import product
import numpy as np
import pandas as pd
import xgboost as xgb
from pmdarima import auto_arima
from sklearn.preprocessing import MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from lightgbm import LGBMRegressor
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from icecream import ic
import tkinter as tk
from tkinter import messagebox


# load data
# df = pd.read_excel('dataset/train.xlsx', header=None, index_col=0)
df = pd.read_excel('dataset/train_national_holiday_off.xlsx', header=None, index_col=0)


def decompose(col):
    result = seasonal_decompose(pd.Series(col), period=7, two_sided=False)
    trend = result.trend.dropna()
    seasonal = result.seasonal.dropna()
    residual = result.resid.dropna()

    return trend, seasonal, residual


def ARIMA_(series):
    def autoarima(series):
        ARIMA_model = auto_arima(series, max_P=4, max_Q=4)
        forecast = ARIMA_model.predict(n_periods=3)
        return forecast

    def customarima(series):
        p = 2
        d = 1
        q = 2
        ARIMA_model = ARIMA(series, order=(p, d, q),freq='D')
        fit = ARIMA_model.fit()
        forecast = fit.forecast(steps=3)

    forecast = autoarima(series)
    # forecast = customarima(series)
    return forecast.tolist()


def xgboost(kind,series=None,df=None,window=7,window2=3,window3= 1,max_depth=8,learning_rate = 0.02):
    prediction = []
    if df is None:
        df = create_feature(series,kind)
    for i in range (3):
        X = df.iloc[:, 1:]  # 特征
        y = df.iloc[:, 0]  # 目标值

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        model = xgb.XGBRegressor(objective='reg:squarederror',
                                 learning_rate=learning_rate,
                                 max_depth=max_depth, n_estimators=500, early_stopping_rounds=2,
                                 # num_leaves=2**max_depth,
                                 random_state=42)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        future_day = (df.index[-1] + pd.Timedelta(days=1)).date()
        future_features = pd.DataFrame({
            'month': [future_day.month],
            'day': [future_day.day],
            'weekday': [future_day.weekday()],
            'is_weekday': [1 if future_day.weekday() < 5 else 0],
            str(window) + '_day_avg': [df[str(window) + '_day_avg'].iloc[-1]],
            str(window2) + '_day_avg': [df[str(window2) + '_day_avg'].iloc[-1]]})
        if window3 is not None:
            future_features[str(window3) + '_day_avg'] = [df[str(window3) + '_day_avg'].iloc[-1]]


        future_prediction = model.predict(future_features)
        prediction.append(future_prediction[0])

        new_row = pd.DataFrame({
            kind: future_prediction,
            'month': [future_day.month],
            'day': [future_day.day],
            'weekday': [future_day.weekday()],
            'is_weekday': [1 if future_day.weekday() < 5 else 0],
            str(window) + '_day_avg': (df[str(window) + '_day_avg'].iloc[-1]) - df[kind][-window]/window + future_prediction/window,
            str(window2) + '_day_avg': (df[str(window2) + '_day_avg'].iloc[-1]) - df[kind][-window2]/window2 + future_prediction/window2},
            index=[pd.to_datetime(future_day)])
        if window3 is not None:
            new_row[str(window3) + '_day_avg'] = (df[str(window3) + '_day_avg'].iloc[-1]) - df[kind][-window3] / window3 + future_prediction / window3

        df = pd.concat([df, new_row])

    return prediction


def lightGBM(kind, series=None, df=None, window=7, window2 = 3, window3=1):
    prediction = []
    if df is None:
        df = create_feature(series, kind)
    for i in range (3):
        X = df.iloc[:, 1:]  # 特征
        y = df.iloc[:, 0]  # 目标值

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        model = LGBMRegressor(num_leaves= 5, learning_rate=0.02,
                              n_estimators=500, early_stopping_rounds=2, random_state=7)

        model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

        future_day = (df.index[-1] + pd.Timedelta(days=1)).date()
        future_features = pd.DataFrame({
            'month': [future_day.month],
            'day': [future_day.day],
            'weekday': [future_day.weekday()],
            str(window) + '_day_avg': [df[str(window) + '_day_avg'].iloc[-1]],
            str(window2) + '_day_avg': [df[str(window2) + '_day_avg'].iloc[-1]],
            str(window3) + '_day_avg': [df[str(window3) + '_day_avg'].iloc[-1]]})

        future_prediction = model.predict(future_features)
        prediction.append(future_prediction[0])

        new_row = pd.DataFrame({
            kind: future_prediction,
            'month': [future_day.month],
            'day': [future_day.day],
            'weekday': [future_day.weekday()],
            str(window) + '_day_avg': (df[str(window) + '_day_avg'].iloc[-1]) - df[kind][-window]/window + future_prediction/window,
            str(window2) + '_day_avg': (df[str(window2) + '_day_avg'].iloc[-1]) - df[kind][-window2]/window2 + future_prediction/window2,
            str(window3) + '_day_avg': (df[str(window3) + '_day_avg'].iloc[-1]) - df[kind][-window3]/window3 + future_prediction/window3},
            index=[pd.to_datetime(future_day)])

        df = pd.concat([df, new_row])

    return prediction


def SARIMA_X_(series = None,featured_df = None,kind = None,window = None, window2=None, max=3, way = 'sarimax'):

    def find_best_params(series, max):
        p = range(1, max)
        d = [1]
        q = range(1, max)
        pdq = list(product(p, d, q))
        seasonal_pdq = [(x[0], x[1], x[2], 7) for x in list(product(p, d, q))]

        # 寻找最佳参数组合
        best_aic = 10000000
        best_order = None
        best_seasonal_order = None
        for order in tqdm(pdq, total=len(pdq)):
            for seasonal_order in seasonal_pdq:
                model = SARIMAX(series, order=order, seasonal_order=seasonal_order)
                ic.enable()
                ic(series)
                ic.disable()
                results = model.fit()
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = order
                    best_seasonal_order = seasonal_order
        return best_order, best_seasonal_order

    def auto_find_param(data):
        stepwise_fit = auto_arima(data, start_p=1, start_q=1,
                                  max_p=3, max_q=3, m=7)
        best_params = stepwise_fit.order
        best_params_seasonal = stepwise_fit.seasonal_order
        return best_params, best_params_seasonal

    def SARIMA_(series, best_order, best_seasonal_order):
        model = SARIMAX(series, order=best_order, seasonal_order=best_seasonal_order)
        results = model.fit()
        forecast = results.forecast(steps=3)
        return forecast

    def SARIMAX_(featured_df, kind, window,window2, best_order, best_seasonal_order):
        exog = featured_df[['month', 'day', 'weekday', str(window)+'_day_avg', str(window2)+'_day_avg']]
        model = SARIMAX(featured_df[kind], exog=exog,
                        order=best_order, seasonal_order=best_seasonal_order)
        results = model.fit()
        forecast = []
        avg = featured_df[str(window) + '_day_avg'].iloc[-1]
        avg2 = featured_df[str(window2) + '_day_avg'].iloc[-1]
        for i in range(3):
            future_day = featured_df.index[-1] + pd.Timedelta(days=1+i)
            uni_forecast = results.get_forecast(steps=1,
                            exog = pd.DataFrame({'month':future_day.month, 'day':future_day.day, 'weekday':future_day.weekday(),
                                                str(window)+'_day_avg': avg, str(window2)+'_day_avg': avg2},
                                                index = [pd.to_datetime(future_day)]))
            forecast.append(uni_forecast.predicted_mean.iloc[0])
            avg = avg + uni_forecast.predicted_mean.iloc[0] / window - featured_df[str(window) + '_day_avg'].iloc[-window + i] / window
            avg2 = avg2 + uni_forecast.predicted_mean.iloc[0] / window2 - featured_df[str(window2) + '_day_avg'].iloc[-window2 + i] / window2
        return forecast


    if series is None:
        series = featured_df[kind]
    # best_order, best_seasonal_order = find_best_params(series, max)
    best_order, best_seasonal_order = auto_find_param(series)

    if way == 'sarima':
        forecast = SARIMA_(series, best_order, best_seasonal_order)
    if way == 'sarimax':
        forecast = SARIMAX_(featured_df,kind,window,window2, best_order, best_seasonal_order)
    return forecast


def LSTM_(series,kind,look_back=3,epochs = 200): # 很差

    def create_dataset(X, y, time_steps):
        Xs, ys = [], []
        for i in range(len(X) - time_steps):
            v = list(X.iloc[i:(i + time_steps)].values)
            Xs.append(v)
            ys.append(y.iloc[i + time_steps])
        return np.array(Xs), np.array(ys)

    df = create_feature(series,kind)
    features = ["day", "month", "weekday", "7_day_avg"]
    n_features = len(features)

    # 预处理
    scaler = MinMaxScaler(feature_range=(0, 1))
    df_scaled = df[features].copy()
    df_scaled[kind] = scaler.fit_transform(df[[kind]])

    # 准备数据集
    X, y = create_dataset(df_scaled[features], df_scaled[kind], look_back)

    # 构建LSTM
    model = Sequential()
    model.add(LSTM(50, input_shape=(look_back, n_features)))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')

    # 训练
    model.fit(X, y,
              epochs=epochs, batch_size=64,
              verbose=1)

    # 预测未来3个时间点
    predictions = []
    last_sequence = df_scaled[features].values[-look_back:]
    current_sequence = last_sequence
    ic(current_sequence)
    for i in range(3):
        future_day = (df.index[-1] + pd.Timedelta(days=1+i)).date()
        pred = model.predict(current_sequence[np.newaxis, :, :])
        predictions.append(pred[0, 0])
        pred = np.array([future_day.day,future_day.month,future_day.weekday(), pred[0, 0]])
        ic(pred)
        current_sequence = np.concatenate((current_sequence, pred.reshape(1, -1)),axis=0)
        ic(current_sequence)
        current_sequence = current_sequence[1:]
        ic(current_sequence)


    # 反归一
    predictions = scaler.inverse_transform([predictions])
    return predictions.flatten().tolist()


def plain_periodic(series):
    series = list(series)
    length = len(series)
    next_indices = [(length - 1 + i) % 7 for i in range(1, 4)]
    next_values = [series[i] for i in next_indices]

    return next_values


def create_feature(data,kind,window=7,window2=3,window3=None):
    data = pd.DataFrame(data)
    data.index = pd.to_datetime(data.index)
    data['month'] = data.index.month
    data['day'] = data.index.day
    data['weekday'] = data.index.dayofweek
    data['is_weekday'] = data['weekday'].apply(lambda x: 1 if x < 5 else 0)
    ic(data)
    data[str(window)+'_day_avg'] = data[kind].rolling(window=window).mean()
    data[str(window2) + '_day_avg'] = data[kind].rolling(window=window2).mean()
    if window3 is not None:
        data[str(window3) + '_day_avg'] = data[kind].rolling(window=window3).mean()

    return data.dropna()

def trend_processing(series):
    # return xgboost(series,'trend')
    # return SARIMA_X_(series = series)
    return ARIMA_(series)
    # return LSTM_(series,'trend')

def season_processing(series):
    # return SARIMA_X_(series=series,kind='seasonal',way='sarima')
    return plain_periodic(series)

def residual_processing(series):
    # return ARIMA_(series)
    # return xgboost(series,'resid')
    return xgboost(series=series,kind = 'resid')


def write_csv(prediction,filename='prediction.csv'):
    df = pd.read_csv('dataset/submission_3days.csv',header=None)
    df.iloc[:, 1] = prediction
    if os.path.exists(filename):
        os.remove(filename)
    df.to_csv(filename, index=False, header=None)


def predict_decomposed(df):
    ic.enable()
    ic.disable()
    # testing = True
    testing = False
    prediction = ['Predicted']
    for col in df.columns:
        trend, seasonal, residual = decompose(df[col])
        trend_prediction = trend_processing(trend)
        ic(trend_prediction)
        if testing:
            break
        season_prediction = season_processing(seasonal)
        ic(season_prediction)
        residual_prediction = residual_processing(residual)
        ic(residual_prediction)
        prediction.extend([a + b + c for a, b, c in zip(trend_prediction, season_prediction, residual_prediction)])
        ic(prediction)

    if not testing:
        print(prediction)
        write_csv(prediction,filename='tr a;se pl;re xg; nh_off.csv')

def predict_as_whole(df,window = 7,window2=3,window3=None):   # 根据方程分类预测方法
    ic.enable()
    testing = False
    scale = False
    prediction = ['Predicted']
    for col in df.columns:
        # var = np.var(df[col])
        extended_series = df[col].to_frame(name='value')
        if scale:
            scaler = MinMaxScaler()
            extended_series['value'] = scaler.fit_transform(extended_series[['value']])
        featured_df = create_feature(extended_series, 'value', window=window,window2=window2,window3 = window3)
        ic(extended_series)
        ic(featured_df)
        if scale:
            scaled_prediction = SARIMA_X_(series=None, featured_df=featured_df, kind='value', window=window,window2=window2, way='sarimax')
            prediction.extend(np.array(scaler.inverse_transform([scaled_prediction])).flatten().tolist())
        else:
            # prediction.extend(LSTM_(featured_df,'value',3,epochs=300))
            prediction.extend(xgboost(df=featured_df, kind='value',max_depth= 7,window = window,window2 = window2,window3 = window3))
            # prediction.extend(lightGBM(df=featured_df, kind='value',window=window,window2 = window2))
            # prediction.extend(ARIMA_(df[col]))
            # prediction.extend(SARIMA_X_(series=None, featured_df = featured_df,kind='value',window=7,max= 4,way='sarimax'))
        if testing:
            print(prediction)
            return 0
    print(prediction)
    write_csv(prediction,filename='wkd_feature.csv')


if __name__ == "__main__":
    # predict_decomposed(df)
    predict_as_whole(df)
    messagebox.showinfo('done','done')
