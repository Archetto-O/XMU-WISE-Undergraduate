import numpy as np
import pandas as pd
from main_predict import write_csv
df = pd.read_excel('dataset/train_national_holiday_off.xlsx', header=None, index_col=0)


def judge_wkd(col):
    data = df[col].to_frame(name='value')
    data.index = pd.to_datetime(data.index)
    data['weekday'] = data.index.dayofweek
    data['is_weekday'] = data['weekday'].apply(lambda x: 1 if x < 5 else 0)
    weekday = data[data['is_weekday'] == 1]['value']
    weekend = data[data['is_weekday'] == 0]['value']
    return np.array(weekday), np.array(weekend)


def linfit(X,num):
    k, b = np.polyfit(X[:-2], X[1:-1], 1)
    predicted_values = [k * x + b for x in X[-num:]]
    return predicted_values


prediction = ['Predicted']
for col in df.columns:
    weekday, weekend = judge_wkd(col)
    prediction.extend(linfit(weekday,2))
    prediction.extend(linfit(weekend,1))

write_csv(prediction,'linreg.csv')


