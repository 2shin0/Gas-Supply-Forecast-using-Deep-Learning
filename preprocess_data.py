import pandas as pd
import numpy as np
from datetime import datetime
from workalendar.asia import SouthKorea

def preprocess_total_data(total_path, temp_path):
    total = pd.read_csv(total_path, encoding='cp949')

    d_map = {}
    for i, d in enumerate(total['구분'].unique()):
        d_map[d] = i
    total['구분'] = total['구분'].map(d_map)

    total['연월일'] = pd.to_datetime(total['연월일'])
    total['year'] = total['연월일'].dt.year
    total['month'] = total['연월일'].dt.month
    total['day'] = total['연월일'].dt.day
    total['weekday'] = total['연월일'].dt.weekday
    total['시간'] = total['시간'] - 1

    holidays = pd.concat([pd.Series(np.array(SouthKorea().holidays(year))[:, 0]) for year in range(2013, 2019)]).reset_index(drop=True)
    total['휴일'] = total['연월일'].dt.date.isin(holidays).astype(int)
    total['주말'] = total['weekday'].map({0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 1, 6: 1})
    total['휴일_주말'] = (total['주말'] + total['휴일']).map({0: 0, 1: 1, 2: 1})

    temp = pd.read_csv(temp_path, encoding='cp949')
    temp['연월일'] = pd.to_datetime(temp['연월일'])
    temp['year'] = temp['연월일'].dt.year
    temp['month'] = temp['연월일'].dt.month
    temp['day'] = temp['연월일'].dt.day
    temp['weekday'] = temp['연월일'].dt.weekday
    temp['시간'] = temp['연월일'].dt.time
    temp['시간'] = temp['시간'].apply(lambda x: x.strftime('%H')).astype(int)
    temp = temp.drop(['연월일'], axis=1)

    total = pd.merge(total, temp, how="left", on=["year", 'month', 'day', 'weekday', '시간'])
    total['temp'].fillna(total['temp'].mean(), inplace=True)
    total = total.rename(columns={'시간': 'time', "구분": "kind_of_gas", "공급량(톤)": "supply"})

    return total