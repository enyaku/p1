import datetime
import pandas_datareader.data as web

import matplotlib.pyplot as plt

#期間の設定
start = datetime.datetime(2017, 1, 1)
end = datetime.datetime(2017, 5, 30)

#株価取得
df = web.DataReader('TM', 'google', start, end)
df = toyota.drop('Volume', axis=1)  # DataFrameからVolumeを消去
df = df.loc[:, ['Open', 'Close', 'High','Low']]


print(df)

df.plot()
plt.show()
