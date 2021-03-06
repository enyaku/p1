#-*- coding:utf-8 -*-
import pandas as pd
import jsm
import datetime
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from matplotlib.dates import date2num

# 株価のデータ取得（銘柄コード, 開始日, 終了日）
def get_stock(code, start_date, end_date):
    # 期間設定
    year, month, day = start_date.split("-")
    start = datetime.date(int(year), int(month), int(day))
    year, month, day = end_date.split("-") 
    end = datetime.date(int(year), int(month), int(day))
    # 株価データ取得
    q = jsm.Quotes()
    target = q.get_historical_prices(code, jsm.DAILY, start_date = start, end_date = end)
    # 項目ごとにリストに格納して返す
    date = [data.date for data in target]
    open = [data.open for data in target]
    close = [data.close for data in target]
    high = [data.high for data in target]
    low = [data.low for data in target]
    # 日付が古い順に並び替えて返す
    return [date[::-1], open[::-1], close[::-1], high[::-1], low[::-1]]
    
def main():
    # 株価の取得(銘柄コード, 開始日, 終了日)
    code = 3906
    today = datetime.date.today()

    print(today)
    todaystr = today.strftime('%Y-%m-%d')
    data = get_stock(code, '2017-6-1', todaystr)
    print(todaystr)
    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    # データフレームの作成
    df = pd.DataFrame({'始値':data[1], '終値':data[2], '高値':data[3], '安値':data[4]}, index = data[0])


    print(df)
    
    # グラフにプロット
    fig = plt.figure()
    ax = plt.subplot()
    mpf.candlestick2_ohlc(ax, df['始値'], df['高値'], df['安値'], df['終値'], width=0.5, colorup="g", colordown="r")

    ax.set_xticklabels([(df.index[int(x)].strftime("%Y/%M/%D") if x <= df.shape[0] else x) for x in ax.get_xticks()], rotation=90)
   
    ax.set_xlim([0, df.shape[0]]) # 横軸の範囲はデータの個数(df.shape[0]個)までに変更しておく

    ax.grid()
    ax.legend()
    fig.autofmt_xdate() #x軸のオートフォーマット
    plt.show()
    
if __name__ == "__main__":
    main()

