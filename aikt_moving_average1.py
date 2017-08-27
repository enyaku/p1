# -*- coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys

import pandas as pd
import jsm
import datetime
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




# 移動平均線の計算(データ, 日数)
def move_average(data, day):
    return np.convolve(data, np.ones(day)/float(day), 'valid')
    
def main():
    args = sys.argv
    print( args)

    print( u'銘柄コード（第１引数）：' + args[1])
    print( u'前年データ（第２引数）：' + args[2])
    print( u'当年データ（第３引数）：' + args[3])
    print( u'移動平均線（第４引数）：' + args[4])
    print( u'移動平均線（第５引数）：' + args[5])

    data1 = args[1]
    data2 = args[2]
    data3 = args[3]
    data4 = int(args[4])
    data5 = int(args[5])



    # 株価の取得(銘柄コード, 開始日, 終了日)
    code = args[1]
    today = datetime.date.today()
    print(today)
    todaystr = today.strftime('%Y-%m-%d')
    print(todaystr)

    data_to_df1 = get_stock(code, '2017-1-1', todaystr)
    
    dataDF1 = "data_to_df1.csv"
    dataDF2 = "data_to_df2.csv"
    dataDF3 = "data_to_df3.csv"


    # データフレームの作成
    df1 = pd.DataFrame({'始値':data_to_df1[1], '終値':data_to_df1[2], '高値':data_to_df1[3], '安値':data_to_df1[4]}, index = data_to_df1[0])
    df1.to_csv(dataDF1)

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    data_to_df2 = get_stock(code, '2016-1-1', '2016-12-31')

    # データフレームの作成
    df2 = pd.DataFrame({'始値':data_to_df2[1], '終値':data_to_df2[2], '高値':data_to_df2[3], '安値':data_to_df2[4]}, index = data_to_df2[0])
    df2.to_csv(dataDF2)


    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

    data_to_df3 = get_stock(code, '2016-1-1', todaystr)

    # データフレームの作成
    df3 = pd.DataFrame({'始値':data_to_df3[1], '終値':data_to_df3[2], '高値':data_to_df3[3], '安値':data_to_df3[4]}, index = data_to_df3[0])
    df3.to_csv(dataDF3)





     # CSVのロード(2015年と2016年のデータ)
    #data15 = np.genfromtxt("nikkei15.csv", delimiter=",", skip_header=1, dtype='float')
    #data16 = np.genfromtxt("nikkei16.csv", delimiter=",", skip_header=1, dtype='float')

    #data15 = np.genfromtxt(data2, delimiter=",", skip_header=1, dtype='float')
    #data16 = np.genfromtxt(data3, delimiter=",", skip_header=1, dtype='float')

    data15 = np.genfromtxt(dataDF1, delimiter=",", skip_header=1, dtype='float')
    data16 = np.genfromtxt(dataDF2, delimiter=",", skip_header=1, dtype='float')


    # 5列目の終値だけを日付古い順に並び替えて取り出し
    f15, f16 = data15[:,4], data16[:,4]
    f15, f16 = f15[::-1], f16[::-1]
    
    # 移動平均線(25日線)の計算
    day = data4    # 日数
    data = np.r_[f15[len(f15)-day+1:len(f15)], f16]    #　2015年の終値の一部と2016年の終値を結合
    ma_25d = move_average(data, day)

    print(data)

    # 移動平均線(75日線)の計算
    day = data5    # 日数
    data = np.r_[f15[len(f15)-day+1:len(f15)], f16]    #　2015年の終値の一部と2016年の終値を結合
    ma_75d = move_average(data, day)
    

    print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    print(data)

    # グラフにプロット
    plt.plot(f16,  label="f")
    plt.plot(ma_25d, "--", color="r", label="MA 25d")
    plt.plot(ma_75d, "--", color="g", label="MA 75d")  
    

    #　ラベル軸
    plt.title(data1)
    plt.xlabel("Day")
    plt.ylabel("f")
    # 凡例
    plt.legend(loc="4")
    # グリッド
    plt.grid()
    # グラフ表示
    plt.show()

    
if __name__ == "__main__":
    main()

