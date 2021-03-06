# -*- coding: utf-8
import os

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
    print( u'開始日　　（第２引数）：' + args[2])
    print( u'終了日　　（第３引数）：' + args[3])
    print( u'移動平均線（第４引数）：' + args[4])
    print( u'移動平均線（第５引数）：' + args[5])

    data1 = args[1]
    data2 = args[2]
    data3 = args[3]
    data4 = int(args[4])
    data5 = int(args[5])



    # 株価の取得(銘柄コード, 開始日, 終了日)
    code = args[1]


    z_startTmp = data2[:4] + "-" + data2[-4:-2] + "-" + data2[-2:]
    z_start = z_startTmp

    yestodaystrOfYear = data2[:4]
    z_end = yestodaystrOfYear + "-12-31"


    todaystrOfYear = data3[:4]
    #t_start = todaystrOfYear + "-01-01"
    t_startYn = int(yestodaystrOfYear)+1
    t_start = str(t_startYn) + "-01-01"

    today = datetime.date.today()    
    todaystr = today.strftime('%Y-%m-%d')
    
    t_endTmp = data3[:4] + "-" + data3[-4:-2] + "-" + data3[-2:]
    t_end = t_endTmp
    

    dataDF1 = code + "_" + z_start + "_" + z_end + ".csv"
    dataDF1 = dataDF1.replace('-', '')
    dataDF2 = code + "_" + t_start + "_" + t_end + ".csv"
    dataDF2 = dataDF2.replace('-', '')

    #print("データ１："+dataDF1)
    #print("データ２："+dataDF2)

    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")
    #/Users/yuanyue/Documents/GitHub/p1/data_to_df1.csv
    if os.path.exists(dataDF1) :
        print(dataDF1 + " Exists")
    else :
        print(dataDF1 + " Not Exists")
        print( "Now downloading......")
        data_to_df1 = get_stock(code, z_start, z_end)
        # データフレームの作成
        df1 = pd.DataFrame({'始値':data_to_df1[1], '終値':data_to_df1[2], '高値':data_to_df1[3], '安値':data_to_df1[4]}, index = data_to_df1[0])

        df1.to_csv(dataDF1)
        #df1.to_csv(dataDF1, columns=["始値", "終値", "高値", "安値"])
        #print(df1)
        print( "Download end")

    print("open end")

    df1 = pd.read_csv(dataDF1)
    #print(df1)
    #print("@@@@@@@@@@###############")
    df1 = df1.sort_values(by="Unnamed: 0", ascending=False)
    #print(df1)
    #print("@@@@@@@@@@###############2222222")
    #df1.to_csv("test009.csv",index=False, header=False, columns=["Unnamed: 0", "始値", "終値", "高値", "安値"])
    df1.to_csv(dataDF1,index=False, columns=["Unnamed: 0", "始値", "高値", "安値", "終値"])
    #df1.to_csv(dataDF1,index=False, columns=["Unnamed: 0", "始値", "終値", "高値", "安値"])

    #dfM1 = pd.read_csv("head_eng.csv")
    #print(dfM1)
    #dfM2 = pd.read_csv("test009.csv",index=False)
    #print(dfM2)
    #dfM1.append(dfM2)
    #print("@@@@@@@@@@###############3333333")
    #dfM1.to_csv("test008.csv")

    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    #/Users/yuanyue/Documents/GitHub/p1/data_to_df2.csv
    if os.path.exists(dataDF2) :
        print(dataDF2 + " Exists")
    else :
        print(dataDF2 + " Not Exists")
        print( "Now downloading......")
        data_to_df2 = get_stock(code, t_start, t_end)

        # データフレームの作成
        df2 = pd.DataFrame({'始値':data_to_df2[1], '終値':data_to_df2[2], '高値':data_to_df2[3], '安値':data_to_df2[4]}, index = data_to_df2[0])

        df2.sort_index(axis=0, ascending=False)

        df2.to_csv(dataDF2, columns=["始値", "終値", "高値", "安値"])
        print( "Download end")
    print("open end")



    df2 = pd.read_csv(dataDF2)
    #print(df2)
    #print("@@@@@@@@@@###############")
    df2 = df2.sort_values(by="Unnamed: 0", ascending=False)
    #print(df2)
    #print("@@@@@@@@@@###############2222222")
    df2.to_csv(dataDF2,index=False, columns=["Unnamed: 0", "始値", "高値", "安値", "終値"])
    #df2.to_csv(dataDF2,index=False, columns=["Unnamed: 0", "始値", "終値", "高値", "安値"])
    #print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")


    

    # CSVのロード(データ１：前年とデータ２：当年のデータ)
    data15 = np.genfromtxt(dataDF1, delimiter=",", skip_header=1, dtype='float')
    data16 = np.genfromtxt(dataDF2, delimiter=",", skip_header=1, dtype='float')

    #print(data15)
    #print("~~~~~~~~~~~~test~~~data15~~data16~~~~~~~~~~~~~~~~")
    #print(data16)

    # 終値の列だけを日付古い順に並び替えて取り出し
    f15, f16 = data15[:,4], data16[:,4]
    #f15, f16 = data15[:,3], data16[:,3]
    f15, f16 = f15[::-1], f16[::-1]
   
    #print(f15)
    #print("~~~~~~~~~~~~test~~~f15~~f16~~~~~~~~~~~~~~~~")
    #print(f16)
 
    # 移動平均線(25日線)の計算
    day = data4    # 日数
    data = np.r_[f15[len(f15)-day+1:len(f15)], f16]    #　前年の終値の一部と当年の終値を結合
    #print(data)
    ma_25d = move_average(data, day)

    

    # 移動平均線(75日線)の計算
    day = data5    # 日数
    data = np.r_[f15[len(f15)-day+1:len(f15)], f16]    #　前年の終値の一部と当年の終値を結合
    #print(data)
    ma_75d = move_average(data, day)
    

    #print("~~~~~~~~~~~~test~~~~~~~~~~~~~~~~~~~~~")
    

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

