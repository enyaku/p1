#-*- coding:utf-8 -*-
#coding: UTF-8
import os
import sys
import numpy as np

import pandas as pd
import jsm
import datetime
import matplotlib.pyplot as plt
import matplotlib.finance as mpf
from matplotlib.dates import date2num


args = sys.argv


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


    print( args)
    print( u'銘柄コード（第１引数）：' + args[1])
    print( u'開始日　　（第２引数）：' + args[2])
    #print( u'終了日　　（第３引数）：' + args[3])

    start, end = 20170101, 20170821



    # 株価の取得(銘柄コード, 開始日, 終了日)
    code = args[1]
    today = datetime.date.today()

    #開始日をyyyy-mm-dd型に変換
    startstr = args[2]
    startstrTmp = startstr[:4] + "-" + startstr[-4:-2] + "-" + startstr[-2:]
    startstr = startstrTmp

    todaystr = today.strftime('%Y-%m-%d')

    dataDF = code + "_" + startstr + "_df_to_csv.csv"

    #/Users/yuanyue/Documents/GitHub/p1/3742_2017-01-01_df_to_csv.csv
    if os.path.exists(dataDF) :
        print("対象データが存在している。")
        print("対象データファイル名：「" + dataDF + "」である。 ")
        print("opening csv file......")
        # データフレームの作成
        df = pd.read_csv(dataDF)
    else :
        print("対象データが存在しない。")
        print( "Now downloading......")
        data = get_stock(code, startstr, todaystr)
        # データフレームの作成
        df = pd.DataFrame({'始値':data[1], '終値':data[2], '高値':data[3], '安値':data[4]}, index = data[0])

        df.to_csv(dataDF)
        print( "Download end")
        print("ダウンロードデータは「" + dataDF + "」で保存した。")
    print("open end")

    
    # グラフにプロット
    fig = plt.figure()
    ax = plt.subplot()
    mpf.candlestick2_ohlc(ax, df['始値'], df['高値'], df['安値'], df['終値'], width=0.5, colorup="r", colordown="g")

    #ax.set_xticklabels([(df.index[int(x)].strftime("%Y/%M/%D") if x <= df.shape[0] else x) for x in ax.get_xticks()], rotation=90)
   
    #ax.set_xlim([0, df.shape[0]]) # 横軸の範囲はデータの個数(df.shape[0]個)までに変更しておく

    ax.grid()
    ax.legend()
    fig.autofmt_xdate() #x軸のオートフォーマット
    
    
    plt.title("AI株投研究所-株価の機械学習-株価予測・分析")
    #plt.title('AI ', code,' TEST')
    #plt.title({'First line';'Second line'})

    plt.xlabel('日付')
    plt.ylabel('株価')
    #plt.text(60, .25, r'$\start=20170601,\ \end=20170822$')

    plt.show()
    
if __name__ == "__main__":
    main()

