# -*- coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
import sys


# 移動平均線の計算(データ, 日数)
def move_average(data, day):
    return np.convolve(data, np.ones(day)/float(day), 'valid')
    
def main():
    args = sys.argv

    print( args)

    print( u'銘柄コード（第１引数）：' + args[1])

    print( u'データ１　（第２引数）：' + args[2])

    print( u'データ２　（第３引数）：' + args[3])


    # CSVのロード(2015年と2016年のデータ)
    data15 = np.genfromtxt(args[2], delimiter=",", skip_header=1, dtype='float')

    print(data15)
    #data15 = data15.sort_index(ascending=False)
    #print(data15)

    data16 = np.genfromtxt(args[3], delimiter=",", skip_header=1, dtype='float')

    print(data16)
    #data16 = data16.sort_index(ascending=False)
    #print(data16)


    # 5列目の終値だけを日付古い順に並び替えて取り出し
    f15, f16 = data15[:,3], data16[:,3]
    f15, f16 = f15[::-1], f16[::-1]
    
    # 移動平均線(25日線)の計算
    day = 25    # 日数
    data = np.r_[f15[len(f15)-day+1:len(f15)], f16]    #　2015年の終値の一部と2016年の終値を結合
    ma_25d = move_average(data, day)

    # 移動平均線(75日線)の計算
    day = 75    # 日数
    data = np.r_[f15[len(f15)-day+1:len(f15)], f16]    #　2015年の終値の一部と2016年の終値を結合
    ma_75d = move_average(data, day)
    
    # グラフにプロット
    plt.plot(f16,  label="f")
    plt.plot(ma_25d, "--", color="r", label="MA 25d")
    plt.plot(ma_75d, "--", color="g", label="MA 75d")  
    
    #　ラベル軸
    plt.xlabel("Day")
    plt.ylabel("f")

    plt.title(args[1])

    # 凡例
    plt.legend(loc="4")
    # グリッド
    plt.grid()
    # グラフ表示
    plt.show()

    
if __name__ == "__main__":
    main()

