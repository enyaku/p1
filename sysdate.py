
#!/usr/bin/python
# coding: UTF-8
 
# 現在の日付・時刻の取得と出力 | datetimeクラスの属性、today()、strftime()メソッドの使い方
 
import datetime # datetimeモジュールのインポート
import locale   # import文はどこに書いてもOK(可読性などの為、慣例でコードの始めの方)
 
 
# today()メソッドで現在日付・時刻のdatetime型データの変数を取得
d = datetime.datetime.today()
#   ↑モジュール名.クラス名.メソッド名
 
print( 'd == %s : %s\n' % (d, type(d))) # Microsecond(10^-6sec)まで取得
 
# datetime型の各属性へのアクセス
# year, month, day
print( '%s年%s月%s日\n' % (d.year, d.month, d.day))
 
# hour, minute, second, microsecond
print( '%s時%s分%s.%s秒n' % (d.hour, d.minute, d.second, d.microsecond))
 
# strftime()メソッドで日付時刻の書式を指定して出力
print( d.strftime("%Y-%m-%d %H:%M:%S"), '\n')
 
# 地域の設定
locale.setlocale(locale.LC_ALL, 'ja') # 属性の出力に影響(曜日とか)
print( locale.getlocale(), '\n')
 
print( d.strftime("%B%d日%A")) # locale.setlocale()でlocale指定してないと"Fri"と表示(デフォルトの設定地域)
print( d.strftime("%p"))       # 指定の地域でのAM/PMの対応する文字列
print( d.strftime("%x %X"))    #               日付と時刻に対応する文字列



