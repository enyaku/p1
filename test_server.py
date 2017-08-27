
#!/usr/bin/env python
# -*- coding: utf-8 -*-
from spyre import server

import pandas as pd
pd.options.display.mpl_style = 'default'

# あらかじめデータを取得しておく
# 終値のみを取得し、一つのDataFrameに結合
import japandas as jpd
toyota = jpd.DataReader(7203, 'yahoojp', start='2015-01-01')[[u'終値']]
toyota.columns = [u'トヨタ']
honda = jpd.DataReader(7267, 'yahoojp', start='2015-01-01')[[u'終値']]
honda.columns = [u'ホンダ']
df = toyota.join(honda)


# spyre.server.App を継承したクラスを作る
class StockExample(server.App):
    title = u"株価のプロット"

    # 左側のペインに表示する UI 要素を辞書のリストで指定
    # ここではドロップダウン一つだけを表示
    inputs = [{"input_type":'dropdown', 
               # ドロップダウン自体の表示ラベル
               "label": 'Frequency',
               # ドロップダウンの選択項目を指定
               # label はドロップダウン項目の表示ラベル
               # value は各項目が選択された時にプログラム中で利用される値
               "options" : [ {"label": "月次", "value":"M"},
                             {"label": "週次", "value":"W"},
                             {"label": "日次", "value":"B"}],
               # 各 UI 要素の入力は各描画メソッド (getData, getPlot) に
               # 辞書型の引数 params として渡される
               # その辞書から値を取り出す際のキー名
               "variable_name": 'freq',
               "action_id": "update_data" }]

    # 画面を更新する設定
    controls = [{"control_type" : "hidden",
                 "label" : "update",
                 "control_id" : "update_data"}]

    # 描画するタブの表示ラベルを文字列のリストで指定
    tabs = [u"トヨタ", u"ホンダ", u"データ"]

    # tabs で指定したそれぞれのタブに描画する内容を辞書のリストで指定
    outputs = [{"output_type" : "plot", # matplotlib のプロットを描画する
                "output_id" : "toyota", # 描画するタブに固有の id
                "control_id" : "update_data",
                "tab" : u"トヨタ",       # 描画するタブの表示ラベル (tabs に含まれるもの)
                "on_page_load" : True },

               {"output_type" : "plot",
                "output_id" : "honda",
                "control_id" : "update_data",
                "tab" : u"ホンダ",
                "on_page_load" : True },

               {"output_type" : "table", # DataFrameを描画する
                "output_id" : "table_id",
                "control_id" : "update_data",
                "tab" : u"データ",
                "on_page_load" : True }]

    def getData(self, params):
        """
        output_type="table" を指定したタブを描画する際に呼ばれるメソッド
        DataFrameを返すこと

        params は UI 要素の入力 + いくつかのメタデータを含む辞書
        UI 要素の入力は inputs で指定した variable_name をキーとして行う
        """
        # ドロップダウンの値を取得
        # 値にはユーザの選択によって、options -> value で指定された M, W, B いずれかが入る
        freq = params['freq']

        # freq でグループ化し平均をとる
        tmp = df.groupby(pd.TimeGrouper(freq)).mean()
        return tmp

    def getPlot(self, params):
        """
        output_type="plot" を指定したタブを描画する際に呼ばれるメソッド
        matplotlib.Figureを返すこと
        """
        tmp = self.getData(params)

        # 同じ output_type で複数のタブを描画したい場合は、 params に含まれる
        # output_id で分岐させる
        # output_id は タブの表示ラベルではなく、outputs 中で指定した output_id
        if params['output_id'] == 'toyota':
            ax = tmp[[u'トヨタ']].plot(legend=False)
            return ax.get_figure()
        elif params['output_id'] == 'honda':
            ax = tmp[[u'ホンダ']].plot(legend=False)
            return ax.get_figure()
        else:
            raise ValueError


app = StockExample()
# port 9093 で Webサーバ + アプリを起動
app.launch(port=9093)

