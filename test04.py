import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt

start="1949/5/16"
end="2016/9/30"#適当に入れ替えてください。

N225 = web.DataReader("NIKKEI225", 'fred',start,end)


print(N225.head(1))
print(type(N225))

print(N225)
