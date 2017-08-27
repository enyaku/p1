import datetime
import pandas_datareader.data as web
import matplotlib.pyplot as plt

start="1949/5/16"
end="2016/9/30"#適当に入れ替えてください。
N225 = web.DataReader("NIKKEI225", 'fred',start,end)

#data = N225.get_all_data()

print(N225.head(1))
print(type(N225))

#print(data.head(1))


apple = web.DataReader("AAPL", 'google',start,end)

print(apple.head(1))
print(type(apple))


stock = web.DataReader(["NIKKEI225","DJIA"], 'fred',start,end)

stock.plot()

plt.show()


