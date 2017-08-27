
import pandas_datareader.data as web
import datetime
start = datetime.datetime(2010, 1, 1)
end = datetime.datetime(2015, 12, 31)
f = web.DataReader('SNE', 'fred', start, end)
print(f)

