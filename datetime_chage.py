import datetime

from datetime import datetime as dt

#文字列から日付(datetime)
tstr = '2012-12-29 13:49:37'
tdatetime = dt.strptime(tstr, '%Y-%m-%d %H:%M:%S')

print(tdatetime)

#文字列から日付(date)
tstr = '2012-12-29 13:49:37'
tdatetime = datetime.datetime.strptime(tstr, '%Y-%m-%d %H:%M:%S')
tdate = datetime.date(tdatetime.year, tdatetime.month, tdatetime.day)
print(tdate)

#日付から文字列

tdatetime = dt.now()
tstr = tdatetime.strftime('%Y/%m/%d')
print(tstr)

