import pandas as pd
csv = pd.read_csv("hoge.csv")
data = csv[['Date','val1','val2']]
print(data)

print("--------------------------")
csv = pd.read_csv("hoge.csv").sort_values(['Date'])
data = csv[['Date','val1','val2']]
print(data)

print("--------------------------")

csv = pd.read_csv("hoge2.csv").sort_values(['Date']).reset_index(drop=True)
data = csv[['Date','val1','val2']]
print(data)

print("--------------------------")

csv = pd.read_csv("hoge2.csv").sort_values(['Date']).reset_index(drop=True)
#csv = pd.read_csv("hoge2.csv")
print(csv)

csv = csv.sort_index(axis=1, ascending=False)
print(csv)
print("@@@@@@@@@@@@@@@@")

data = csv[['Date','val1','val2']]
print(data)

csv = pd.read_csv("hoge2.csv")
print(csv)

#df = df.sort_values(by="Special Defence")
csv= csv.sort_values(by="Date", ascending=False)
print(csv)
print("@@@@@@@@@@@@@@@@")
data = csv[['Date','val1','val2']]
print(data)

csv.to_csv("hoge_test1.csv",index=False, columns=['Date','val1','val2'])


csv.to_csv("hoge_test2.csv",index=False, header=True, columns=['Date','val1','val2'])
csv.to_csv("hoge_test3.csv",index=False, header=False, columns=['Date','val1','val2'])

csv.to_csv("hoge_test4.csv",index=False, columns=['Date','val2','val1'])

