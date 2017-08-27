import pandas as pd


from pandas import DataFrame
df = DataFrame([[3, 4, 2, 8, 2, 3], [3, 5, 1, 9, 4, 2], [9, 3, 1, 6, 3, 3]], index=[2,3,1], columns=["HP", "Attack", "Defence", "Special Attack", "Special Defence", "Speed"])
print(df)
df = df.sort_index()
print(df)



