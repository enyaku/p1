
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

plt.style.use('ggplot')


df = pd.DataFrame({u'データ１': [0, 2, 4],
                   u'データ２': [1, 2, 3]})

print(df)


df.plot(y=[u'データ１', u'データ２'])
plt.xlabel(u'X軸')
plt.ylabel(u'Y軸')
plt.title(u'日本語・テスト')

plt.show()

