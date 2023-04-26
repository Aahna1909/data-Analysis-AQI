#z-score: Z-Scores help identify outliers by values if a particular data point has a Z-score value either less than -3 or greater than +
import os
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request  import urlretrieve
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.stats as st
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

urlretrieve('https://raw.githubusercontent.com/Aahna1909/data-Analysis-AQI/main/Sample.csv%20-%20Sheet1.csv','2018.csv')
df = pd.read_csv('2018.csv')
df.head()

df.dtypes

df.drop(['StationId','NO','NOx','NH3','Benzene','Xylene','Toluene'],axis =1,inplace = True)
df =  df.dropna()
df['Date'] = pd.to_datetime(df.Date)
df.info()
df.describe()

# calculate z-scores and remove outliers
zr = np.abs(st.zscore(df['PM2.5']))
dfn = df[(zr > 3)]

zw = np.abs(st.zscore(df.PM10))
dfnn = df[(zw > 3)]

za = np.abs(st.zscore(df.NO2))
de = df[(za > 3)]

zb = np.abs(st.zscore(df.CO))
db = df[(zb > 3)]

zc = np.abs(st.zscore(df.SO2))
dc = df[(zc > 3)]

zd = np.abs(st.zscore(df.O3))
dd = df[(zd > 3)]

# plot distributions with and without outliers for each feature
plt.figure(figsize=(15, 25))

plt.subplot(6, 2, 1)
sns.distplot(df['PM2.5'])
plt.title('PM2.5 Distribution Before Removing Outliers')

plt.subplot(6, 2, 2)
sns.distplot(dfn['PM2.5'])
plt.title('PM2.5 Distribution After Removing Outliers')

plt.subplot(6, 2, 3)
sns.distplot(df['PM10'])
plt.title('PM10 Distribution Before Removing Outliers')

plt.subplot(6, 2, 4)
sns.distplot(dfnn['PM10'])
plt.title('PM10 Distribution After Removing Outliers')

plt.subplot(6, 2, 5)
sns.distplot(df['NO2'])
plt.title('NO2 Distribution Before Removing Outliers')

plt.subplot(6, 2, 6)
sns.distplot(de['NO2'])
plt.title('NO2 Distribution After Removing Outliers')

plt.subplot(6, 2, 7)
sns.distplot(df['CO'])
plt.title('CO Distribution Before Removing Outliers')

plt.subplot(6, 2, 8)
sns.distplot(db['CO'])
plt.title('CO Distribution After Removing Outliers')

plt.subplot(6, 2, 9)
sns.distplot(df['SO2'])
plt.title('SO2 Distribution Before Removing Outliers')

plt.subplot(6, 2, 10)
sns.distplot(dc['SO2'])
plt.title('SO2 Distribution After Removing Outliers')

plt.subplot(6, 2, 11)
sns.distplot(df['O3'])
plt.title('O3 Distribution Before Removing Outliers')

plt.subplot(6, 2, 12)
sns.distplot(dd['O3'])
plt.title('O3 Distribution After Removing Outliers')

plt.tight_layout()
