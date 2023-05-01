import os
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import scipy.stats as st
import random
import math
import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

warnings.filterwarnings('ignore')
%matplotlib inline

# Downloading data and reading CSV
urlretrieve('https://raw.githubusercontent.com/Aahna1909/data-Analysis-AQI/main/Sample.csv%20-%20Sheet1.csv', '2018.csv')
df = pd.read_csv('2018.csv',index_col='Date')
df.index = pd.to_datetime(df.index)

# Removing unnecessary columns
df.drop(['StationId', 'NO', 'NOx', 'NH3', 'Benzene', 'Xylene', 'Toluene'], axis=1, inplace=True)

# Removing rows with missing values
df = df.dropna()

# Mapping AQI_Bucket to numerical values
aqi_map = {'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4,'Severe': 5}
df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_map)


x = df.values
plt.figure(figsize=(7,7))

plt.subplot(3,1,1)

plt.scatter(x[:,0],x[:,6])

df.info()
nbrs = NearestNeighbors(n_neighbors = 25)
nbrs.fit(x)
distances , indices = nbrs.kneighbors(x)
plt.subplot(3,3,1)

plt.plot(distances.mean(axis =1),color ="red")
plt.tight_layout()
threshold = np.percentile(df, 75)
abn_index = np.where(distances.mean(axis=1)>threshold)
plt.subplot(3,3,2)
plt.scatter(x[:,0],x[:,6])
plt.scatter(x[abn_index,0],x[abn_index,6],edgecolors='red')
plt.xlabel('PM2.5')
plt.ylabel('AQI')

plt.subplot(3,3,3)
plt.scatter(x[:,1],x[:,6])
plt.scatter(x[abn_index,1],x[abn_index,6],edgecolors='red')
plt.xlabel('PM10')
plt.ylabel('AQI')

plt.subplot(3,3,4)
plt.scatter(x[:,2],x[:,6])
plt.scatter(x[abn_index,2],x[abn_index,6],edgecolors='red')
plt.xlabel('NO2')
plt.ylabel('AQI')

plt.subplot(3,3,5)
plt.scatter(x[:,3],x[:,6])
plt.scatter(x[abn_index,3],x[abn_index,6],edgecolors='red')
plt.xlabel('CO')
plt.ylabel('AQI')

plt.subplot(3,3,6)
plt.scatter(x[:,4],x[:,6])
plt.scatter(x[abn_index,4],x[abn_index,6],edgecolors='red')
plt.xlabel('SO2')
plt.ylabel('AQI')

plt.subplot(3,3,7)
plt.scatter(x[:,5],x[:,6])
plt.scatter(x[abn_index,5],x[abn_index,6],edgecolors='red')
plt.xlabel('O3')
plt.ylabel('AQI')
plt.tight_layout()
