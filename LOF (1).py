#LOF 
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
import warnings
from sklearn.neighbors import LocalOutlierFactor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')
%matplotlib inline

urlretrieve('https://raw.githubusercontent.com/Aahna1909/data-Analysis-AQI/main/Sample.csv%20-%20Sheet1.csv', '2018.csv')
df = pd.read_csv('2018.csv')
df.head()

df.dtypes

df.drop(['StationId', 'NO', 'NOx', 'NH3', 'Benzene', 'Xylene', 'Toluene','Date'], axis=1, inplace=True)
df = df.dropna()
aqi_map = {'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4,'Severe': 5}
df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_map)



df.info()
df.sample(10)
d2 = df.values #converting the df into numpy array
lof = LocalOutlierFactor(n_neighbors=20, contamination='auto')
good = lof.fit_predict(d2) == 1
plt.figure(figsize=(10,5))
plt.scatter(d2[good, 1], d2[good, 0], s=2, label="Inliers", color="#4CAF50")
plt.scatter(d2[~good, 1], d2[~good, 0], s=8, label="Outliers", color="#F44336")
plt.title('Outlier Detection using Local Outlier Factor', fontsize=20)
plt.legend (fontsize=15, title_fontsize=15)