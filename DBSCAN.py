import os
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
import scipy.stats as st
import math
import warnings
from sklearn.ensemble import IsolationForest
import matplotlib.gridspec as gridspec
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.cluster import DBSCAN

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
db = DBSCAN(algorithm='auto', eps=0.28, leaf_size=30, metric='euclidean',metric_params=None, min_samples=200, n_jobs=None, p=None)
pred = db.fit_predict(x)

# Separating outliers and inliers
outliers = x[pred == -1]
inliers = x[pred != -1]

plt.figure(figsize=(12, 12))

# Plotting PM2.5 vs AQI
plt.subplot(3, 2, 1)
plt.scatter(inliers[:, 0], inliers[:, 6], alpha=0.5, label='inliers')
plt.scatter(outliers[:, 0], outliers[:, 6], color='red', alpha=0.5, label='outliers')
plt.xlabel('PM2.5')
plt.ylabel('AQI')
plt.legend()

# Plotting PM10 vs AQI
plt.subplot(3, 2, 2)
plt.scatter(inliers[:, 1], inliers[:, 6], alpha=0.5, label='inliers')
plt.scatter(outliers[:, 1], outliers[:, 6], color='red', alpha=0.5, label='outliers')
plt.xlabel('PM10')
plt.ylabel('AQI')
plt.legend()

# Plotting NO2 vs AQI
plt.subplot(3, 2, 3)
plt.scatter(inliers[:, 2], inliers[:, 6], alpha=0.5, label='inliers')
plt.scatter(outliers[:, 2], outliers[:, 6], color='red', alpha=0.5, label='outliers')
plt.xlabel('NO2')
plt.ylabel('AQI')
plt.legend()

# Plotting CO vs AQI
plt.subplot(3, 2, 4)
plt.scatter(inliers[:, 3], inliers[:, 6], alpha=0.5, label='inliers')
plt.scatter(outliers[:, 3], outliers[:,  6], color='red', alpha=0.5, label='outliers')
plt.xlabel('CO')
plt.ylabel('AQI')
plt.legend()

# Plotting SO2 vs AQI
plt.subplot(3, 2, 5)
plt.scatter(inliers[:, 4], inliers[:, 6], alpha=0.5, label='inliers')
plt.scatter(outliers[:, 4], outliers[:,  6], color='red', alpha=0.5, label='outliers')
plt.xlabel('SO2')
plt.ylabel('AQI')
plt.legend()

# Plotting O3 vs AQI
plt.subplot(3, 2, 6)
plt.scatter(inliers[:, 5], inliers[:, 6], alpha=0.5, label='inliers')
plt.scatter(outliers[:, 5], outliers[:,  6], color='red', alpha=0.5, label='outliers')
plt.xlabel('O3')
plt.ylabel('AQI')
plt.legend()
