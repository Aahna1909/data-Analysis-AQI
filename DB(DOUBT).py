import os
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import numpy as np
import os
import pandas as pd
from sklearn.cluster import DBSCAN

from sklearn.neighbors import NearestNeighbors
import scipy.stats as st
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline

urlretrieve('https://raw.githubusercontent.com/Aahna1909/data-Analysis-AQI/main/Sample.csv%20-%20Sheet1.csv', '2018.csv')
df = pd.read_csv('2018.csv')
df.head()

df.drop(['StationId', 'NO', 'NOx', 'NH3', 'Benzene', 'Xylene', 'Toluene','Date'], axis=1, inplace=True)
df = df.dropna()

aqi_map = {'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4,'Severe': 5}
df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_map)

# apply DBSCAN to each column
for col in df.columns[:-1]:
    data = df[[col]].values
    
    # identify outliers using DBSCAN
    model = DBSCAN(eps=0.3, min_samples=10)
    yhat = model.fit_predict(data)
    
    # select all rows that are not outliers
    mask = yhat != -1
    data_inliers = data[mask, :]
    
    # select all rows that are outliers
    data_outliers = data[yhat == -1, :]
    
    # plot inliers and outliers using a scatterplot
    plt.scatter(data_inliers[:, 0], df.loc[mask, 'AQI'], color="k", s=3.0, label="Inliers")
    plt.scatter(data_outliers[:, 0], df.loc[yhat == -1, 'AQI'], color="r", s=5.0, label="Outliers")
    plt.xlabel(col)
    plt.ylabel("AQI_Bucket")
    plt.title("Identifying Outliers using DBSCAN for " + col)
    legend = plt.legend(loc="upper left")
    legend.legendHandles[0]._sizes = [10]
    legend.legendHandles[1]._sizes = [20]
    plt.show()
