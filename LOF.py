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

# Downloading data and reading CSV
urlretrieve('https://raw.githubusercontent.com/Aahna1909/data-Analysis-AQI/main/Sample.csv%20-%20Sheet1.csv', '2018.csv')
df = pd.read_csv('2018.csv')

# Removing unnecessary columns
df.drop(['StationId', 'NO', 'NOx', 'NH3', 'Benzene', 'Xylene', 'Toluene','Date'], axis=1, inplace=True)

# Removing rows with missing values
df = df.dropna()

# Mapping AQI_Bucket to numerical values
aqi_map = {'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4,'Severe': 5}
df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_map)

# Converting the dataframe to numpy array
data = df.values

# Splitting into input and output elements
X, y = data[:, :-1], data[:, -1]

# Splitting into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Identifying outliers in the training dataset
lof = LocalOutlierFactor()
yhat = lof.fit_predict(X_train)

# Selecting all rows that are not outliers
mask = yhat != -1
X_train_inliers, y_train_inliers = X_train[mask, :], y_train[mask]

# Selecting all rows that are outliers
X_train_outliers, y_train_outliers = X_train[yhat == -1, :], y_train[yhat == -1]

# Fitting the model using inliers only
model = LinearRegression()
model.fit(X_train_inliers, y_train_inliers)

# Evaluating the model
yhat = model.predict(X_test)

# Evaluating predictions
mae = mean_absolute_error(y_test, yhat)
print('MAE: %.3f' % mae)

# Plotting inliers and outliers using scatterplots
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 10))

# Scatter plot for PM2.5
axs[0, 0].scatter(X_train_inliers[:, 0], X_train_inliers[:, -1], color="k", s=3.0, label="Inliers")
axs[0, 0].scatter(X_train_outliers[:, 0], X_train_outliers[:, -1], color="r", s=5.0, label="Outliers")
axs[0, 0].set_xlabel("PM2.5")
axs[0, 0].set_ylabel("AQI_Bucket")
axs[0, 0].set_title("Identifying Outliers using Local Outlier Factor")

# Scatter plot for PM10
axs[0, 1].scatter(X_train_inliers[:, 1],  X_train_inliers[:, -1], color="k", s=3.0, label="Inliers")
axs[0, 1].scatter(X_train_outliers[:, 1], X_train_outliers[:, -1], color="r", s=5.0, label="Outliers")
axs[0, 1].set_xlabel("PM10")
axs[0, 1].set_ylabel("AQI_Bucket")
axs[0, 1].set_title("Identifying Outliers using Local Outlier Factor")

# Scatter plot for NO2

axs[0, 2].scatter(X_train_inliers[:, 2],  X_train_inliers[:, -1], color="k", s=3.0, label="Inliers")
axs[0, 2].scatter(X_train_outliers[:, 2], X_train_outliers[:, -1], color="r", s=5.0, label="Outliers")
axs[0, 2].set_xlabel("NO2")
axs[0, 2].set_ylabel("AQI_Bucket")
axs[0, 2].set_title("Identifying Outliers using Local Outlier Factor")
# Scatter plot for CO

axs[1, 0].scatter(X_train_inliers[:, 3],  X_train_inliers[:, -1], color="k", s=3.0, label="Inliers")
axs[1, 0].scatter(X_train_outliers[:, 3], X_train_outliers[:, -1], color="r", s=5.0, label="Outliers")
axs[1, 0].set_xlabel("CO")
axs[1, 0].set_ylabel("AQI_Bucket")
axs[1, 0].set_title("Identifying Outliers using Local Outlier Factor")

# Scatter plot for SO2

axs[1, 1].scatter(X_train_inliers[:, 4],  X_train_inliers[:, -1], color="k", s=3.0, label="Inliers")
axs[1, 1].scatter(X_train_outliers[:, 4], X_train_outliers[:, -1], color="r", s=5.0, label="Outliers")
axs[1, 1].set_xlabel("SO2")
axs[1, 1].set_ylabel("AQI_Bucket")
axs[1, 1].set_title("Identifying Outliers using Local Outlier Factor")
# Scatter plot for O3

axs[1, 2].scatter(X_train_inliers[:, 5],  X_train_inliers[:, -1], color="k", s=3.0, label="Inliers")
axs[1, 2].scatter(X_train_outliers[:, 5], X_train_outliers[:, -1], color="r", s=5.0, label="Outliers")
axs[1, 2].set_xlabel("O3")
axs[1, 2].set_ylabel("AQI_Bucket")
axs[1, 2].set_title("Identifying Outliers using Local Outlier Factor")