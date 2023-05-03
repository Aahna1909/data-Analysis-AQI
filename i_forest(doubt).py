import os
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request import urlretrieve
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')
%matplotlib inline

# Downloading data and reading CSV
urlretrieve('https://raw.githubusercontent.com/Aahna1909/data-Analysis-AQI/main/Sample.csv%20-%20Sheet1.csv', '2018.csv')
df = pd.read_csv('2018.csv', index_col='Date')
df.index = pd.to_datetime(df.index)

# Removing unnecessary columns
df.drop(['StationId', 'NO', 'NOx', 'NH3', 'Benzene', 'Xylene', 'Toluene'], axis=1, inplace=True)

# Removing rows with missing values
df = df.dropna()

# Mapping AQI_Bucket to numerical values
aqi_map = {'Good': 0, 'Satisfactory': 1, 'Moderate': 2, 'Poor': 3, 'Very Poor': 4,'Severe': 5}
df['AQI_Bucket'] = df['AQI_Bucket'].map(aqi_map)

# Splitting the data into training and test sets
X_train, X_test = train_test_split(df.values, train_size= 0.95, random_state=42)

# Scaling the data
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Fitting the Isolation Forest model
clf = IsolationForest(contamination= 0.5, random_state=42)
clf.fit(X_train_scaled)

# Predicting the outliers
predictions = clf.predict(X_test)

# Finding the indices of the abnormal points
abn_index = np.where(predictions == -1)
plt.figure(figsize=(12,12))
plt.subplot(3,2,1)
plt.scatter(X_train[:,0],X_train[:,6])
plt.scatter(X_test[abn_index,0],X_test[abn_index,6],edgecolors='red')
plt.xlabel('PM2.5')
plt.ylabel('AQI')

plt.subplot(3,2,2)
plt.scatter(X_train[:,1],X_train[:,6])
plt.scatter(X_test[abn_index,1],X_test[abn_index,6],edgecolors='red')
plt.xlabel('PM10')
plt.ylabel('AQI')

plt.subplot(3,2,3)
plt.scatter(X_train[:,2],X_train[:,6])
plt.scatter(X_test[abn_index,2],X_test[abn_index,6],edgecolors='red')
plt.xlabel('NO2')
plt.ylabel('AQI')

plt.subplot(3,2,4)
plt.scatter(X_train[:,3],X_train[:,6])
plt.scatter(X_test[abn_index,3],X_test[abn_index,6],edgecolors='red')
plt.xlabel('CO')
plt.ylabel('AQI')

plt.subplot(3,2,5)
plt.scatter(X_train[:,4],X_train[:,6])
plt.scatter(X_test[abn_index,4],X_test[abn_index,6],edgecolors='red')
plt.xlabel('SO2')
plt.ylabel('AQI')

plt.subplot(3,2,6)
plt.scatter(X_train[:,5],X_train[:,6])
plt.scatter(X_test[abn_index,5],X_test[abn_index,6],edgecolors='red')
plt.xlabel('O3')
plt.ylabel('AQI')
plt.tight_layout()

