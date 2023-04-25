import csv
import matplotlib.pyplot as plt
import seaborn as sns
from urllib.request  import urlretrieve
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline
urlretrieve('https://raw.githubusercontent.com/Aahna1909/data-Analysis-AQI/main/Sample.csv%20-%20Sheet1.csv','sample_2018.csv')
sample = pd.read_csv('sample_2018.csv')
print(sample)
df = sample
#'''print(type(sample))
#print(sample.info())
#print(sample.describe())
#sample.columns
#sample.shape
#sample['NH3'][215]'''

Av_PM10= sample.PM10.mean()
print(f"The average value of PM10 pollutant in the year 2018 is  {Av_PM10}. ")
standard_dev = sample.PM10.std()
standard_dev
AQI_severe = (sample.AQI_Bucket == 'Severe')
sample[AQI_severe]
sample_no =sample['Date'][sample['NO']>=100.00]
sample_no
#from IPython.display import display
#with pd.option_context('display.max_rows',50):
 #   display(sample[AQI_severe])
sample.sort_values(by ='PM10').head(10)
new_sample = sample.drop(columns=['Benzene','Toluene','Xylene','NO','NOx','NH3'])
new_sample.sort_values(by = ['PM10'])
new_sample.loc[1:50]
#modifying the value in the FIle : sample.at[index,'col_to_be_modified'] = ( sample.at[index,'col_from_modified']+ sample.at[index,'col_from_modified'])
new_sample.at[9,'PM2.5'] = (new_sample.at[8,'PM2.5']+new_sample.at[13,'PM2.5'])/2
new_sample.loc[1:10]
#working with dates
sample['Date'] = pd.to_datetime(sample.Date)
print(type(sample['Date']))
#segregating year , month,day and weekdays from the data
sample['year'] = pd.DatetimeIndex(sample.Date).year
sample['Month'] = pd.DatetimeIndex(sample.Date).month
sample['day'] = pd.DatetimeIndex(sample.Date).day
sample['weekday'] = pd.DatetimeIndex(sample.Date).weekday

sample
#queries for specific month /day
sample_may = sample[sample.Month==5]
sample_may
sample_subset = sample_may[sample_may.weekday == 6 ]
sample_subset

#extract the subset ofcolumns to be aggregated
sample_subset_2 = sample_subset[['StationId','Date','PM2.5','PM10','O3','NO2','SO2','CO','AQI','AQI_Bucket']]
sample_subset_2
#grouping and aggregation
monthly = sample.groupby('Month')
monthly[['StationId','Date','PM2.5','PM10','O3','NO2','SO2','CO','AQI','AQI_Bucket']].std()
weekly = sample.groupby('weekday')[['StationId','PM10','O3','NO2','SO2','CO','AQI','AQI_Bucket']]
weekly.std()
sample['Total_PM10'] = sample.PM10.cumsum()
sample
#cumsum = a , a+b, a+b+c, a+b+c+d , .....,a+b+c+d+e+..+n
#sample.set_index('Date',inplace = True)
#sample.PM10.plot()
#sample['PM2.5'].plot()
#sample.AQI.plot(title =" AQI_index")
#monthly.O3.plot(title = "monthly",kind = 'bar')
sns.set_style('ticks')
plt.plot(sample.StationId,sample.PM10,marker ="o",color ="orchid")
plt.plot(sample.StationId,sample.NO,marker ="x",color ="thistle",ls ="--",lw =3,ms =3,mec ="white",mew =2,alpha =0.5)
plt.xlabel("StationId")
plt.ylabel("Pollutants")
plt.title('sample_pollutants')
plt.legend(['PM10','NO'])
plt.figure(figsize =(10,10))

#filling missing values
#fillna 
sample_df = sample.fillna(method="ffill")
sample_df
#interpolate
#newer= new_sample.interpolate()
#newer.to_csv('Interpolation.csv')
#newer
#dealing with missing values - dropna
newest = sample.dropna()
newest.loc[1:500]
samp = sample.dropna(how="any")
samp
samp2 = sample.dropna(thresh =2)
samp2
#
new_sample.info()
#correlation between variables / correlation of columns
new_sample.corr()
print(new_sample.shape)
new_sample.head


#deal with missing values : drop , interpolate , linear regression, naive bayes , SVR, imputation(mean/median methods/most commonn /kNN/) 

min_threshold , max_threshold  =  sample.PM10.quantile([0.01,0.999])
min_threshold , max_threshold