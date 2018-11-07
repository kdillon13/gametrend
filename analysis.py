
# coding: utf-8

"""
Exploratory data analysis, model building and validation on final data

First, merge games-features.csv and steamspy.csv into final data set.
Second, perform EDA on combined data set and compute new variables.
Third, create training and testing set split.
Fourth, tune model on training set and evaluate model using testing set.
Finally, export testing and training sets as .csv files to be used by
model_to_csv.py to pickle the model for deployment to website.

Created by: Kyle Dillon
Last updated: 10-7-18
"""

# In[529]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from sklearn.metrics import r2_score
import calendar
from datetime import datetime, timedelta
from dateutil import relativedelta
import re
from patsy import dmatrices
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_row', 1000)


# In[245]:


games=pd.read_csv('/Users/kdillon/Desktop/gametrend/games-features.csv')
steamspy=pd.read_csv('/Users/kdillon/Desktop/gametrend/steamspy.csv')


# In[3]:


steamspy.describe(include="all")


# In[246]:


games=games.drop_duplicates()
steamspy=steamspy.drop_duplicates()


# In[247]:


alldata=games.merge(steamspy, how='inner', left_on='QueryID', right_on='appid', left_index=True, validate='one_to_one')


# In[248]:


alldata.describe(include='all')


# In[249]:


releaseDates=pd.to_datetime(alldata['ReleaseDate'], errors='coerce')
alldata=alldata.assign(release=releaseDates)


# In[250]:


now=datetime.now()
age=((now-alldata['release']) / timedelta(days=30))+1
alldata=alldata.assign(age=age)
alldata=alldata.loc[alldata['age'] > 1]


# In[251]:


ownersLow=alldata['owners'].str.split(' .. ').str[0]
ownersHigh=alldata['owners'].str.split(' .. ').str[1]

for i in range(0,len(ownersLow)):
    ownersLow[ownersLow.keys()[i]] = int(ownersLow[ownersLow.keys()[i]].replace(',',''))

for i in range(0,len(ownersHigh)):
    ownersHigh[ownersHigh.keys()[i]] = int(ownersHigh[ownersHigh.keys()[i]].replace(',',''))


# In[252]:


alldata=alldata.assign(ownersL=ownersLow)
alldata=alldata.assign(ownersH=ownersHigh)

alldata['ownersL'] = alldata['ownersL'].astype(np.float64)
alldata['ownersH'] = alldata['ownersH'].astype(np.float64)

avgown=(alldata['ownersH']+alldata['ownersL'])/2
alldata = alldata.assign(ownavg=avgown)


# In[253]:


playerbase = alldata['ownavg'] / alldata['age']
alldata = alldata.assign(players=playerbase)


# In[254]:


lnplay = np.log(alldata['players'])
alldata = alldata.assign(lnplayers=lnplay)


# In[449]:


plt.figure()
alldata['lnplayers'].plot(kind='hist')
plt.show()


# In[255]:


alldata = alldata.loc[alldata['GenreIsNonGame'] == False]
alldata['initialprice'] = alldata['initialprice'] / 100


# In[256]:


posnegrev = alldata['positive'] / (alldata['negative']+1)
alldata = alldata.assign(recratio=posnegrev)

logrecratio = np.log(alldata['recratio']+1)
alldata = alldata.assign(lnrecratio=logrecratio)


# In[536]:


plt.figure()
np.log(alldata['median_forever']+1).plot(kind='hist')
plt.show()


# In[456]:


plt.figure()
np.log(alldata['average_forever']+1).plot(kind='hist')
plt.show()


# In[176]:


plt.figure()
np.log(alldata['ownavg']).plot(kind='hist')
plt.show()


# In[118]:


alldata.loc[alldata['devcount'] > 25].describe(include='all')


# In[21]:


devs = alldata.developer.unique()
len(devs)


# In[257]:


devcount = alldata['developer'].value_counts()


# In[258]:


alldata['devcount'] = alldata.groupby('developer')['developer'].transform('size')


# In[259]:


bdev = alldata.loc[alldata['devcount'] > 25]
bdev.describe()

alldata.loc[alldata['devcount'] > 25, 'bigdev'] = 1
alldata.loc[alldata['devcount'] <= 25, 'bigdev'] = 0


#alldata['bigdev'] = alldata.groupby('devcount')['devcount'].transform('size')


# In[260]:


alldata.loc[alldata['age'].between(0,12), 'years'] = 1
alldata.loc[alldata['age'].between(12,24), 'years'] = 2
alldata.loc[alldata['age'].between(24,36), 'years'] = 3
alldata.loc[alldata['age'].between(36,48), 'years'] = 4
alldata.loc[alldata['age'].between(48,60), 'years'] = 5
alldata.loc[alldata['age'].between(60,72), 'years'] = 6
alldata.loc[alldata['age'].between(72,84), 'years'] = 7
alldata.loc[alldata['age'].between(84,96), 'years'] = 8
alldata.loc[alldata['age'].between(96,108), 'years'] = 9
alldata.loc[alldata['age'].between(108,120), 'years'] = 10
alldata.loc[alldata['age'].between(120,132), 'years'] = 11
alldata.loc[alldata['age'].between(132,144), 'years'] = 12
alldata.loc[alldata['age'].between(144,156), 'years'] = 13
alldata.loc[alldata['age'].between(156,168), 'years'] = 14
alldata.loc[alldata['age'].between(168,180), 'years'] = 15
alldata.loc[alldata['age'].between(180,192), 'years'] = 16
alldata.loc[alldata['age'].between(192,204), 'years'] = 17
alldata.loc[alldata['age'].between(204,216), 'years'] = 18
alldata.loc[alldata['age'].between(216,228), 'years'] = 19
alldata.loc[alldata['age'].between(228,240), 'years'] = 20

alldata.describe()


# In[32]:


plt.figure()
alldata.groupby('developer').size().plot(kind='hist')
plt.show()

alldata['developer'].value_counts()
#bigdevs = alldata.loc[alldata['developer'].value_counts() > 1]


# In[515]:

logmedian = np.log(alldata['median_forever']+1)
alldata = alldata.assign(lnmedian=logmedian)



# In[514]:


predict = alldata[['QueryID',
                'ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'developer',
                'positive',
                'negative',
                'average_forever',
                'median_forever',
                'initialprice',
                'age',
                'ownavg',
                'players',
                'lnplayers',
                'ownersL',
                'recratio',
                'lnrecratio',
                'devcount',
                'bigdev',
                'years',
                'lnmedian']]


# In[273]:


predict = predict.loc[predict['age'] > 24]


# In[524]:


predict = predict.loc[predict['ownersL'] > 0]


# In[112]:


predict = predict.loc[predict['ownavg'] < 600000]


# In[505]:


predict = predict.loc[predict['median_forever'] > 0]


# In[75]:


predict = predict.loc[predict['age'] < 160]


# In[525]:


predict = predict.loc[predict['years'] < 14]


# In[149]:


predict = predict.loc[predict['devcount'] < 25]


# In[140]:


predict = predict.dropna(subset=['devcount', 'bigdev'])


# In[526]:


predict = predict.dropna(subset=['years'])


# In[163]:


predict['years'].notna


# In[171]:


predict.corr()


# In[415]:


predict.describe(include='all')


# In[289]:


ols = linear_model.LinearRegression();
ols.fit(predict[['ControllerSupport',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'IsFree',
                'initialprice',
                'years']], predict['lnplayers'])


# In[290]:


ols.score(predict[['ControllerSupport',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'IsFree',
                'initialprice',
                'years']], predict['lnplayers'])


# In[291]:


rreg = linear_model.Ridge();
rreg.fit(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice','years']], predict['lnplayers'])


# In[292]:


rreg.score(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice','years']], predict['lnplayers'])


# In[220]:


X_train, X_test, y_train, y_test = train_test_split(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',
                'years']], predict['lnplayers'], test_size=0.2)

#model = rreg.fit(X_train, y_train)
#preds = rreg.predict(X_test)

model2 = ols.fit(X_train, y_train)
preds2 = ols.predict(X_test)

#len(preds)
error=np.sqrt(np.mean((np.exp(preds2-y_test))**2))
error


# In[491]:


train = predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',]]

modelCoef = []
modelCoef.append(train.columns.values)
modelCoef.append(ols.coef_)

#for i in range(0,len(ols.coef_)):
#    print(train.columns.values[i], ols.coef_[i])


# In[470]:


ols.intercept_


# In[170]:


ols.coef_


# In[506]:


X_train, X_test, y_train, y_test = train_test_split(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',]], predict['lnplayers'], test_size=0.2)


# In[507]:


model = ols.fit(X_train,y_train)
preds = ols.predict(X_test)
model2 = rreg.fit(X_train, y_train)
preds2 = rreg.predict(X_test)


# In[503]:


## The line / model
plt.scatter(y_test, preds)
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()


# In[508]:


residper=(np.exp(preds)-np.exp(y_test))/np.exp(y_test)
np.mean(residper)*100


# In[509]:


model.score(X_test,y_test)


# In[497]:


np.mean((preds-y_test)**2)


# In[500]:


np.mean((preds2-y_test)**2)


# In[499]:


predcorr = predict.corr()


# In[500]:


predcorr[predcorr > .5]


# In[ ]:


preds-y_test


# In[186]:


ownavgtrim = alldata.loc[alldata['ownavg'] < 1000000]
ownavgtrim.describe()


# In[283]:


plt.scatter(ownavgtrim['age'], ownavgtrim['ownavg'], color='r')
plt.xlabel('Age of Game (months)')
plt.ylabel('Downloads')
plt.axis([0, 250, 0, 800000])
plt.plot(np.unique(ownavgtrim['age']), np.poly1d(np.polyfit(ownavgtrim['age'], ownavgtrim['ownavg'], 1))(np.unique(ownavgtrim['age'])), color='b')
#plt.show()
plt.savefig('timeanddownloads1.png', bbox_inches='tight')


# In[243]:


#plt.rcParams.update({'axes.titlesize': 6})
plt.rcParams.update({'axes.labelsize': 20})
plt.rcParams.update({'xtick.labelsize': 18})
plt.rcParams.update({'ytick.labelsize': 18})
plt.rcParams.update({'font.family': 'sans-serif'})


#plt.rc('font', size=40)          # controls default text sizes
#plt.rc('axes', titlesize=38)     # fontsize of the axes title
#plt.rc('axes', labelsize=14)    # fontsize of the x and y labels
#plt.rc('xtick', labelsize=16)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=16)    # fontsize of the tick labels
#plt.rc('legend', fontsize=14)    # legend fontsize
#plt.rc('figure', titlesize=22)  # fontsize of the figure title


# In[270]:


mediantrim = alldata.loc[alldata['median_forever'] < 100000]
mediantrim.describe()


# In[287]:


plt.scatter(mediantrim['age'], mediantrim['median_forever'], color='b')
plt.xlabel('Age of Game (months)')
plt.ylabel('Median Hours Played')
#plt.axis([0, 250, 0, 800000])
plt.plot(np.unique(mediantrim['age']), np.poly1d(np.polyfit(mediantrim['age'], mediantrim['median_forever'], 1))(np.unique(mediantrim['age'])), color='r')
plt.show()
#plt.savefig('timeandhoursplayed.png', bbox_inches='tight')


# In[123]:


plt.scatter(predict['initialprice'], predict['lnplayers'], color='b', alpha=.1)
plt.xlabel('Price')
plt.ylabel('Downloads')
#plt.yscale('log')
#plt.axis([0, 250, 0, 800000])
plt.plot(np.unique(predict['initialprice']), np.poly1d(np.polyfit(predict['initialprice'], predict['lnplayers'], 1))(np.unique(predict['initialprice'])), color='r')
plt.show()
#plt.savefig('timeandhoursplayed.png', bbox_inches='tight')


# In[80]:


plt.scatter(predict['age'], predict['ownavg'], color='b', alpha=.01)
plt.xlabel('Age of Game (months)')
plt.ylabel('Downloads')
#plt.yscale("log")
#plt.axis([0, 250, 0, 800000])
plt.plot(np.unique(predict['age']), np.poly1d(np.polyfit(predict['age'], predict['ownavg'], 1))(np.unique(predict['age'])), color='r')
plt.show()
#plt.savefig('timeandhoursplayed.png', bbox_inches='tight')


# In[523]:


plt.rcParams.update({'axes.titlesize': 20})
plt.rcParams.update({'axes.labelsize': 20})
plt.rcParams.update({'xtick.labelsize': 18})
plt.rcParams.update({'ytick.labelsize': 18})
plt.rcParams.update({'font.family': 'sans-serif'})

data_set1 = predict.loc[predict['years'] == 1]
data_set2 = predict.loc[predict['years'] == 2]
data_set3 = predict.loc[predict['years'] == 3]
data_set4 = predict.loc[predict['years'] == 4]
data_set5 = predict.loc[predict['years'] == 5]
data_set6 = predict.loc[predict['years'] == 6]
data_set7 = predict.loc[predict['years'] == 7]
data_set8 = predict.loc[predict['years'] == 8]
data_set9 = predict.loc[predict['years'] == 9]
data_set10 = predict.loc[predict['years'] == 10]
data_set11 = predict.loc[predict['years'] == 11]
data_set12 = predict.loc[predict['years'] == 12]
data_set13 = predict.loc[predict['years'] == 13]
data_set14 = predict.loc[predict['years'] == 14]
data_set15 = predict.loc[predict['years'] == 15]
data_set16 = predict.loc[predict['years'] == 16]
data_set17 = predict.loc[predict['years'] == 17]
data_set18 = predict.loc[predict['years'] == 18]
data_set19 = predict.loc[predict['years'] == 19]
data_set20 = predict.loc[predict['years'] == 20]

data = [data_set1['ownavg'].median(axis=0), 
        data_set2['ownavg'].median(axis=0), 
        data_set3['ownavg'].median(axis=0), 
        data_set4['ownavg'].median(axis=0), 
        data_set5['ownavg'].median(axis=0),
        data_set6['ownavg'].median(axis=0),
        data_set7['ownavg'].median(axis=0),
        data_set8['ownavg'].median(axis=0),
        data_set9['ownavg'].median(axis=0),
        data_set10['ownavg'].median(axis=0),
        data_set11['ownavg'].median(axis=0),
        data_set12['ownavg'].median(axis=0),
        data_set13['ownavg'].median(axis=0),
        data_set14['ownavg'].median(axis=0),
        data_set15['ownavg'].median(axis=0),
        data_set16['ownavg'].median(axis=0),
        data_set17['ownavg'].median(axis=0),
        data_set18['ownavg'].median(axis=0),
        data_set19['ownavg'].median(axis=0),
        data_set20['ownavg'].median(axis=0)]

data2 = [data_set1['ownavg'].mean(axis=0), 
        data_set2['ownavg'].mean(axis=0), 
        data_set3['ownavg'].mean(axis=0), 
        data_set4['ownavg'].mean(axis=0), 
        data_set5['ownavg'].mean(axis=0),
        data_set6['ownavg'].mean(axis=0),
        data_set7['ownavg'].mean(axis=0),
        data_set8['ownavg'].mean(axis=0),
        data_set9['ownavg'].mean(axis=0),
        data_set10['ownavg'].mean(axis=0),
        data_set11['ownavg'].mean(axis=0),
        data_set12['ownavg'].mean(axis=0),
        data_set13['ownavg'].mean(axis=0),
        data_set14['ownavg'].mean(axis=0),
        data_set15['ownavg'].mean(axis=0),
        data_set16['ownavg'].mean(axis=0),
        data_set17['ownavg'].mean(axis=0),
        data_set18['ownavg'].mean(axis=0),
        data_set19['ownavg'].mean(axis=0),
        data_set20['ownavg'].mean(axis=0)]

sem = [data_set1['ownavg'].sem(axis=0), 
        data_set2['ownavg'].sem(axis=0), 
        data_set3['ownavg'].sem(axis=0), 
        data_set4['ownavg'].sem(axis=0), 
        data_set5['ownavg'].sem(axis=0),
        data_set6['ownavg'].sem(axis=0),
        data_set7['ownavg'].sem(axis=0),
        data_set8['ownavg'].sem(axis=0),
        data_set9['ownavg'].sem(axis=0),
        data_set10['ownavg'].sem(axis=0),
        data_set11['ownavg'].sem(axis=0),
        data_set12['ownavg'].sem(axis=0),
        data_set13['ownavg'].sem(axis=0),
        data_set14['ownavg'].sem(axis=0),
        data_set15['ownavg'].sem(axis=0),
        data_set16['ownavg'].sem(axis=0),
        data_set17['ownavg'].sem(axis=0),
        data_set18['ownavg'].sem(axis=0),
        data_set19['ownavg'].sem(axis=0),
        data_set20['ownavg'].sem(axis=0)]

xyears = pd.DataFrame(data = {'year':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})
xticks = pd.DataFrame(data = {'tick':[2, 4, 6, 8, 10, 12, 14]})

#data_set1['ownavg'].describe()
plt.axis([0, 21, 10000, 100000000])
plt.xlabel('Age of Game (years)')
plt.ylabel('Downloads')
#plt.scatter(xyears['year'], data, color='r', alpha=1)
#plt.scatter(xyears['year'], data2, color='r', alpha=1, yerr=sem)
plt.errorbar(xyears['year'], data2, yerr=sem, fmt='x', color='k', ecolor='r')
#plt.boxplot(data, showfliers=False)
plt.yscale('log')
#plt.xticks(xticks['tick'], xticks['tick'])
plt.show()
#plt.savefig('ageanddownloads.png', bbox_inches='tight')


# In[736]:


#plt.rcParams.update({'axes.titlesize': 6})
plt.rcParams.update({'axes.labelsize': 20})
plt.rcParams.update({'xtick.labelsize': 18})
plt.rcParams.update({'ytick.labelsize': 18})
plt.rcParams.update({'font.family': 'sans-serif'})

data_set1 = predict.loc[predict['years'] == 1]
data_set2 = predict.loc[predict['years'] == 2]
data_set3 = predict.loc[predict['years'] == 3]
data_set4 = predict.loc[predict['years'] == 4]
data_set5 = predict.loc[predict['years'] == 5]
data_set6 = predict.loc[predict['years'] == 6]
data_set7 = predict.loc[predict['years'] == 7]
data_set8 = predict.loc[predict['years'] == 8]
data_set9 = predict.loc[predict['years'] == 9]
data_set10 = predict.loc[predict['years'] == 10]
data_set11 = predict.loc[predict['years'] == 11]
data_set12 = predict.loc[predict['years'] == 12]
data_set13 = predict.loc[predict['years'] == 13]
data_set14 = predict.loc[predict['years'] == 14]
data_set15 = predict.loc[predict['years'] == 15]
data_set16 = predict.loc[predict['years'] == 16]
data_set17 = predict.loc[predict['years'] == 17]
data_set18 = predict.loc[predict['years'] == 18]
data_set19 = predict.loc[predict['years'] == 19]
data_set20 = predict.loc[predict['years'] == 20]

dataM = [data_set1['median_forever'].mean(axis=0), 
        data_set2['median_forever'].mean(axis=0), 
        data_set3['median_forever'].mean(axis=0), 
        data_set4['median_forever'].mean(axis=0), 
        data_set5['median_forever'].mean(axis=0),
        data_set6['median_forever'].mean(axis=0),
        data_set7['median_forever'].mean(axis=0),
        data_set8['median_forever'].mean(axis=0),
        data_set9['median_forever'].mean(axis=0),
        data_set10['median_forever'].mean(axis=0),
        data_set11['median_forever'].mean(axis=0),
        data_set12['median_forever'].mean(axis=0),
        data_set13['median_forever'].mean(axis=0),
        data_set14['median_forever'].mean(axis=0),
        data_set15['median_forever'].mean(axis=0),
        data_set16['median_forever'].mean(axis=0),
        data_set17['median_forever'].mean(axis=0),
        data_set18['median_forever'].mean(axis=0),
        data_set19['median_forever'].mean(axis=0),
        data_set20['median_forever'].mean(axis=0)]

semM = [data_set1['median_forever'].sem(axis=0), 
        data_set2['median_forever'].sem(axis=0), 
        data_set3['median_forever'].sem(axis=0), 
        data_set4['median_forever'].sem(axis=0), 
        data_set5['median_forever'].sem(axis=0),
        data_set6['median_forever'].sem(axis=0),
        data_set7['median_forever'].sem(axis=0),
        data_set8['median_forever'].sem(axis=0),
        data_set9['median_forever'].sem(axis=0),
        data_set10['median_forever'].sem(axis=0),
        data_set11['median_forever'].sem(axis=0),
        data_set12['median_forever'].sem(axis=0),
        data_set13['median_forever'].sem(axis=0),
        data_set14['median_forever'].sem(axis=0),
        data_set15['median_forever'].sem(axis=0),
        data_set16['median_forever'].sem(axis=0),
        data_set17['median_forever'].sem(axis=0),
        data_set18['median_forever'].sem(axis=0),
        data_set19['median_forever'].sem(axis=0),
        data_set20['median_forever'].sem(axis=0)]

xyears = pd.DataFrame(data = {'year':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]})
xticks = pd.DataFrame(data = {'tick':[2, 4, 6, 8, 10, 12, 14]})

#data_set1['ownavg'].describe()
plt.axis([0, 21, 1, 10000])
plt.xlabel('Age of Game (years)')
plt.ylabel('Hours Played per Owner')
#plt.scatter(xyears['year'], data, color='b', alpha=1)
plt.errorbar(xyears['year'], dataM, yerr=semM, fmt='x', color='k', ecolor='b')
#plt.boxplot(data, showfliers=False)
plt.yscale('log')
#plt.xticks(xticks['tick'], xticks['tick'])
#plt.boxplot(data, showfliers=False)
#plt.yscale('log')
plt.show()
#plt.savefig('ageandtimeplayed.png', bbox_inches='tight')


# In[732]:

ols = linear_model.LinearRegression();
rreg = linear_model.Ridge();
ransac = linear_model.RANSACRegressor(min_samples=10, max_trials=10000000)

X_train, X_test, y_train, y_test = train_test_split(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',
                'years']], predict['lnplayers'], test_size=0.2)



model = ols.fit(X_train, y_train)
preds = ols.predict(X_test)

model2 = rreg.fit(X_train, y_train)
preds2 = rreg.predict(X_test)

model3 = ransac.fit(X_train, y_train)
preds3 = ransac.predict(X_test)

rmse=np.sqrt(np.mean((np.exp(preds)-np.exp(y_test))**2))
print(rmse)

rmse2=np.sqrt(np.mean((np.exp(preds2)-np.exp(y_test))**2))
print(rmse2)

rmse3=np.sqrt(np.mean((np.exp(preds3)-np.exp(y_test))**2))
print(rmse3)

print(ols.score(X_test, y_test))
print(rreg.score(X_test, y_test))
print(ransac.score(X_test, y_test))

plt.scatter(y_test, preds2, color='b', alpha=.4)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, preds2, 1))(np.unique(y_test)), color='r')
plt.axis([5, 14, 5, 14])
plt.title('Downloads')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
#plt.savefig('playersresid.png', bbox_inches='tight')


stats.probplot(y_train, dist="norm", plot=plt)
plt.show()


# In[737]:


model2.coef_


# In[746]:


columns=['Support Controller',
         'In-Game Purchases',
         'Support Mac OS',
         'Support Multiplayer',
         'Support Cooperative',
         'Support MMO',
         'Indie Genre',
         'Action Genre',
         'Casual Genre',
         'Strategy Genre',
         'RPG Genre',
         'Simulation Genre',
         'Sports Genre',
         'Racing Genre',
         'Initial Price',
         'Age of Game']

coefs2 = pd.DataFrame([model2.coef_], columns=columns)
abscoefs2 = pd.DataFrame([np.abs(model2.coef_)], columns=columns)
coefs2 = coefs2.append(abscoefs2, ignore_index=True)
coefs2 = coefs2.sort_values(by=1, ascending=True, axis=1)
coefs2.iloc[0].plot(kind='barh')
plt.axis([-.05, 1, 10.5, 15.5])
plt.title('Feature Contributions to Downloads')
#plt.xlabel('True Values')
#plt.ylabel('Predictions')
plt.show()
#plt.savefig('downloadsfeatcontrib5.png', bbox_inches='tight')


# In[752]:


X_train, X_test, y_train, y_test = train_test_split(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',
                'years']], predict['lnmedian'], test_size=0.2)


model5 = ols.fit(X_train, y_train)
preds5 = ols.predict(X_test)

model6 = rreg.fit(X_train, y_train)
preds6 = rreg.predict(X_test)

ransac = linear_model.RANSACRegressor(min_samples=10, max_trials=10000000)
model7 = ransac.fit(X_train, y_train)
preds7 = ransac.predict(X_test)

rmse5=np.sqrt(np.mean((np.exp(preds5)-np.exp(y_test))**2))
print(rmse5)

rmse6=np.sqrt(np.mean((np.exp(preds6)-np.exp(y_test))**2))
print(rmse6)

rmse7=np.sqrt(np.mean((np.exp(preds7)-np.exp(y_test))**2))
print(rmse7)

print(ols.score(X_test, y_test))
print(rreg.score(X_test, y_test))
print(ransac.score(X_test, y_test))

plt.scatter(y_test, preds6, color='b', alpha=.4)
plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, preds6, 1))(np.unique(y_test)), color='r')
plt.axis([0, 10, 0, 10])
plt.title('Median Hours Played')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.show()
#plt.savefig('hoursresid.png', bbox_inches='tight')


# In[753]:


model6.coef_


# In[758]:


#model6.coef_
#X_train.columns

#coef = sorted(model6.coef_, key=abs)
#print(np.abs(model6.coef_))
#model6.coef_

                
columns=['Support Controller',
         'In-Game Purchases',
         'Support Mac OS',
         'Support Multiplayer',
         'Support Cooperative',
         'Support MMO',
         'Indie Genre',
         'Action Genre',
         'Casual Genre',
         'Strategy Genre',
         'RPG Genre',
         'Simulation Genre',
         'Sports Genre',
         'Racing Genre',
         'Initial Price',
         'Age of Game']

coefs6 = pd.DataFrame([model6.coef_], columns=columns)
abscoefs6 = pd.DataFrame([np.abs(model6.coef_)], columns=columns)
coefs6 = coefs6.append(abscoefs6, ignore_index=True)
coefs6 = coefs6.sort_values(by=1, ascending=True, axis=1)
coefs6.iloc[0].plot(kind='barh')
plt.axis([-1.75, .5, 10.5, 15.5])
plt.title('Feature Contributions to Hours Played')
#plt.xlabel('True Values')
#plt.ylabel('Predictions')
plt.show()
#plt.savefig('hoursfeatcontrib5.png', bbox_inches='tight')
#coefs[coefs.iloc[0].argsort()]
#coefs.iloc[0].argsort()

#idx = np.abs(coefs.iloc[0]).argsort()
#idx
#coefs2 = coef[idx]
#coefs2
#coefs.sort_values(by=0, ascending=False, axis=1)
#df.sort_values(by=1, ascending=False, axis=1)
#coefs = sorted(coefs, key=abs)
#df.append(coefs)
#df
#df.append([model6.coef_], ignore_index=True)
#df = pd.DataFrame(model6.coef_.T, columns=X_train.columns)
#df.append(model6.coef_)
#df
#coefs = coefs.sort()
#coefs.plot(kind='barh')
#plt.show()
#coef


# In[347]:


X_train, X_test, y_train, y_test = train_test_split(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',
                'years']], predict['lnplayers'], test_size=0.2)

#model = rreg.fit(X_train, y_train)
#preds = rreg.predict(X_test)

model3 = ols.fit(X_train, y_train)
#preds3 = ols.predict(X_test)

ransac = linear_model.RANSACRegressor()
model4 = ransac.fit(X_train, y_train)
inlier_mask = ransac.inlier_mask_
outlier_mask = np.logical_not(inlier_mask)

print(ols.coef_)
print(ransac.estimator_.coef_)

#print(X_train[inlier_mask].shape)
#print(y_train[inlier_mask].shape)
#X_train[inlier_mask]
#print(len(X_train[outlier_mask]))
#print(len(y_train[outlier_mask]))

#plt.scatter(X_train[inlier_mask], y_train[inlier_mask], color='yellowgreen', marker='.', label='Inliers')
#plt.scatter(X_train[outlier_mask], y_train[outlier_mask], color='gold', marker='.', label='Outliers')
#plt.show()

#len(preds)
#error=np.sqrt(np.mean((np.exp(preds2-y_test))**2))
#error
#plt.scatter(y_test, preds2, color='b', alpha=.1)
#plt.plot(np.unique(y_test), np.poly1d(np.polyfit(y_test, preds3, 1))(np.unique(y_test)), color='r')
#plt.axis([5, 14, 5, 14])
#plt.xlabel('True Values')
#plt.ylabel('Predictions')
#plt.show()


# In[270]:


predict2 = predict[['ControllerSupport',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'IsFree',
                'initialprice',
                'years',
                'lnplayers']]

#for i in range(1,len(predict2)-1):
plt.scatter(predict2.iloc['ControllerSupport'], predict2['lnplayers'], color='b', alpha=.1)
#    data2.T.iloc[i].plot(kind='hist')
#    plt.show()



# In[268]:


predict2.describe(include='all')


# In[497]:


X_train, X_test, y_train, y_test = train_test_split(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',
                'years']], predict['lnplayers'], test_size=0.2)

ranfor = RandomForestRegressor(max_depth=4, n_estimators=100)
model4 = ranfor.fit(X_train, y_train)
preds4 = ranfor.predict(X_test)

rmse=np.sqrt(np.mean((np.exp(preds4)-np.exp(y_test))**2))
print(rmse)

print(ranfor.score(X_test, y_test))


# In[548]:


X_train, X_test, y_train, y_test = train_test_split(predict[['ControllerSupport',
                'IsFree',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'initialprice',
                'years']], predict['lnplayers'], test_size=0.2)



model = ols.fit(X_train, y_train)
preds = ols.predict(X_test)

model2 = rreg.fit(X_train, y_train)
preds2 = rreg.predict(X_test)

ransac = linear_model.RANSACRegressor()
model3 = ransac.fit(X_train, y_train)
preds3 = ransac.predict(X_test)

rmse=np.sqrt(np.mean((np.exp(preds)-np.exp(y_test))**2))
print(rmse)

rmse=np.sqrt(np.mean((np.exp(preds2)-np.exp(y_test))**2))
print(rmse2)

rmse=np.sqrt(np.mean((np.exp(preds3)-np.exp(y_test))**2))
print(rmse)

print(ols.score(X_test, y_test))
print(rreg.score(X_test, y_test))
print(ransac.score(X_test, y_test))

#rsquareddividend=r2_score(y_test, preds2)
#rsquareddividend

#vifdividend=1/(1-(rsquareddividend))
#print(vifdividend)


#X = X_train.assign(const=1)
cc = scipy.corrcoef(X_train, rowvar=False)
VIF = np.linalg.inv(cc)
VIF.diagonal()


# In[]:

modelcsv = X_train[['ControllerSupport',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'IsFree',
                'initialprice',
                'years']]

modelcsv = modelcsv.assign(lnplayers = y_train)
# In[]:

modelhours = X_train[['ControllerSupport',
                'PlatformMac',
                'CategoryMultiplayer',
                'CategoryCoop',
                'CategoryMMO',
                'GenreIsIndie',
                'GenreIsAction',
                'GenreIsCasual',
                'GenreIsStrategy',
                'GenreIsRPG',
                'GenreIsSimulation',
                'GenreIsSports',
                'GenreIsRacing',
                'IsFree',
                'initialprice',
                'years']]

modelhours = modelhours.assign(lnmedian = y_train)
modelhours
# In[]:

modelcsv.to_csv('modeltrain.csv', index=False)

# In[]:

modelhours.to_csv('modelhours.csv', index=False)


# In[]:

#X_train.head()
y_train.head()

