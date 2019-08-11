# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 19:59:49 2019


"""

import datetime as dt
import pandas as pd
import numpy as np
from matplotlib import style
import matplotlib.pyplot as plt
import pandas_datareader.data as web
style.use('ggplot')
start = dt.datetime(2015,1,1)
end = dt.datetime.now()
df = web.DataReader("TSLA" , 'yahoo',start ,end)


print(df.head())

df = pd.read_csv('tsla.csv', parse_dates = True, index_col=0)
print(df.head())
df.plot()
#plt.show()
df['Adj Close'].plot()
df['100ma'] = df['Adj Close'].rolling(window=100,min_periods=0).mean()
print(df.head(10))
df['100ma'] = df['Adj Close'].rolling(window=100,min_periods=0).mean()
print(df.head(10))

ax1 = plt.subplot2grid((6,1),(0,0), rowspan=5 , colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)
ax1.plot(df.index, df['Adj Close'])
ax1.plot(df.index, df['100ma'])
ax2.bar(df.index, df['Volume'])
plt.show()


from mpl_finance import candlestick_ohlc
import matplotlib.dates as mdates
df_ohlc = df['Adj Close'].resample('10D').ohlc()#resampling the given set
df_volume = df['Volume'].resample('10D').sum()
print(df_ohlc.head)

df_ohlc.reset_index(inplace=True)
df_ohlc['Date'] = df_ohlc['Date'].map(mdates.date2num)
ax1 = plt.subplot2grid((6,1),(0,0),rowspan=5,colspan=1)
ax2 = plt.subplot2grid((6,1),(5,0),rowspan=1,colspan=1,sharex=ax1)
ax1.xaxis_date() #to covert from raw mdates to dates
 
candlestick_ohlc(ax1, df_ohlc.values, width=5,colorup='g')
ax2.fill_between(df_volume.index.map(mdates.date2num), df_volume.values,0)
plt.show()

#scrapping all the 500  names
import bs4 as bs
import pickle
import requests
def sp500():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text)
    table = soup.find('table', {'class' : 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[1].text
        tickers.append(ticker)
        
    with open("sp500tickers.pickle","wb") as f:
        pickle.dump(tickers,f)
        print(tickers)
        return tickers
        
        
sp500()


import os #for directories

def get_data_from_yahoo(reload_sp500 = False):
    if reload_sp500:
        tickers = sp500()
    else:
        with open("sp500tickers.pickle","rb") as f:
            tickers = pickle.load(f)
            
    if not os.path.exists('stock_dfs'):
         os.makedirs('stock_dfs')
         
    start = dt.datetime(2000,12,31)
    end = dt.datetime(2015,1,31)
     
    for ticker in tickers[:250]:
         if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker,'yahoo',start,end)
            df.reset_index(inplace=True)
            df.set_index("Date",inplace=True)
           # df=df.drop("Symbol",axis=1)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))  
            
         else:
             print("Already have  {}".format(ticker))
             

get_data_from_yahoo()

 #compile_data():
def compile_data():
    with open("sp500tickers.pickle","rb") as f:
        tickers = pickle.load(f)
        
    main_df = pd.DataFrame()
    
    for count,ticker in enumerate(tickers):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date',inplace=True)
        df.rename(columns = {'Adj Close':ticker},inplace=True)
        df.drop(['Open','High','Low','Close','Volume'],1,inplace=True)
        if (count==180):
            break
        
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')
            
        if (count%10 == 0):
            print(count)
        
     
        
    print(main_df.head(10))
    main_df.to_csv('sp500_joined_closes.csv')
    

compile_data()    
def visualize_data():
    
    df = pd.read_csv('sp500_joined_closes.csv')
    df_corr = df.corr()
    print(df_corr.head())
    df_corr.to_csv('sp500corr.csv')
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()
    
visualize_data()    
  
#ml begins from here    
def process_data_for_labels(ticker):
    hm_days = 7
    df = pd.read_csv('sp500_joined_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)
    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
    df.fillna(0, inplace=True)
    return tickers, df


def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

#just above random equilibrium
    
from collections import Counter #for proper distribution of classes 

def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)

    df['{}_target'.format(ticker)] = list(map( buy_sell_hold,
                                               df['{}_1d'.format(ticker)],
                                               df['{}_2d'.format(ticker)],
                                               df['{}_3d'.format(ticker)],
                                               df['{}_4d'.format(ticker)],
                                               df['{}_5d'.format(ticker)],
                                               df['{}_6d'.format(ticker)],
                                               df['{}_7d'.format(ticker)]))

    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan)
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change()
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values
    return X, y, df





from sklearn import svm, cross_validation, neighbors
from sklearn.ensemble import VotingClassifier, RandomForestClassifier
def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)

    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25)

    clf = VotingClassifier([('lsvc', svm.LinearSVC()),
                            ('knn', neighbors.KNeighborsClassifier()),
                            ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print('accuracy:', confidence)
    predictions = clf.predict(X_test)
    print('predicted class counts:', Counter(predictions))
   
    return confidence

do_ml('AAP')
do_ml('MMM')
do_ml('ABT')

from statistics import mean

with open("sp500tickers.pickle","rb") as f:
    tickers = pickle.load(f)

accuracies = []
for count,ticker in enumerate(tickers):

    if count%10==0:
        print(count)

    accuracy = do_ml(ticker)
    accuracies.append(accuracy)
    print("{} accuracy: {}. Average accuracy:{}".format(ticker,accuracy,mean(accuracies)))








