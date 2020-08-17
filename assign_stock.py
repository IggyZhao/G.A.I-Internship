import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date
import time
from multiprocessing import Pool

from datetime import datetime as dt
import pymongo
import dns
import json
import random

import warnings
warnings.filterwarnings("ignore")
from cal_stat import cal_stat1, cal_stat2, cal_stat3

#------------------------------------- fetch data and store it in a shared holder -------------------------------------
# get 5-year close price data of S&P 500 ETF, Tesla, Netflix, Amazon, Apple
# we choose today as end_date, so the date range will naturally contain the last trading day
tickers = ['SPY','AAPL','FB','NFLX','MSFT']
start_date = '2015-05-01'
end_date = date.today()
stocks = yf.download(tickers,start_date,end_date)['Adj Close']

# store data into csv
stocks.to_csv('stock_data.csv')

#--------------------------------- read data from shared holder and put it into MongoDB --------------------------------
# read stored data
stocks = pd.read_csv('stock_data.csv',index_col = 0)
print(stocks.head())
# connect with MongoDB and get a database&collection
client = pymongo.MongoClient\
("mongodb+srv://Newuser:GLOBALAI@cluster0-ujbuf.mongodb.net/test?retryWrites=true&w=majority")
db = client['stock']
col = db.collection

# convert stock price to dict and put data in MongoDB
stocks_dict = stocks.to_dict('records')
col.insert_many(stocks_dict)

# #--------------------------------- Calculate statistics and speed up with parallel multi-processing ----------------------
if __name__ == '__main__':
    # without multi-processing, using function cal_stat1
    t1 = time.time()
    for i in range(100):
        stock_stat1 = cal_stat1(stocks)
    t2 = time.time()

    # without multi-processing, using function cal_stat2
    for i in range(100):
        res = []
        res.append(list(map(cal_stat2, (stocks[[i]] for i in tickers))))
        stock_stat2 = pd.concat((res[0][i] for i in range(len(tickers))), axis = 1, sort = False)
    t3 = time.time()

    # with multi-processing, using function cal_stat2
    p = Pool(4)
    res = []
    res.append(p.map(cal_stat2, (stocks[[i]] for i in tickers)))
    p.close()
    p.join()
    stock_stat3 = pd.concat((res[0][i] for i in range(len(tickers))), axis = 1, sort = False)
    t4 = time.time()

    # without multi-processing, using function cal_stat3
    res = []
    res.append(list(map(cal_stat3, (stocks[[i]] for i in tickers))))
    stock_stat4 = pd.concat((res[0][i] for i in range(len(tickers))), axis = 1, sort = False)
    t5 = time.time()

    # with multi-processing, using function cal_stat3
    p = Pool(4)
    res = []
    res.append(p.map(cal_stat3, (stocks[[i]] for i in tickers)))
    p.close()
    p.join()
    stock_stat5 = pd.concat((res[0][i] for i in range(len(tickers))), axis = 1, sort = False)
    t6 = time.time()
    print('\n')
    print('\n')
    print('function \t multi-processing \t  running time \n')
    print('cal_stat1 \t No \t\t\t  %.3fs on average' % float((t2-t1)/100) )
    print('cal_stat2 \t No \t\t\t  %.3fs on average' % float((t3-t2)/100) )
    print('cal_stat2 \t Yes \t\t\t  %.3fs' % float((t4-t3)) )
    print('cal_stat3 \t No \t\t\t  %.3fs' % float((t5-t4)) )
    print('cal_stat3 \t Yes \t\t\t  %.3fs' % float((t6-t5)) )