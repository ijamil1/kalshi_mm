import json
import pymysql
import yfinance as yf
from datetime import date, timedelta
import numpy as np
import pandas as pd

def get_mysql_credentials():
        with open("mysql_config.json", "r") as jsonfile:
            data = json.load(jsonfile)
        return (data['host'], data['user'], data['password'], data['database'])
    
def get_cursor():
    creds = get_mysql_credentials()
    dbinstance_endpoint = creds[0]
    db_username = creds[1]
    db_pw = creds[2]
    db_name = creds[3]
    connection = pymysql.connect(host = dbinstance_endpoint, user = db_username, password = db_pw, database = db_name, autocommit=True)
    cursor = connection.cursor()
    return cursor

def insert_ndx_return(last_dt, cursor_ = None):
     if cursor_:
        ndx_daily = yf.Ticker('^NDX').history(start=(last_dt+timedelta(days=1)).isoformat(), end = (date.today() + timedelta(days=1)).isoformat())
        ndx_daily = ndx_daily.reset_index()
        ndx_daily['Date'] = ndx_daily.apply(func = lambda x: x['Date'].date(), axis=1)
        ndx_daily['Open'] = ndx_daily.apply(func = lambda x: round(x['Open'], 2), axis=1)
        ndx_daily['Close'] = ndx_daily.apply(func = lambda x: round(x['Close'], 2), axis=1)
        ndx_return = (np.array(ndx_daily['Open'].iloc[1:]) - np.array(ndx_daily['Open'].iloc[:-1]))/np.array(ndx_daily['Open'].iloc[:-1])
        ndx_daily = ndx_daily.iloc[:-1]
        ndx_daily['return'] = pd.Series(data=ndx_return)

        ndx_tups = []

        for idx, row in ndx_daily.iterrows():
            ndx_tups.append((row['Date'], row['Open'], row['Close'], row['return']))
        
        cursor_.executemany('INSERT INTO ndx VALUES (%s, %s, %s, %s)', ndx_tups)

def insert_spx_return(last_dt, cursor_ = None):
     if cursor_:
        spx_daily = yf.Ticker('^SPX').history(start=(last_dt+timedelta(days=1)).isoformat(), end = (date.today() + timedelta(days=1)).isoformat())
        spx_daily = spx_daily.reset_index()
        spx_daily['Date'] = spx_daily.apply(func = lambda x: x['Date'].date(), axis=1)
        spx_daily['Open'] = spx_daily.apply(func = lambda x: round(x['Open'], 2), axis=1)
        spx_daily['Close'] = spx_daily.apply(func = lambda x: round(x['Close'], 2), axis=1)
        spx_return = (np.array(spx_daily['Open'].iloc[1:]) - np.array(spx_daily['Open'].iloc[:-1]))/np.array(spx_daily['Open'].iloc[:-1])
        spx_daily = spx_daily.iloc[:-1]
        spx_daily['return'] = pd.Series(data=spx_return)

        spx_tups = []

        for idx, row in spx_daily.iterrows():
            spx_tups.append((row['Date'], row['Open'], row['Close'], row['return']))
        
        cursor_.executemany('INSERT INTO spx VALUES (%s, %s, %s, %s)', spx_tups)


def handler(event, context):
    cursor = get_cursor()
    
    cursor.execute('select max(Date) from spx;')
    spx_last_dt = cursor.fetchall()[0][0]
    insert_spx_return(spx_last_dt, cursor)

    cursor.execute('select max(Date) from ndx;')
    ndx_last_dt = cursor.fetchall()[0][0]
    insert_ndx_return(ndx_last_dt, cursor)





