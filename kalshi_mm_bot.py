import numpy as np
import pymysql
from agent_state import State, Strategy
import pandas as pd
from scipy.stats import norm
import yfinance as yf
from datetime import datetime  
from datetime import date
from datetime import timedelta
import requests
import json
import  time
import logging
import ast


#snapshot schema: command id, seq num, market ticker, bids, asks, processed ts
#delta schema: command id, seq num, market_ticker, price, delta, side, processed ts


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

def run(data_date, ticker):
    cursor =  get_cursor()
    agent_state = State()
    strat = Strategy(agent_state)
    cursor.execute('select count(distinct(command_id)) from ob_snapshot where processed_ts like \'%{}%\' and market_ticker = \'{}\';'.format(data_date, ticker))
    num_cmd_ids = cursor.fetchall()[0][0]
    for cmd_id in range(1, num_cmd_ids+1):
        cursor.execute('select * from ob_snapshot where command_id = {} and processed_ts like \'%{}%\' and market_ticker = \'{}\''.format(cmd_id, data_date, ticker))
        snapshot_rows = cursor.fetchall()
        for row in snapshot_rows:
            bids = ast.literal_eval(row[3])
            asks = ast.literal_eval(row[4])
            agent_state.set_orderbook(bids, asks)
            agent_state.set_imbalance_indicators()
          
        cursor.execute('select * from ob_delta where command_id = {} and processed_ts like \'%{}%\' and market_ticker = \'{}\' order by seq_num ASC'.format(cmd_id, data_date, ticker))
        delta_rows = cursor.fetchall()
        for row in delta_rows:
            price = row[3]
            delta = row[4]
            side = row[5]
            processed_ts = row[6]
            agent_state.process_actions(datetime.strptime(processed_ts, '%Y-%m-%dT%H:%M:%S.%fZ'))
                        
            agent_state.adjust_orderbook(price, delta, side)
            
            agent_state.set_imbalance_indicators()
            strat.run_naive_strat(datetime.strptime(processed_ts, '%Y-%m-%dT%H:%M:%S.%fZ'))    
    print(agent_state.PnL*-1)    
    print(agent_state.calc_final_PnL())
    print(agent_state.yes_contracts_bought - agent_state.no_contracts_bought)
if __name__ == '__main__':
    run_date = '2024-01-17'
    market = 'NASDAQ100D-24JAN17-B16750'
    run(run_date, market)