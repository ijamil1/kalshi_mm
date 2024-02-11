import numpy as np
import pymysql
import json
import bisect

month_to_abbr = {
    1: 'JAN',
    2: 'FEB',
    3: 'MAR',
    4: 'APR',
    5: 'MAY',
    6: 'JUN',
    7: 'JUL',
    8: 'AUG',
    9: 'SEP',
    10: 'OCT',
    11: 'NOV',
    12: 'DEC'
}

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

def calc_orderboook_imb(bids, asks, level=1):
    '''
    bids: ascending [price, qty] list
    asks: ascending [price, qty] list
    level: depth of orderbook for which to calculate order book imbalance
    '''
    bids_depth = len(bids)
    asks_depth = len(asks)
    if level > min([bids_depth, asks_depth]) or level == 0:
        if level <= 1:
            if bids_depth:
                return 1
            elif asks_depth:
                return -1
            else:
                return 0
        return calc_orderboook_imb(bids, asks, min([bids_depth, asks_depth]))
    else:
        bids_vol = 0
        asks_vol = 0
        for idx in range(level):
            bids_vol += bids[len(bids)-1-idx][1]
            asks_vol += asks[idx][1]
        return (bids_vol - asks_vol)/(bids_vol + asks_vol)

def get_best_bid(bids):
    '''
    bids: ascending [price, qty] list
    '''
    if len(bids):
        return bids[len(bids)-1][0]
    else:
        return 0

def get_best_ask(asks):
    '''
    asks: ascending [price, qty] list
    '''
    if len(asks):
        return asks[0][0]
    else:
        return np.inf


def get_bbv(bids):
    '''
    bids: ascending [price, qty] list
    return best bid volume
    '''
    if len(bids):
        return bids[len(bids)-1][1]
    else:
        return 0

def get_bav(asks):
    '''
    asks: ascending [price, qty] list
    return best ask volume
    '''
    if len(asks):
        return asks[0][1]
    else:
        return 0

def update_bids(bids, price, delta):
    '''
    bids: ascending [price, qty] list
    price: price level at which delta qty is being added to or removed
    delta: contracts being placed or removed from a price level
    '''
    bisect_left_ip = bisect.bisect_left(bids, price, key = lambda x: x[0])
    if bisect_left_ip == len(bids):
        #price is greater than all of the bids
        assert delta > 0
        bids.insert(bisect_left_ip, [price, delta])
    else:
        #price is greater than bids[:bisect_left_ip]
        if bids[bisect_left_ip][0] == price:
            #price exists in the current bids
            bids[bisect_left_ip][1]+=delta #update qty
            if bids[bisect_left_ip][1] <= 0:
                #remove price leveel
                del bids[bisect_left_ip]
        else:
            #price does not exist in the current bids
            assert delta > 0
            bids.insert(bisect_left_ip, [price, delta])

def update_asks(asks, price, delta):
    '''
    asks: ascending [price, qty] list
    price: price level at which delta qty is being added to or removed
    delta: contracts being placed or removed from a price level
    '''
    bisect_left_ip = bisect.bisect_left(asks, price, key = lambda x: x[0])
    if bisect_left_ip == len(asks):
        #price is greater than all of the asks
        assert delta > 0
        asks.insert(bisect_left_ip, [price, delta])
        
    else:
        #price is greater than asks[:bisect_left_ip]
        if asks[bisect_left_ip][0] == price:
            #price exists in the current bids
            asks[bisect_left_ip][1]+=delta #update qty
            if asks[bisect_left_ip][1] <= 0:
                #remove price leveel
                del asks[bisect_left_ip]
        else:
            #price does not exist in the current bids
            assert delta > 0
            asks.insert(bisect_left_ip, [price, delta])
    
            
def get_above_below_event_tickers(today):
    month = month_to_abbr[today.month]
    year = str(today.year - 2000)
    day = str(today.day) if today.day >= 10 else '0'+str(today.day)

    return 'INXU-' + year + month + day, 'NASDAQ100U-' + year + month + day

def get_range_event_tickers(today):
    month = month_to_abbr[today.month]
    year = str(today.year - 2000)
    day = str(today.day) if today.day >= 10 else '0'+str(today.day)

    return 'INX-' + year + month + day, 'NASDAQ100-' + year + month + day

def map_value_to_ab_ticker(ab_tickers):
    d = {}
    for ticker in ab_tickers:
        value = round(float(ticker[ticker.index('-T')+2:]))
        d[value] = ticker
    return d

def check_partition(l):
  logging.info('entering check_partition function')
  new = []
  for s in l:
    split = s.split(' ')
    if 'or' in s and 'below' in s:
      new.append([-np.inf, float(split[0].replace(',',''))])
    elif 'or' in s and 'above' in s:
      new.append([float(split[0].replace(',','')), np.inf])
    elif 'to' in s:
      new.append([float(split[0].replace(',','')), float(split[2].replace(',',''))])
    else:
      logging.info('unexpected format of market subtitle')
      return False
  new.sort(key = lambda x: x[0])
  prev_inc_ub = 0
  for i in range(len(new)):
    if i == 0:
      if new[i][0] != -1 * np.inf:
        logging.info('markets don\'t form a partition')
        return False
      prev_inc_ub = new[i][1]
    elif i == len(new)-1:
      if new[i][1] != np.inf:
        logging.info('markets don\'t form a partition')
        return False
      if np.ceil(prev_inc_ub) != new[i][0]:
        logging.info('markets don\'t form a partition')
        return False
    else:
      if np.ceil(prev_inc_ub) != new[i][0]:
        logging.info('markets don\'t form a partition')
        return False
      prev_inc_ub = new[i][1]
  logging.info('leaving check_partition function')
  return True


def map_range_ticker_to_ab_tickers(range_ticker_subtitles, ab_ticker_map):
    d = {}
    for tup in range_ticker_subtitles:
        ticker = tup[0]
        subtitle = tup[1]
        if 'below' in subtitle or 'above' in subtitle:
            continue
        split = subtitle.split(' to ')
        lb = round(float(split[0].replace(',','')))
        ub = round(float(split[1].replace(',', '')))
        if lb not in ab_ticker_map or ub not in ab_ticker_map:
            continue
        d[ticker] = [ab_ticker_map[lb], ab_ticker_map[ub]]
    rev_d = {}
    for range_ticker, ab_ticker_list in d.items():
        long_ticker = ab_ticker_list[0]
        short_ticker = ab_ticker_list[1]
        if long_ticker in rev_d:
            if range_ticker not in d[long_ticker]:
                rev_d[long_ticker].append(range_ticker)
        else:
            rev_d[long_ticker] = [range_ticker]
        if short_ticker in rev_d:
            if range_ticker not in d[short_ticker]:
                rev_d[short_ticker].append(range_ticker)
        else:
            rev_d[short_ticker] = [range_ticker]
    return d, rev_d