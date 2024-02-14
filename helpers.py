import numpy as np
import pymysql
import json
import bisect
import uuid

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

def get_kalshi_creds():
  with open("kalshi_creds.json", "r") as jsonfile:
    data = json.load(jsonfile)
    return (data['email'], data['password'])
    
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
    '''
    range_ticker_subtitles: list of 2-tup where first element is Kalshi ticker, second element is the ticker's subtitle
    ab_ticker_map: dict that maps lower bound to corresponding above/below ticker 
    '''
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
        d[ticker] = [ab_ticker_map[lb], ab_ticker_map[ub]] #maps range ticker to the 2 above/below tickers that would be needed to replicate the payoff this range ticker
    rev_d = {} #will map above/below ticker to any range ticker in which the current above/below ticker can be combined  with another above/below
                #ticker to replicate the range ticker's payoff
    for range_ticker, ab_ticker_list in d.items():
        long_ticker = ab_ticker_list[0]
        short_ticker = ab_ticker_list[1] 
        #longing long_ticker and shorting short_ticker would replicate payoff of current range_ticker
        
        if long_ticker in rev_d:
            if range_ticker not in rev_d[long_ticker]:
                rev_d[long_ticker].append(range_ticker)
        else:
            rev_d[long_ticker] = [range_ticker]
        
        if short_ticker in rev_d:
            if range_ticker not in rev_d[short_ticker]:
                rev_d[short_ticker].append(range_ticker)
        else:
            rev_d[short_ticker] = [range_ticker]
    return d, rev_d

def get_event_bid_sum(tickers, best_bids):
    '''
    tickers: list of Kalshi tickers that form a partition of an event (ie: complete coverage of the outcome space)
    best_bids: dict mapping ticker to best bid for that ticker
    '''
    sum = 0
    for t in tickers:
        sum += best_bids[t]
    return sum

def get_event_ask_sum(tickers, best_asks):
    '''
    tickers: list of Kalshi tickers that form a partition of an event (ie: complete coverage of the outcome space)
    best_asks: dict mapping ticker to best ask for that ticker
    '''
    sum = 0
    for t in tickers:
        sum += best_asks[t]
    return sum

def get_event_bid_vol(tickers, best_bids, bid_dict):
    min_vol = np.inf
    for t in tickers:
        bb  = best_bids[t]
        vol = bid_dict[t][bb]
        if vol < min_vol:
            min_vol = vol
    return min_vol

def get_event_ask_vol(tickers, best_asks, ask_dict):
    min_vol = np.inf
    for t in tickers:
        ba = best_asks[t]
        vol = ask_dict[t][ba]
        if vol < min_vol:
            min_vol = vol
    return min_vol 

def get_taker_fill(api_client, order_resp):
    '''
    api_client: Python object that serves as an interface to API
    order_resp: json object returned by  Kalshi api client's create order call

    return order_id for this order and the fill
    '''
    order_id = order_resp['order']['order_id']
    ord = api_client.get_order(order_id)
    return ord['order']['taker_fill_count']


def submit_market_order(api_client, ticker, side, vol):
    params =    {'ticker': ticker,
                'client_order_id': str(uuid.uuid4()),
                'type':'market',
                'action':'buy',
                'side': side,
                'count': vol,
                'yes_price': None,
                'no_price': None,
                'expiration_ts': 1,
                'sell_position_floor': None,
                'buy_max_cost': None}
    
    api_client.create_order(**params)
    return

def submit_limit_buy(api_client, ticker, side, vol, yes_price):
    params =    {'ticker': ticker,
                'client_order_id': str(uuid.uuid4()),
                'type':'limit',
                'action':'buy',
                'side': side,
                'count': vol,
                'yes_price': yes_price,
                'no_price': None,
                'expiration_ts': 1,
                'sell_position_floor': None,
                'buy_max_cost': None}

    resp = api_client.create_order(**params)
    return resp

def process_cross_event_arb_orders(api_client, x, x_vol, y, y_vol, z, z_vol, short_range_ind = True):
    '''
    if  short_range_ind => shorted x, longing y, shorted z
    else => long x, short y, long z 

    x, y, z: tickers
    x_vol, y_vol, z_vol: respective order volumes
    '''
    min_vol = min([x_vol, y_vol, z_vol])
    if x_vol != min_vol:
        diff = x_vol - min_vol
        if short_range_ind:
            #shorted x => bought No at the ask so now must buy yes at the ask
            submit_market_order(api_client, x, 'yes', diff)
        else:
            #longed x => bought yes at the ask so now must sell at the bid => buy No at the ask
            submit_market_order(api_client, x, 'no', diff)
    if y_vol != min_vol:
        diff = y_vol - min_vol
        if short_range_ind:
            #long y => bought Yes at the ask so now must sell at the bid => buy No at the ask
            submit_market_order(api_client, y, 'no', diff)
        else:
            #short y => bought No at the ask so now must buy yes at the ask 
            submit_market_order(api_client, x, 'yes', diff)
    if z_vol != min_vol:
        diff = z_vol - min_vol
        if short_range_ind:
            #short z => bought No at the ask so now must buy yes at the ask
            submit_market_order(api_client, z, 'yes', diff)
        else:
            #long z => bought yes at the ask so now must sell at the bid => buy No at the ask
            submit_market_order(api_client, z, 'no', diff)

