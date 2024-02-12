'''IDEA: eliminate any arb opportunities

  If we are dealing with an event that has n mutually exclusive markets that cover the entire outcome space, a buy side arb is possible. It would appear if the sum of the prices of the lowest asks across
  the n markets is less than 100 cents. Let's say that this sum is some value x. Then, for each set of n contracts that you buy where each contract is for a distinct one of the n markets, the profit is 100 - x.
  If there are m such sets. Then, the profit is m * (100 - x). And, the limiting factor is m which will be the minimum of the size of the lowest asks across the n markets.

   If wew are dealing with an event that has n mutually exclusive markets that cover the entire outcome space, a sell side arb is possible. It would appear if the sum of best bids across markets is > 1, then sell at the market across all markets the same number of contracts per markets to net (# of contracts * (sum of best bids -  1)) profit by buying
   no contracts at the market across all markets. Again, the limiting factor is the set of contracts for which you can do this for which is the minimum size of the highest bids across the n markets.

  fees = ceil(.035 * num_contracts * price_in_usd * (1-price_in_usd))
  max of x*(1-x) for x in [0,1] = 0.25  so ceil (.035 * num_contracts * .25) = .00875 * num_contracts
  ie: max fees of .01$/contract
   consider fees and optimize speed and pattern of logging calls and other logic
'''


from audioop import reverse
from pytz import timezone
from KalshiClientsBaseV2 import KalshiClient, HttpError, ExchangeClient
import requests
import  uuid
import json
import logging
from datetime import date, datetime as dt
from urllib3.exceptions import HTTPError
from dateutil import parser
from datetime import datetime
from datetime import timedelta
import asyncio
import websockets
import numpy as np
from helpers import *

def process_buy_order(order_response):
  order_id = order_response['order']['order_id']
  ticker = order_response['order']['ticker']
  cursor = None
  count = 0
  while True:
    response = exchange_client.get_fills(ticker,order_id, cursor)
    if 'fills' not in response:
      break
    fills = response['fills']
    if len(fills)>0:
      side = fills[0]['side']
      buy_price = 0
      if side == 'yes':
        buy_price = fills[0]['yes_price']
      else:
        buy_price = fills[0]['no_price']
      for fill in fills:
        count+=fill['count']
    cursor = response['cursor']
    if cursor is None or cursor == '':
      break
  if count:
    logging.info('Limit Buy: ticker: %s, side: %s, buy_price: %s, count: %s', ticker, side, str(buy_price/100), str(count))
  return count

def process_sell_order(order_response):
  order_id = order_response['order']['order_id']
  ticker = order_response['order']['ticker']
  cursor = None
  count = 0
  price = 0
  wt_sum = 0
  while True:
    response = exchange_client.get_fills(ticker,order_id, cursor)
    if 'fills' not in response:
      break
    fills = response['fills']
    if len(fills)>0:
      side = fills[0]['side']
      for fill in fills:
        side = fill['side']
        cur_count = fill['count']
        if side == 'yes':
          price = fill['yes_price']
        else:
          price = fill['no_price']
        wt_sum += price * cur_count
        count+=cur_count
    cursor = response['cursor']
    if cursor is None or cursor == '':
       break
  if count:
    logging.info('Market Sell: ticker: %s, side: %s, avg_price: %s, count: %s', ticker, side, str(wt_sum/(100*count)), str(count))

def compute_event_bid_ask_sums(event):
  bid_sum = 0
  ask_sum = 0
  if event == 'sp500':
    for ticker in sp_market_tickers:
      bid_sum += best_bids[ticker]
      ask_sum += best_asks[ticker]
    event_bid_sum['sp500'] = bid_sum
    event_ask_sum['sp500'] = ask_sum
  elif event == 'nasdaq':
    for ticker in nasdaq_market_tickers:
      bid_sum += best_bids[ticker]
      ask_sum += best_asks[ticker]
    event_bid_sum['nasdaq'] = bid_sum
    event_ask_sum['nasdaq'] = ask_sum
        
def check_buy_arb(ticker_list):
  nas = False
  sp = False
  offer_sum = 0
  if 'NASDAQ' in ticker_list[0]:
    nas = True
    sp = False
    offer_sum = event_ask_sum['nasdaq']
  elif 'INXD' in ticker_list[0]:
    sp = True
    nas = False
    offer_sum = event_ask_sum['sp500']

  min_offer_size = np.inf
  ticker_price_dict = {}
  ticker_filled_dict = {}
  ticker_size_tup_list = []

  for k in ticker_list:
    ticker_price_dict[k] = best_asks[k]
    min_offer_size = min([best_ask_sizes[k],min_offer_size])
    ticker_size_tup_list.append((k,best_ask_sizes[k]))

  if min_offer_size == 0:
    return
  min_offer_size*=position_cap
  if nas:
    if (nasdaq_negative_delta_ask_amt >= 0.5*order_params['count'] and (datetime
        .now()-nasdaq_negative_delta_ask_dt).seconds <= 60) or ((datetime.now()-nasdaq_negative_delta_ask_dt).seconds<= 10):
      return
  elif sp:
    if (sp_negative_delta_ask_amt >= 0.5*order_params['count'] and (datetime.now()-sp_negative_delta_ask_dt).seconds <= 60) or ((datetime.now()-sp_negative_delta_ask_dt).seconds <= 10):
      return
  order_params['side'] = 'yes'
  if (min_offer_size * offer_sum * 100) <= max_amt_bankroll_allocated:
    order_params['count'] = min_offer_size
  else:
    order_params['count'] = int(np.floor(max_amt_bankroll_allocated/(100*offer_sum)))
  ticker_size_tup_list.sort(key=lambda x: x[1])
  for cur_ticker_tup in ticker_size_tup_list:
    cur_ticker = cur_ticker_tup[0]
    order_params['client_order_id']= str(uuid.uuid4())
    order_params['ticker'] = cur_ticker
    order_params['yes_price'] = ticker_price_dict[cur_ticker]*100
    order_params['no_price'] = None
    order_params['buy_max_cost'] = ticker_price_dict[cur_ticker]*100
    filled = process_buy_order(exchange_client.create_order(**order_params))
    ticker_filled_dict[cur_ticker] = filled
    if filled != order_params['count']:
      prev_tickers = ticker_list[:ticker_list.index(cur_ticker)]
      for old_ticker in prev_tickers:
        #market sell yes contracts
        order_params['client_order_id']= str(uuid.uuid4())
        order_params['action'] = 'sell'
        order_params['type'] = 'market'
        order_params['ticker'] = old_ticker
        order_params['count'] = ticker_filled_dict[old_ticker] - filled
        ticker_filled_dict[old_ticker] = filled
        order_params['yes_price'] = None
        order_params['no_price'] = None
        order_params['buy_max_cost'] = None
        process_sell_order(exchange_client.create_order(**order_params))
      order_params['type'] = 'limit'
      order_params['action'] = 'buy'
      order_params['count'] = filled
      if filled == 0:
          break
   
def check_sell_arb(ticker_list):
    sp = False
    nas = False
    bid_sum = 0
    no_sum = 0
    if 'NASDAQ' in ticker_list[0]:
      nas = True
      bid_sum = event_bid_sum['nasdaq']
      no_sum = len(nasdaq_market_tickers) - bid_sum
    elif 'INXD' in ticker_list[0]:
      sp = True
      bid_sum = event_bid_sum['sp500']
      no_sum = len(sp_market_tickers) - bid_sum
    
    
    #sum of buy no = sum 1-bid(ith market) = num markets - bid_sum
    min_bid_size = np.inf
    ticker_price_dict = {}
    ticker_filled_dict = {}
    ticker_size_tup_list = []
    for k in ticker_list:
      if best_bids[k] == 0:
        continue
      ticker_price_dict[k] = 1-best_bids[k]
      min_bid_size = min([best_bid_sizes[k], min_bid_size])
      ticker_size_tup_list.append((k, best_bid_sizes[k]))

    if min_bid_size == 0:
      return
    min_bid_size*=position_cap
    if nas:
      if (nasdaq_negative_delta_bid_amt >= 0.5 * order_params['count'] and (datetime.now()-nasdaq_negative_delta_bid_dt).seconds <= 60) or ((datetime.now()-nasdaq_negative_delta_bid_dt).seconds <= 10):
        return
    elif sp:
      if (sp_negative_delta_bid_amt >= 0.5 * order_params['count'] and (datetime.now()-nasdaq_negative_delta_bid_dt).seconds <= 60) or ((datetime.now()-sp_negative_delta_bid_dt).seconds <= 10):
        return
    order_params['side'] = 'no'
    if min_bid_size * no_sum * 100 <= max_amt_bankroll_allocated:
      order_params['count'] = min_bid_size
    else:
      order_params['count'] = int(np.floor(max_amt_bankroll_allocated/(100*no_sum)))
    ticker_size_tup_list.sort(key=lambda x:x[1])
    for cur_ticker_tup in ticker_size_tup_list:
      cur_ticker = cur_ticker_tup[0]
      order_params['ticker'] = cur_ticker
      order_params['yes_price'] = None
      order_params['no_price'] = ticker_price_dict[cur_ticker]*100
      order_params['buy_max_cost'] = ticker_price_dict[cur_ticker]*100
      filled = process_buy_order(exchange_client.create_order(**order_params))
      ticker_filled_dict[cur_ticker] = filled
      if filled != order_params['count']:
        prev_tickers = ticker_list[:ticker_list.index(cur_ticker)]
        for old_ticker in prev_tickers:
          #market sell no contracts
          order_params['client_order_id']= str(uuid.uuid4())
          order_params['action'] = 'sell'
          order_params['type'] = 'market'
          order_params['ticker'] = old_ticker
          order_params['count'] = ticker_filled_dict[old_ticker] - filled
          ticker_filled_dict[old_ticker] = filled
          order_params['yes_price'] = None
          order_params['no_price'] = None
          order_params['buy_max_cost'] = None
          process_sell_order(exchange_client.create_order(**order_params))
        order_params['type'] = 'limit'
        order_params['action'] = 'buy'
        order_params['count'] = filled
        if filled == 0:
          break

def check_range_arbs(ndx_tickers, spx_tickers):
  '''
  ndx_tickers: list of NASDAQ100 range tickers that form a partition of the event
  spx_tickers: list of SP500 range tickers that form a partition of the event 
  '''
  bid_sum = get_event_bid_sum(ndx_tickers, bb_dict)
  if bid_sum > 1.025:
    #NDX sell arb possible 
    min_vol = get_event_bid_vol(ndx_tickers, bb_dict, bids)
    pass

  bid_sum = get_event_bid_sum(spx_tickers, bb_dict)
  if bid_sum > 1.025:
    #SPX sell arb possible
    min_vol = get_event_bid_vol(spx_tickers, bb_dict, bids)
    pass

  ask_sum = get_event_ask_sum(ndx_tickers, ba_dict)
  if ask_sum < 0.975:
    #NDX buy arb possible
    min_vol = get_event_ask_vol(ndx_tickers, ba_dict, asks)
    pass

  ask_sum = get_event_ask_sum(spx_tickers, ba_dict)
  if ask_sum < 0.975:
    #spx buy arb possible
    min_vol = get_event_ask_vol(spx_tickers, ba_dict, asks)
    pass
  
def check_cross_event_arbs(ndx_range_tickers, spx_range_tickers):
  for ticker in ndx_range_tickers:
    '''cross event arb for NASDAQ100 events'''
    if ticker not in ndx_range_ticker_to_ab_tickers:
      continue
    ab_tickers = ndx_range_ticker_to_ab_tickers[ticker]
    lb_ticker = ab_tickers[0]
    ub_ticker = ab_tickers[1]
    if bb_dict[ticker] > ba_dict[lb_ticker] - bb_dict[ub_ticker]:
      # proceeds from selling to best bid for the range ticker > total cost of (buying long ticker above/below ticker and shorting short ticker above/below ticker)
      #need to figure out min volume across the 2 shorts and 1 long
      min_vol = min([bids[ticker][bb_dict[ticker]], bids[ub_ticker][bb_dict[ub_ticker]], asks[lb_ticker][ba_dict[lb_ticker]]])
    elif ba_dict[ticker] < bb_dict[lb_ticker] - ba_dict[ub_ticker]:
      #proceeds from shorting (selling lb ab ticker and longing ub ab ticker) > cost of longing range ticker
      #need to figure out min volume across the 1 short and 2 longs
      min_vol = min([asks[ticker][ba_dict[ticker]], bids[lb_ticker][bb_dict[lb_ticker]], asks[ub_ticker][ba_dict[ub_ticker]]])
  
  for ticker in spx_range_tickers:
    '''cross event arb for SP500 events'''
    if ticker not in spx_range_ticker_to_ab_tickers:
      continue
    ab_tickers = spx_range_ticker_to_ab_tickers[ticker]
    lb_ticker = ab_tickers[0]
    ub_ticker = ab_tickers[1]
    if bb_dict[ticker] > ba_dict[lb_ticker] - bb_dict[ub_ticker]:
      # proceeds from selling to best bid for the range ticker > total cost of (buying long ticker above/below ticker and shorting short ticker above/below ticker)
      #need to figure out min volume across the 2 shorts and 1 long
      min_vol = min([bids[ticker][bb_dict[ticker]], bids[ub_ticker][bb_dict[ub_ticker]], asks[lb_ticker][ba_dict[lb_ticker]]])
    elif ba_dict[ticker] < bb_dict[lb_ticker] - ba_dict[ub_ticker]:
      #proceeds from shorting (selling lb ab ticker and longing ub ab ticker) > cost of longing range ticker
      #need to figure out min volume across the 1 short and 2 longs
      min_vol = min([asks[ticker][ba_dict[ticker]], bids[lb_ticker][bb_dict[lb_ticker]], asks[ub_ticker][ba_dict[ub_ticker]]])


def handle_orderbook_snapshot(ticker, response):
  cur_market_ticker = ticker
  resp = response
  reset()

  yes_bids = None
  no_bids = None
  if 'yes' in resp['msg']:
    yes_bids = resp['msg']['yes']
  if 'no' in resp['msg']:
    no_bids = resp['msg']['no']
  #loop thru both yes bids and offers. If bid is higher than current largest bid, update current largest bid and current largest bid's size. Regardless of it's higher than or lower than current largest bid, add
  # current bid and current size to the current ticker's orderbook. If offer is lower than current lowest offer, update current lowest offer and current lowest offer's size. Regardless of it's lower
  #or higher than current lowest offer, add current offer and size to the current ticker's orderbook
  if yes_bids is not None:
    for bid_qty in yes_bids:
      bid = round(bid_qty[0]/100, 2)
      bids[cur_market_ticker][bid] = bid_qty[1]
    
    bb = max(bids[cur_market_ticker].keys())
    bb_dict[cur_market_ticker] = bb
  
  if no_bids is not None:
    for bid_qty in no_bids:
      ask = round((100-bid_qty[0])/100,2)
      asks[cur_market_ticker][ask] = bid_qty[1]

    ba = min(asks[cur_market_ticker].keys())
    ba_dict[cur_market_ticker] = ba
  '''#check for arb
  if 'NASDAQ' in cur_market_ticker:
    #nasdaq
    event_ask_sum['nasdaq']+=ask_price_delta
    event_bid_sum['nasdaq']+=bid_price_delta
    if event_ask_sum['nasdaq'] < 1 - len(nasdaq_market_tickers):
      check_buy_arb(nasdaq_market_tickers)
      print('here')
      bankroll = exchange_client.get_balance()['balance']
    elif event_bid_sum['nasdaq'] > 1 + len(nasdaq_market_tickers):
      check_sell_arb(nasdaq_market_tickers)
      print('here')
      bankroll = exchange_client.get_balance()['balance']
  elif 'INXD' in cur_market_ticker:
    #sp500
    event_ask_sum['sp500']+=ask_price_delta
    event_bid_sum['sp500']+=bid_price_delta
    if event_ask_sum['sp500'] < 1 - len(sp_market_tickers):
      check_buy_arb(sp_market_tickers)
      print('here')
      bankroll = exchange_client.get_balance()['balance']
    elif event_bid_sum['sp500'] > 1 + len(sp_market_tickers):
      check_sell_arb(sp_market_tickers)
      print('here')
      bankroll = exchange_client.get_balance()['balance']
      '''

def handle_yes_orderbook_delta(ticker, response):
    cur_market_ticker = ticker
    resp = response
    price = round(resp['msg']['price']/100, 2)
    delta = resp['msg']['delta']
    if delta == 0:
      return
    
    if price not in bids[cur_market_ticker]:
      #bid is not in existing order book
      assert delta > 0
      bids[cur_market_ticker][price] = delta
      if price > bb_dict[cur_market_ticker]:
        bb_dict[cur_market_ticker] = price
    else:
      #bid is in existing order book
      bids[cur_market_ticker][price] += delta
      if bids[cur_market_ticker][price] <= 0:
        del bids[cur_market_ticker][price]
        if price == bb_dict[cur_market_ticker]:
          bb_dict[cur_market_ticker] = max(bids[cur_market_ticker].keys()) if len(bids[cur_market_ticker]) else 0

    ''' if price > best_bids[cur_market_ticker][0]:
      #new highest bid
      if delta < 0:
        logging.error('tried to create new best bid with a negative delta')
        raise Exception

      best_bids[cur_market_ticker] = [price, delta]
      best_bid_sizes[cur_market_ticker] = delta
      #check arb
      if 'NASDAQ' in cur_market_ticker:
        event_bid_sum['nasdaq'] += price_delta
        if event_bid_sum['nasdaq'] > 1 + len(nasdaq_market_tickers):
          check_sell_arb(nasdaq_market_tickers)
          print('here')
          bankroll = exchange_client.get_balance()['balance']
      elif 'INXD' in cur_market_ticker:
        event_bid_sum['sp500'] += price_delta
        if event_bid_sum['sp500'] > 1 + len(sp_market_tickers):
          check_sell_arb(sp_market_tickers)
          print('here')
          bankroll = exchange_client.get_balance()['balance']
    elif price == best_bids[cur_market_ticker]:
      #affects highest bid for this market
      if delta < 0:
        if abs(delta) > ob[cur_market_ticker]['bids'][price]:
          logging.error('negative delta larger than availability at a price level')
          raise Exception
        if 'NASDAQ' in cur_market_ticker:
          nasdaq_negative_delta_bid_amt = abs(delta)
          nasdaq_negative_delta_bid_dt = datetime.now()
        elif 'INXD' in cur_market_ticker:
          sp_negative_delta_bid_amt = abs(delta)
          sp_negative_delta_bid_dt = datetime.now()
      ob[cur_market_ticker]['bids'][price] += delta
      best_bid_sizes[cur_market_ticker] += delta
      if ob[cur_market_ticker]['bids'][price] == 0:
        cur_best_bid = price
        del ob[cur_market_ticker]['bids'][price]
        #recompute best bid
        if len(ob[cur_market_ticker]['bids'].keys()):
          best_bids[cur_market_ticker] = max(ob[cur_market_ticker]['bids'].keys())
          best_bid_sizes[cur_market_ticker] = ob[cur_market_ticker]['bids'][best_bids[cur_market_ticker]]
          price_delta = best_bids[cur_market_ticker] - price
        else:
          best_bids[cur_market_ticker] = 0
          best_bid_sizes[cur_market_ticker] = 0
          price_delta = -price
        if 'NASDAQ' in cur_market_ticker:
          event_bid_sum['nasdaq']+=price_delta
        elif 'INXD' in cur_market_ticker:
          event_bid_sum['sp500']+=price_delta
    else: #doesnt affect highest bid for this market
      if price in ob[cur_market_ticker]['bids']:
        #bid price is already in OB so delta can be pos or neg
        #if neg, then it's possible that best bid will change
        if delta < 0 and abs(delta) > ob[cur_market_ticker]['bids'][price]:
          logging.error('negative delta larger than availability at a price level')
          raise Exception
        ob[cur_market_ticker]['bids'][price] += delta
        if ob[cur_market_ticker]['bids'][price] == 0:
          #delta was negative and removes this price level
          del ob[cur_market_ticker]['bids'][price]
      else: #price was not in OB
        if delta < 0:
          logging.error('new bid that wasn\'t in the orderbook has a negative delta')
          raise Exception
        ob[cur_market_ticker]['bids'][price] = delta'''

def handle_no_orderbook_delta(ticker, response):
    #side is no
    cur_market_ticker = ticker
    resp = response
    price = (100 - resp['msg']['price'])/100
    delta = resp['msg']['delta']
    if delta == 0:
      return
    
    if price not in asks[cur_market_ticker]:
      #ask is not in existing order book
      assert delta > 0
      asks[cur_market_ticker][price] = delta
      if price < ba_dict[cur_market_ticker]:
        ba_dict[cur_market_ticker] = price
    else:
      #ask is in existing order book
      asks[cur_market_ticker][price] += delta
      if asks[cur_market_ticker][price] <= 0:
        del asks[cur_market_ticker][price]
        if price == ba_dict[cur_market_ticker]:
          ba_dict[cur_market_ticker] = min(asks[cur_market_ticker].keys()) if len(asks[cur_market_ticker]) else 1  

    '''if price < best_asks[cur_market_ticker]:
      if delta < 0:
        logging.error('new best ask has a negative delta')
        raise Exception
      ob[cur_market_ticker]['asks'][price] = delta
      price_delta = price-best_asks[cur_market_ticker]
      best_asks[cur_market_ticker] = price
      best_ask_sizes[cur_market_ticker] = delta
      if 'NASDAQ' in cur_market_ticker:
        event_ask_sum['nasdaq']+=price_delta
        if event_ask_sum['nasdaq'] < 1 - len(nasdaq_market_tickers):
          check_buy_arb(nasdaq_market_tickers)
          print('here')
          bankroll = exchange_client.get_balance()['balance']
      elif 'INXD' in cur_market_ticker:
        event_ask_sum['sp500'] += price_delta
        if event_ask_sum['sp500'] < 1 - len(sp_market_tickers):
          check_buy_arb(sp_market_tickers)
          print('here')
          bankroll = exchange_client.get_balance()['balance']
    elif price == best_asks[cur_market_ticker]:
      #affects lowest ask for this market
      if delta < 0:
        if abs(delta) > ob[cur_market_ticker]['asks'][price]:
          logging.error('negative delta larger than availability at a price level')
          raise Exception
        if 'NASDAQ' in cur_market_ticker:
          nasdaq_negative_delta_ask_amt = abs(delta)
          nasdaq_negative_delta_ask_dt = datetime.now()
        elif 'INXD' in cur_market_ticker:
          sp_negative_delta_ask_amt = abs(delta)
          sp_negative_delta_ask_dt = datetime.now()
      ob[cur_market_ticker]['asks'][price] += delta
      best_ask_sizes[cur_market_ticker] += delta
      if ob[cur_market_ticker]['asks'][price] == 0:
        del ob[cur_market_ticker]['asks'][price]
        #recompute best ask
        if len(ob[cur_market_ticker]['asks'].keys()):
          best_asks[cur_market_ticker] = min(ob[cur_market_ticker]['asks'].keys())
          best_ask_sizes[cur_market_ticker] = ob[cur_market_ticker]['asks'][best_asks[cur_market_ticker]]
          price_delta = best_asks[cur_market_ticker]-price
        else:
          best_asks[cur_market_ticker] = 1
          best_ask_sizes[cur_market_ticker] = 0
          price_delta = 1-price
        if 'NASDAQ' in cur_market_ticker:
          event_ask_sum['nasdaq']+=price_delta
        elif 'INXD' in cur_market_ticker:
          event_ask_sum['sp500']+=price_delta
    else: #doesnt affect lowest ask for this market
      if price in ob[cur_market_ticker]['asks']:
        #ask price is already in OB so delta can be pos or neg
        #if neg, then it's possible that best offer will change
        if delta < 0 and abs(delta) > ob[cur_market_ticker]['asks'][price]:
          logging.error('negative delta larger than availability at a price level')
          raise Exception
        ob[cur_market_ticker]['asks'][price] += delta
        if ob[cur_market_ticker]['asks'][price] == 0:
          #delta was negative and removes this price level
          del ob[cur_market_ticker]['asks'][price]
      else: #price was not in OB
          if delta < 0:
            logging.error('new ask has a negative delta')
          ob[cur_market_ticker]['asks'][price] = delta'''

def reset():
  '''
  clears bids and asks dicts and resets bb and ba for each ticker to 0, 1 respectively with 0 volume for each
  '''
  bids.clear()
  asks.clear()
  bb_dict.clear()
  ba_dict.clear()
  for ticker in markets:    
    bids[ticker] = {0: 0}
    asks[ticker] = {1: 0}
    bb_dict[ticker] = 0
    ba_dict[ticker] = 1

async def get_data(market_tickers):
  command_id = 1
  breakout = False
  async for ws in websockets.connect(uri='wss://trading-api.kalshi.com/trade-api/ws/v2', extra_headers={'Authorization': 'Bearer {}'.format(token)}):
    if breakout:
      break
    bankroll = exchange_client.get_balance()['balance'] #current bankroll in cents
    reset()
    prev_seq_num = 0
    try:
      d = {"id": command_id,"cmd": "subscribe","params": {"channels": ["orderbook_delta", "market_lifecycle"], "market_tickers": market_tickers}}
      await ws.send(json.dumps(d))  #subscribe to channels after establishing new ws connection
      while True:
        if bankroll <= 0.5 * starting_bankroll:
          #break if we've lost or wagered half our starting bankroll for the day
          breakout = True
          break
        if datetime.now(tz=eastern).hour < 9 or (datetime.now(tz=eastern).hour == 16 and datetime.now(tz=eastern).minute >= 30) or datetime.now(tz=eastern).hour > 16:
          #break if before 9 am ET or later than 4:30 ET
          breakout = True
          break
        msg = await ws.recv()
        resp = json.loads(msg)
        if resp['type'] == 'orderbook_snapshot':
          if resp['seq'] - 1 != prev_seq_num:
            logging.error('orderbook channel message received out of order based on sequence number')
            break
          prev_seq_num = resp['seq']
          cur_market_ticker = resp['msg']['market_ticker']
          handle_orderbook_snapshot(cur_market_ticker, resp)
        elif resp['type'] == 'orderbook_delta':
          if resp['seq'] - 1 != prev_seq_num:
            logging.error('orderbook channel message received out of order based on sequence number')
            break
          prev_seq_num = resp['seq']
          cur_market_ticker = resp['msg']['market_ticker']
          if resp['msg']['side'] == 'yes':
            handle_yes_orderbook_delta(cur_market_ticker, resp)
          else:
            handle_no_orderbook_delta(cur_market_ticker, resp)
        elif resp['type'] == 'market_lifecycle':
          cur_market_ticker = resp['msg']['market_ticker']
          if 'NASDAQ' in cur_market_ticker and cur_market_ticker not in ndx_ab_markets and cur_market_ticker not in ndx_range_markets:
            logging.error('new NASDAQ market added that is not accounted for')
            breakout = True
            break
          elif 'INXD' in cur_market_ticker and cur_market_ticker not in spx_ab_markets and cur_market_ticker not in spx_range_markets:
            logging.error('new SP500 market added that not accounted for')
            breakout = True      
            break   
    except websockets.ConnectionClosed:
      command_id+=1
      logging.error('websocket error')
    except Exception as e:
      logging.error("error thrown in get_data function", exc_info=True)
      breakout = True
      command_id+=1



today = datetime.today().date()
spx_range_ticker, ndx_range_ticker = get_range_event_tickers(today)
spx_ab_ticker, ndx_ab_ticker = get_above_below_event_tickers(today)

eastern = timezone('US/Eastern')
order_params = {'ticker':None,
                     'client_order_id':None,
                     'type':'limit',
                     'action':'buy',
                     'side': None,
                     'count':None,
                     'yes_price':None,
                     'no_price':None,
                     'expiration_ts':1,
                     'sell_position_floor':0,
                     'buy_max_cost':None}

bids = {} #ticker: {price: qty}
asks = {} #ticker: {price: qty}
bb_dict = {} #ticker: best bid price
ba_dict = {} #ticker: best ask price

sp_negative_delta_ask_dt = datetime.fromtimestamp(0)
sp_negative_delta_ask_amt = 0
nasdaq_negative_delta_ask_dt = datetime.fromtimestamp(0)
nasdaq_negative_delta_ask_amt = 0
sp_negative_delta_bid_dt = datetime.fromtimestamp(0)
sp_negative_delta_bid_amt = 0
nasdaq_negative_delta_bid_dt = datetime.fromtimestamp(0)
nasdaq_negative_delta_bid_amt = 0
buffer = 0.0
position_cap = 0.8
bankroll = 0
kalshi_creds = get_kalshi_creds()
exchange_client = ExchangeClient(exchange_api_base="https://trading-api.kalshi.com/trade-api/v2", email = kalshi_creds[0], password = kalshi_creds[1])
starting_bankroll = exchange_client.get_balance()['balance'] #in cent
token = exchange_client.token
logging.basicConfig(filename='logging_arb_script.txt', encoding='utf-8', level=logging.DEBUG, format="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ",)
try:
  ndx_ab_event = exchange_client.get_event(ndx_ab_ticker)
  spx_ab_event = exchange_client.get_event(spx_ab_ticker)
  ndx_range_event = exchange_client.get_event(ndx_range_ticker)
  spx_range_event = exchange_client.get_event(spx_range_ticker)

  ndx_ab_markets = [x['ticker'] for x in ndx_ab_event['markets']]
  ndx_val_ab_ticker_map = map_value_to_ab_ticker(ndx_ab_markets) #maps lower bound to corresponding above/beloww ticker for NASDAQ100
  spx_ab_markets = [x['ticker'] for x in spx_ab_event['markets']]
  spx_val_ab_ticker_map = map_value_to_ab_ticker(spx_ab_markets) #maps lower bound to corresponding above/below ticker for SP500
  
  ndx_range_markets = [x['ticker'] for x in ndx_range_event['markets']]
  spx_range_markets = [x['ticker'] for x in spx_range_event['markets']]

  ndx_range_subtitles = [(x['ticker'], x['subtitle']) for x in ndx_range_event['markets']]
  spx_range_subtitles = [(x['ticker'], x['subtitle']) for x in spx_range_event['markets']]

  ndx_range_ticker_to_ab_tickers, ndx_ab_ticker_to_range_tickers = map_range_ticker_to_ab_tickers(ndx_range_subtitles, ndx_val_ab_ticker_map)
  spx_range_ticker_to_ab_tickers, spx_ab_ticker_to_range_tickers = map_range_ticker_to_ab_tickers(spx_range_subtitles, spx_val_ab_ticker_map)
  #{}_range_ticker_to_ab_tickers maps a range ticker to the 2 above/below tickers that can be used to replicate the payoff of the range ticker
  #{}_ab_ticker_to_range_tickers maps an above/below ticker to range tickers it can be used to replicate the payoff of

  markets = ndx_range_markets + ndx_ab_markets + spx_range_markets + spx_ab_markets


except Exception as e:
  logging.error("an exception was thrown in the try block of the script that finds the markets for the events", exc_info=True)
else:
  if len(markets) > 0:
    asyncio.run(get_data(markets))
  else:
    logging.debug('no events\' markets form a partition')








