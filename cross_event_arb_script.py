from pytz import timezone
from KalshiClientsBaseV2 import KalshiClient, HttpError, ExchangeClient
import requests
import  uuid
import json
import logging
from datetime import date, datetime as dt, timedelta
from urllib3.exceptions import HTTPError
from dateutil import parser
from datetime import datetime
from datetime import timedelta
import asyncio
import websockets
import numpy as np
from helpers import *
import time
import math
  
def execute_cross_event_arb(range_ticker, lb_ticker, ub_ticker, vol_, short_range = True):
  '''
  short_range = True => short range ticker, long lb ticker, short ub ticker 
  short_range = False => long range ticker, short lb ticker, long ub ticker
  '''
  global bankroll
  vol = vol_
  if vol <= 0:
    return
  
  now = datetime.now(tz=eastern)
  if now.hour == 15 and now.minute == 59:
    return

  if short_range:
    range_order_resp = submit_limit_buy(exchange_client, range_ticker, 'no', vol, bb_dict[range_ticker]) #place short order on range ticker
    lb_order_resp = submit_limit_buy(exchange_client, lb_ticker, 'yes', vol, ba_dict[lb_ticker]) #place long order on lb ticker
    ub_order_resp = submit_limit_buy(exchange_client, ub_ticker, 'no', vol, bb_dict[ub_ticker])  #place short order on ub ticker
    logging.info('placed orders to short range ticker, long lb ticker, and short ub ticker')
    time.sleep(.1)

    range_vol = get_taker_fill(exchange_client, range_order_resp)
    bankroll -= range_vol * (100 - (bb_dict[range_ticker]))
    logging.info('Limit Buy: ticker: %s, side: %s, buy_price in cents: %s, count: %s', range_ticker, 'no', str(100 - (bb_dict[range_ticker])), str(range_vol))

    lb_vol = get_taker_fill(exchange_client, lb_order_resp)
    bankroll -= lb_vol * (ba_dict[lb_ticker])
    logging.info('Limit Buy: ticker: %s, side: %s, buy_price in cents: %s, count: %s', lb_ticker, 'yes', str(ba_dict[lb_ticker]), str(lb_vol))

    ub_vol = get_taker_fill(exchange_client, ub_order_resp)
    bankroll -= ub_vol * (100 - (bb_dict[ub_ticker]))
    logging.info('Limit Buy: ticker: %s, side: %s, buy_price in cents: %s, count: %s', ub_ticker, 'no', str(100 - (bb_dict[ub_ticker])), str(ub_vol))

    process_cross_event_arb_orders(exchange_client, range_ticker, range_vol, lb_ticker, lb_vol, ub_ticker, ub_vol)

  else:
    
    lb_order_resp = submit_limit_buy(exchange_client, lb_ticker, 'no', vol, bb_dict[lb_ticker]) #place short order on lb ticker
    ub_order_resp = submit_limit_buy(exchange_client, ub_ticker, 'yes', vol, ba_dict[ub_ticker]) #place long order on ub ticker      
    range_order_resp = submit_limit_buy(exchange_client, range_ticker, 'yes', vol, ba_dict[range_ticker]) #place long order on range ticker
    logging.info('placed orders to long range ticker, short lb ticker, long ub ticker')
    time.sleep(0.1)
    
    lb_vol = get_taker_fill(exchange_client, lb_order_resp)
    bankroll -= lb_vol * (100 - (bb_dict[lb_ticker]))
    logging.info('Limit Buy: ticker: %s, side: %s, buy_price in cents: %s, count: %s', lb_ticker, 'no', str(100 - (bb_dict[lb_ticker])), str(lb_vol))

    ub_vol = get_taker_fill(exchange_client, ub_order_resp)
    bankroll -= ub_vol * ba_dict[ub_ticker]
    logging.info('Limit Buy: ticker: %s, side: %s, buy_price in cents: %s, count: %s', ub_ticker, 'yes', str(ba_dict[ub_ticker]), str(ub_vol))

    range_vol = get_taker_fill(exchange_client, range_order_resp)
    bankroll -= range_vol * ba_dict[range_ticker]
    logging.info('Limit Buy: ticker: %s, side: %s, buy_price in cents: %s, count: %s', range_ticker, 'yes', str(ba_dict[range_ticker]), str(range_vol))
      
    process_cross_event_arb_orders(exchange_client, range_ticker, range_vol, lb_ticker, lb_vol, ub_ticker, ub_vol, False)

async def check_cross_event_arbs(ndx_range_tickers, spx_range_tickers):
  global breakout
  while datetime.now(tz=eastern).hour >= 9 and datetime.now(tz=eastern).hour <= 16 and breakout==False:
    await asyncio.sleep(5)
    for ticker in ndx_range_tickers:
      '''cross event arb for NASDAQ100 events'''
      if ticker not in ndx_range_ticker_to_ab_tickers:
        continue
      ab_tickers = ndx_range_ticker_to_ab_tickers[ticker]
      lb_ticker = ab_tickers[0]
      ub_ticker = ab_tickers[1]
    
      #fees = math.ceil((.035 * price_in_dollars * (1-price_in_dollars) * num_contract) * 100)

      if bb_dict[ticker] > ba_dict[lb_ticker] - bb_dict[ub_ticker]:
        # proceeds from selling to best bid for the range ticker > total cost of (buying long ticker above/below ticker and shorting short ticker above/below ticker)
        #need to figure out min volume across the 2 shorts and 1 long
        min_vol = min([bids[ticker][bb_dict[ticker]], bids[ub_ticker][bb_dict[ub_ticker]], asks[lb_ticker][ba_dict[lb_ticker]]])
        vol = adjust_order_volume(100 - bb_dict[ticker], ba_dict[lb_ticker], 100 - bb_dict[ub_ticker], min_vol, bankroll, min_bankroll)
        sr_fees = math.ceil(100 * (.035 * vol * (bb_dict[ticker]/100) * (1-(bb_dict[ticker]/100)))) + math.ceil(100 * (.035 * vol * (ba_dict[lb_ticker]/100) * (1-(ba_dict[lb_ticker]/100)))) + math.ceil(100 * (.035 * vol * (bb_dict[ub_ticker]/100) * (1-(bb_dict[ub_ticker]/100))))
        if vol * (bb_dict[ticker] - (ba_dict[lb_ticker] - bb_dict[ub_ticker])) > sr_fees:
          logging.info('will try to short range ticker')
          execute_cross_event_arb(ticker, lb_ticker, ub_ticker, vol)
      elif ba_dict[ticker]  < bb_dict[lb_ticker] - ba_dict[ub_ticker]:
        #proceeds from shorting (selling lb ab ticker and longing ub ab ticker) > cost of longing range ticker
        #need to figure out min volume across the 1 short and 2 longs
        min_vol = min([asks[ticker][ba_dict[ticker]], bids[lb_ticker][bb_dict[lb_ticker]], asks[ub_ticker][ba_dict[ub_ticker]]])
        vol = adjust_order_volume(ba_dict[ticker], 100 - bb_dict[lb_ticker], ba_dict[ub_ticker], min_vol, bankroll, min_bankroll)
        lr_fees = math.ceil(100 * (.035 * vol * (ba_dict[ticker]/100) * (1-(ba_dict[ticker]/100)))) + math.ceil(100 * (.035 * vol * (bb_dict[lb_ticker]/100) * (1-(bb_dict[lb_ticker]/100)))) + math.ceil(100 * (.035 * vol * (ba_dict[ub_ticker]/100) * (1-(ba_dict[ub_ticker]/100))))
        if vol * (bb_dict[lb_ticker] - ba_dict[ub_ticker] - ba_dict[ticker]) > lr_fees:
          logging.info('will try to long range ticker')
          execute_cross_event_arb(ticker, lb_ticker, ub_ticker, vol, False)
    
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
        vol = adjust_order_volume(100 - bb_dict[ticker], ba_dict[lb_ticker], 100 - bb_dict[ub_ticker], min_vol, bankroll, min_bankroll)
        sr_fees = math.ceil(100 * (.035 * vol * (bb_dict[ticker]/100) * (1-(bb_dict[ticker]/100)))) + math.ceil(100 * (.035 * vol * (ba_dict[lb_ticker]/100) * (1-(ba_dict[lb_ticker]/100)))) + math.ceil(100 * (.035 * vol * (bb_dict[ub_ticker]/100) * (1-(bb_dict[ub_ticker]/100))))
        if vol * (bb_dict[ticker] - (ba_dict[lb_ticker] - bb_dict[ub_ticker])) > sr_fees:
          logging.info('will try to short range ticker')
          execute_cross_event_arb(ticker, lb_ticker, ub_ticker, vol)
      elif ba_dict[ticker] < bb_dict[lb_ticker] - ba_dict[ub_ticker]:
        #proceeds from shorting (selling lb ab ticker and longing ub ab ticker) > cost of longing range ticker
        #need to figure out min volume across the 1 short and 2 longs
        min_vol = min([asks[ticker][ba_dict[ticker]], bids[lb_ticker][bb_dict[lb_ticker]], asks[ub_ticker][ba_dict[ub_ticker]]])
        vol = adjust_order_volume(ba_dict[ticker], 100 - bb_dict[lb_ticker], ba_dict[ub_ticker], min_vol, bankroll, min_bankroll)
        lr_fees = math.ceil(100 * (.035 * vol * (ba_dict[ticker]/100) * (1-(ba_dict[ticker]/100)))) + math.ceil(100 * (.035 * vol * (bb_dict[lb_ticker]/100) * (1-(bb_dict[lb_ticker]/100)))) + math.ceil(100 * (.035 * vol * (ba_dict[ub_ticker]/100) * (1-(ba_dict[ub_ticker]/100))))
        if vol * (bb_dict[lb_ticker] - ba_dict[ub_ticker] - ba_dict[ticker]) > lr_fees:
          logging.info('will try to long range ticker')
          execute_cross_event_arb(ticker, lb_ticker, ub_ticker, vol, False)

def handle_orderbook_snapshot(cmd_id, response, processed_ts):
    cur_market_ticker = response['msg']['market_ticker']
    resp = response
    global bids, asks, bb_dict, ba_dict, snapshot_vals
    bids[cur_market_ticker] = {0: 0}
    asks[cur_market_ticker] = {100: 0}
    bb_dict[cur_market_ticker] = 0
    ba_dict[cur_market_ticker] = 100
    yes_bids = None
    no_bids = None
    if 'yes' in resp['msg']:
        yes_bids = resp['msg']['yes']
    if 'no' in resp['msg']:
        no_bids = resp['msg']['no']

    if yes_bids is not None:
        for bid_qty in yes_bids:
            bid = bid_qty[0]
            bids[cur_market_ticker][bid] = bid_qty[1]

    bb = max(bids[cur_market_ticker].keys())
    bb_dict[cur_market_ticker] = bb

    if no_bids is not None:
        for bid_qty in no_bids:
            ask = 100-bid_qty[0]
            asks[cur_market_ticker][ask] = bid_qty[1]

    ba = min(asks[cur_market_ticker].keys())
    ba_dict[cur_market_ticker] = ba
    snapshot_vals.append((cmd_id, resp['seq'], resp['msg']['market_ticker'], str(bids[cur_market_ticker]), str(asks[cur_market_ticker]), processed_ts.isoformat()+'Z'))

def handle_yes_orderbook_delta(cmd_id, response, processed_ts):
    global bids, asks, bb_dict, ba_dict, delta_vals
    cur_market_ticker = response['msg']['market_ticker']
    resp = response
    price = resp['msg']['price']
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
    delta_vals.append((cmd_id, resp['seq'], resp['msg']['market_ticker'], price, delta, 'yes', processed_ts.isoformat()+'Z'))

def handle_no_orderbook_delta(cmd_id, response, processed_ts):
    global bids, asks, bb_dict, ba_dict, delta_vals
    #side is no
    cur_market_ticker = response['msg']['market_ticker']
    resp = response
    price = 100 - resp['msg']['price']
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
    delta_vals.append((cmd_id, resp['seq'], resp['msg']['market_ticker'], price, delta, 'no', processed_ts.isoformat()+'Z'))

def write_data():
    global snapshot_vals, delta_vals
    if len(snapshot_vals):
      cursor.executemany('INSERT INTO ob_snapshot VALUES (%s, %s, %s, %s, %s, %s)', snapshot_vals)
      snapshot_vals.clear()
    if len(delta_vals):
       cursor.executemany('INSERT INTO ob_delta VALUES (%s, %s, %s, %s, %s, %s, %s)', delta_vals)
       delta_vals.clear()

def reset():
  '''
  clears bids and asks dicts and resets bb and ba for each ticker to 0, 1 respectively with 0 volume for each
  '''
  global bids, asks, bb_dict, ba_dict
  bids.clear()
  asks.clear()
  bb_dict.clear()
  ba_dict.clear()
  for ticker in markets:    
    bids[ticker] = {0: 0}
    asks[ticker] = {100: 0}
    bb_dict[ticker] = 0
    ba_dict[ticker] = 100

async def get_data(market_tickers):
  command_id = 1
  global breakout, bb_dict, ba_dict, bids, asks
  async for ws in websockets.connect(uri='wss://trading-api.kalshi.com/trade-api/ws/v2', extra_headers={'Authorization': 'Bearer {}'.format(token)}):
    if breakout:
      write_data()
      break
    reset()
    count = 0
    prev_seq_num = 0
    try:
      d = {"id": command_id,"cmd": "subscribe","params": {"channels": ["orderbook_delta", "market_lifecycle"], "market_tickers": market_tickers}}
      await ws.send(json.dumps(d))  #subscribe to channels after establishing new ws connection
      next_sleep_time = get_next_sleep_time(30, 90)
      while True:
        if datetime.utcnow() >= next_sleep_time:
          await asyncio.sleep(0.25)
          next_sleep_time = get_next_sleep_time(30, 90)
        if bankroll <= min_bankroll:
          #break if we've lost or wagered half our starting bankroll for the day
          breakout = True
          break
        if datetime.now(tz=eastern).hour < 9 or (datetime.now(tz=eastern).hour == 16 and datetime.now(tz=eastern).minute >= 30) or datetime.now(tz=eastern).hour > 16:
          #break if before 9 am ET or later than 4:30 ET
          breakout = True
          break
        msg = await ws.recv()
        resp = json.loads(msg)
        count+=1
        if count % 50000 == 0:
          write_data()
        if resp['type'] == 'orderbook_snapshot':
          if resp['seq'] - 1 != prev_seq_num:
            logging.error('orderbook channel message received out of order based on sequence number')
            break
          prev_seq_num = resp['seq']
          handle_orderbook_snapshot(command_id, resp, datetime.utcnow())
        elif resp['type'] == 'orderbook_delta':
          if resp['seq'] - 1 != prev_seq_num:
            logging.error('orderbook channel message received out of order based on sequence number')
            break
          prev_seq_num = resp['seq']
          if resp['msg']['side'] == 'yes':
            handle_yes_orderbook_delta(command_id, resp, datetime.utcnow())
          else:
            handle_no_orderbook_delta(command_id, resp, datetime.utcnow())
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
      write_data()
      command_id+=1
      logging.error('websocket error')
    except Exception as e:
      logging.error("error thrown in get_data function", exc_info=True)
      breakout = True
      command_id+=1

async def main():
  if len(markets) > 0:
    await asyncio.gather(get_data(markets), check_cross_event_arbs(ndx_range_markets, spx_range_markets))
  else:
    logging.debug('no markets')

today = datetime.today().date()
spx_range_ticker, ndx_range_ticker = get_range_event_tickers(today)
spx_ab_ticker, ndx_ab_ticker = get_above_below_event_tickers(today)

eastern = timezone('US/Eastern')
breakout = False
snapshot_vals = []
delta_vals = []
bids = {} #ticker: {price in cents: qty}
asks = {} #ticker: {price in cents: qty}
bb_dict = {} #ticker: best bid price in cents
ba_dict = {} #ticker: best ask price in cents

cursor = get_cursor()
kalshi_creds = get_kalshi_creds()
exchange_client = ExchangeClient(exchange_api_base="https://trading-api.kalshi.com/trade-api/v2", email = kalshi_creds[0], password = kalshi_creds[1])
min_bankroll = round(exchange_client.get_balance()['balance'] * 0.5) #in cents; min bankroll is half of the balance in Kalshi account
bankroll = exchange_client.get_balance()['balance']
token = exchange_client.token
logging.basicConfig(filename='cross_event_arb_log.txt', encoding='utf-8', level=logging.DEBUG, format="%(levelname)s | %(asctime)s | %(message)s", datefmt="%Y-%m-%dT%H:%M:%SZ",)
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

  ndx_range_ticker_to_ab_tickers = map_range_ticker_to_ab_tickers(ndx_range_subtitles, ndx_val_ab_ticker_map)
  spx_range_ticker_to_ab_tickers = map_range_ticker_to_ab_tickers(spx_range_subtitles, spx_val_ab_ticker_map)
  #{}_range_ticker_to_ab_tickers maps a range ticker to the 2 above/below tickers that can be used to replicate the payoff of the range ticker
  #{}_ab_ticker_to_range_tickers maps an above/below ticker to range tickers it can be used to replicate the payoff of

  markets = ndx_range_markets + ndx_ab_markets + spx_range_markets + spx_ab_markets

except Exception as e:
  logging.error("an exception was thrown in the try block of the script that finds the markets for the events", exc_info=True)

asyncio.run(main())






