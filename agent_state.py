from datetime import timedelta, datetime
from types import NoneType
import numpy as np


class State:
    #if market making a single ticker, this is the state
    def __init__(self) -> None:
        self.bids = {} #(price: qty) key-value pairs (price is yes price in terms of cents)
        self.asks = {}
        self.best_bid = 0
        self.best_ask  = np.inf
        self.resting_bids = {} # (price: (pos, qty, id)) key-value pairs (price is yes price in terms of cents) -- only 1 resting order per price level allowed
        self.resting_asks = {} # (price : (pos, qty, id)) key-value pairs (price is yes price in terms of cents)
        self.yes_contracts_bought = 0
        self.no_contracts_bought = 0
        self.l1_vol_imb =  None #volume imbalance when considering only best bid and ask
        self.vol_imb =  None #volume imbalance when considering all volumes (vol_bid -  vol_ask)/(vol_bid + vol_ask)
        self.actions = []
        self.PnL = 0 #in cents
        self.epsilon = 0.5
    

    def calc_final_PnL(self):
        if self.yes_contracts_bought == self.no_contracts_bought:
            matched_pnl = (100*self.yes_contracts_bought - self.PnL)/100 #return PnL in dollars
        elif self.yes_contracts_bought > self.no_contracts_bought:
            matched_pnl = (100*self.no_contracts_bought - self.PnL)/100
        else:
            matched_pnl = (100*self.yes_contracts_bought - self.PnL)/100
        return matched_pnl


    def set_orderbook(self, bids, asks):
        #handles snapshots
        self.bids.clear()
        self.asks.clear()
        if bids is None or len(bids) == 0:
            self.bids = {0: 0}
            self.best_bid = 0
        else:
            self.bids = bids
            self.best_bid = max(list(bids.keys()))
        if asks is None or len(asks) == 0:
            self.asks = {np.inf: 0}
            self.best_ask = np.inf
        else:
            self.asks = asks
            self.best_ask = min(list(asks.keys()))
    
    def set_imbalance_indicators(self):
        if self.best_bid == 0 and self.best_ask == np.inf:
            self.l1_vol_imb = 0
            self.vol_imb = 0
        elif self.best_bid == 0:
            self.l1_vol_imb = -1
            self.vol_imb = -1
        elif self.best_ask == np.inf:
            self.l1_vol_imb = 1
            self.vol_imb = 1
        else:
            bb_vol = self.bids[self.best_bid]
            ba_vol = self.asks[self.best_ask]
            bid_vol = sum(self.bids.values())
            ask_vol = sum(self.asks.values())
            self.l1_vol_imb = (bb_vol - ba_vol)/(bb_vol + ba_vol)
            self.vol_imb = (bid_vol - ask_vol)/(bid_vol + ask_vol)

    def add_resting_bid(self, price, qty):
        #price is yes_price in cents
        assert qty > 0
        assert price not in self.resting_bids

        #price not in resting bids

        if price not in self.bids:
            existing_qty = 0
        else:
            existing_qty = self.bids[price]
        pos = existing_qty + 1
        self.resting_bids[price] = [pos, qty]

    def add_resting_ask(self, price, qty):
        #price is yes price in cents
        assert qty > 0
        assert price not in self.resting_asks
        
        #price not in resting asks
        if price in self.asks:
            #price in orderbook
            existing_qty = self.asks[price]
        else:
            existing_qty = 0
        pos = existing_qty + 1
      
    
        self.resting_asks[price] = [pos, qty]

    def adjust_orderbook(self, yes_price, delta, side, agent_order=False, agent_cancel = False):
        '''
        yes_price: [1,99], unit is cents
        delta: an integer, can be positive or negative, indicates number of contracts
        side: ['yes', 'no'] where 'yes':bid side and 'no': ask side
        '''
        #handles deltas
        if delta > 0:
            #increasing liquidity regardless if agent order or not
            if side == 'yes':
                #buy side
                if yes_price in self.bids:
                    #adding liquidity to existing bid price level
                    if agent_order:
                        #add to resting bids
                        if yes_price not in self.resting_bids:
                            self.add_resting_bid(yes_price, delta)
                        else:
                            return
                    self.bids[yes_price]+=delta
                else:
                    #adding liquidity to  a bid price level that does not exist
                    if True:
                        if yes_price > self.best_bid and agent_order:
                            if yes_price >= self.best_ask:
                                self.yes_contracts_bought += abs(delta)
                                self.PnL += -1 * (yes_price * abs(delta))
                            return
                        if agent_order:
                            #add to resting bids
                            assert yes_price not in self.resting_bids
                            self.add_resting_bid(yes_price, delta)
                        self.bids[yes_price] = delta
                        if yes_price > self.best_bid:
                            self.best_bid = yes_price
            else:
                #ask side
                if yes_price in self.asks:
                    #adding liquidity to existing ask price level
                    if agent_order:
                        #add to resting asks
                        if yes_price not in self.resting_asks:
                           self.add_resting_ask(yes_price, delta)
                        else:
                            return
                    self.asks[yes_price] += delta
                else:
                    #adding liquidity to an ask price level that doesn't currently exist
                    if True: 
                        if yes_price < self.best_ask and agent_order:
                            if yes_price <= self.best_bid: 
                                self.no_contracts_bought += delta
                                self.PnL += -1 * (delta * (100-yes_price))
                            return  
                        if agent_order:
                            #add to resting asks 
                            self.add_resting_ask(yes_price, delta)
                        self.asks[yes_price] = delta
                        if yes_price < self.best_ask:
                            self.best_ask = yes_price
        else:
            #decreasing liquidity
            #could be agent order (canceling) or agent order (market buy) or non-agent cancel or fill 
            if agent_cancel and agent_order:
                raise Exception #both of the booleans cannot be true
            if agent_cancel:
                #agent cancel 
                if side == 'yes':
                    #agent is cancelling one of his bids
                    if yes_price not in self.bids:
                        raise Exception
                    self.bids[yes_price]+=delta
                    if self.bids[yes_price] <= 0:
                        del self.bids[yes_price]
                        if yes_price == self.best_bid:
                            if len(self.bids):
                                self.best_bid = max(self.bids.keys())
                            else:
                                self.best_bid = 0
                else:
                    #agent is cancelling an ask
                    if yes_price not in self.asks:
                        raise Exception 
                    self.asks[yes_price] += delta
                    if self.asks[yes_price] <= 0:
                        del self.asks[yes_price]
                        if yes_price == self.best_ask:
                            if len(self.asks):
                                self.best_ask = min(self.asks.keys())
                            else:
                                self.best_ask = np.inf
            elif agent_order:
                #agent market order -- taking liquidity
                #raise Exception
                if side == 'yes':
                    #selling at the bid
                    if len(self.bids):
                        best_bid = self.best_bid
                        amt_bought = min([self.bids[best_bid], abs(delta)])
                        self.no_contracts_bought += amt_bought
                        self.PnL += -1 * (100-best_bid) * amt_bought
                else:
                    #buying at the ask
                    if len(self.asks):
                        best_ask = self.best_ask
                        amt_bought = min([self.asks[best_ask], abs(delta)])
                        self.yes_contracts_bought += amt_bought
                        self.PnL += -1 * best_ask *  amt_bought
            else:
                #neither an agent cancel nor an agent market order
                if side == 'yes':
                    #negative delta in the bid book
                    if yes_price not in self.resting_bids:
                        #no resting bids of mine will be affected
                        if yes_price not in self.bids:
                            print(self.bids)
                            print(yes_price, delta, side, sep=', ')
                            raise Exception
                        else:
                            self.bids[yes_price]+=delta
                            if self.bids[yes_price] <= 0:
                                del self.bids[yes_price]
                                if yes_price == self.best_bid:
                                    if len(self.bids):
                                        self.best_bid = max(self.bids.keys())
                                    else:
                                        self.best_bid = 0
                    else:
                        #resting bid of mine may be affected
                        if yes_price not in self.bids:
                            raise Exception
                        if abs(delta) < self.resting_bids[yes_price][0]:
                            #first resting bid at this price is not being filled
                            resting_bid = self.resting_bids[yes_price]
                            self.resting_bids[yes_price][0]+=delta
                            self.bids[yes_price]+=delta
                            if self.bids[yes_price] <= resting_bid[1]:
                                #no real liquidity avail anymore
                                if np.random.uniform() <= self.epsilon:
                                    #assume my resting orders are hit
                                    self.yes_contracts_bought += resting_bid[1]
                                    self.PnL+= -1 * resting_bid[1] * yes_price
                            
                                del self.resting_bids[yes_price]
                                del self.bids[yes_price]
                                if yes_price == self.best_bid:
                                    if len(self.bids):
                                        self.best_bid = max(self.bids.keys())
                                    else:
                                        self.best_bid = 0           
                        else:
                            #resting bid of agent will be affected   
                            resting_bid = self.resting_bids[yes_price]
                            pos = resting_bid[0]
                            resting_bid_preqty = resting_bid[1]
                            resting_bid[0] = 1
                            resting_bid_postqty = resting_bid[1] - (abs(delta)-pos+1)
                            amt_filled = None
                            if resting_bid_postqty <= 0:
                                amt_filled = resting_bid_preqty
                            else:
                                amt_filled = resting_bid_preqty - resting_bid_postqty
                            self.PnL -=  amt_filled * yes_price
                            self.yes_contracts_bought += amt_filled
                            self.bids[yes_price]-= abs(delta) + amt_filled

                            if self.bids[yes_price] <= 0:
                                assert  resting_bid_postqty <= 0
                                del self.bids[yes_price]
                                del self.resting_bids[yes_price]
                                if yes_price == self.best_bid:
                                    if len(self.bids):
                                        self.best_bid = max(self.bids.keys())
                                    else:
                                        self.best_bid = 0
                            elif self.bids[yes_price] == resting_bid_postqty:
                                #no mmore real liq left
                                if np.random.uniform() <= self.epsilon:
                                    #assume bids were hit
                                    self.yes_contracts_bought += resting_bid_postqty
                                    self.PnL -= resting_bid_postqty * yes_price
                                del self.bids[yes_price]
                                del self.resting_bids[yes_price]
                                if yes_price == self.best_bid:
                                    if len(self.bids):
                                        self.best_bid = max(self.bids.keys())
                                    else:
                                        self.best_bid = 0
                            else:
                                if resting_bid_postqty > 0:
                                    self.resting_bids[yes_price] = [1, resting_bid_postqty]
                                else:
                                    del self.resting_bids[yes_price]                            
                else:
                    #negative delta in the ask book
                    if yes_price not in self.resting_asks:
                        #no resting asks of mine will be affected
                        if yes_price not in self.asks:
                            raise Exception
                        else:
                            self.asks[yes_price] += delta
                            if self.asks[yes_price] <= 0:
                                del self.asks[yes_price]
                                if yes_price == self.best_ask:
                                    if len(self.asks):
                                        self.best_ask = min(self.asks.keys())
                                    else:
                                        self.best_ask = np.inf
                    else:
                        #resting ask of mine may be affected
                        if yes_price not in self.asks:
                            raise Exception
                        if abs(delta) < self.resting_asks[yes_price][0]:
                            #first resting ask at this price is not being filled
                          
                            resting_ask = self.resting_asks[yes_price]
                            resting_ask[0]+=delta
                            self.asks[yes_price]+=delta
                            if self.asks[yes_price] <= resting_ask[1]:
                                #no real liquidity avail anymore
                                if np.random.uniform() <= self.epsilon:
                                    #assume my resting orders are lifted
                                    self.no_contracts_bought += resting_ask[1]
                                    self.PnL+= -1 * resting_ask[1] * (100-yes_price)
                            
                                del self.resting_asks[yes_price]
                                del self.asks[yes_price]
                                if yes_price == self.best_ask:
                                    if len(self.asks):
                                        self.best_ask = min(self.asks.keys())
                                    else:
                                        self.best_ask = np.inf           
                        else:
                            #first resting ask at this price is being lifted at least partially
                            resting_ask = self.resting_asks[yes_price]
                            pos = resting_ask[0]
                            pre_qty = resting_ask[1]
                            post_qty = resting_ask[1] - (abs(delta)-pos+1)
                            amt_filled = None
                            if post_qty <= 0:
                                amt_filled = pre_qty
                            else:
                                amt_filled = pre_qty - post_qty
                            
                            self.PnL -= amt_filled * (100-yes_price)
                            self.no_contracts_bought += amt_filled
                            self.asks[yes_price]-= abs(delta) + amt_filled
                            if self.asks[yes_price] <= 0:
                                assert post_qty <= 0
                                del self.asks[yes_price]
                                del self.resting_asks[yes_price]
                                if yes_price == self.best_ask:
                                    if len(self.asks):
                                        self.best_ask = min(self.asks.keys())
                                    else:
                                        self.best_ask = np.inf
                            elif self.asks[yes_price] == post_qty:
                                #no mmore real liq left
                                if np.random.uniform() <= self.epsilon:
                                    #assume asks were hit
                                    self.no_contracts_bought += post_qty
                                    self.PnL -= post_qty * (100-yes_price)
                                del self.asks[yes_price]
                                del self.resting_asks[yes_price]
                                if yes_price == self.best_ask:
                                    if len(self.asks):
                                        self.best_ask = min(self.asks.keys())
                                    else:
                                        self.best_ask = np.inf
                            else:
                                if post_qty > 0:
                                    self.resting_asks[yes_price] = [1, post_qty]
                                else:
                                    del self.resting_asks[yes_price]

    

    def handle_cancel_order(self, order):
        price = order.price
        if order.order_action == 'buy':
            #agent is cancelling a bid
            if price not in self.resting_bids:
                return
            else:
                rb = self.resting_bids[price]
                del self.resting_bids[price]
                self.adjust_orderbook(price, -1 * rb[1], 'yes', False, True)
        else:
            #agent is cancelling an ask
            if price not in self.resting_asks:
                return
            else:
                ra = self.resting_asks[price]
                del self.resting_asks[price]
                self.adjust_orderbook(price, -1 * ra[1], 'no', False, True)
        

    def process_actions(self, ts):
        #ts is the timestamp of the current delta message you are about to process
        if self.actions is None or len(self.actions)==0:
            return
        else:
            processed = 0
            for action in self.actions:
                if action.ts >= ts:
                    break
                else:
                    processed += 1
                if type(action) == Order:
                    #place an order
                    if action.type == 'limit':
                        assert action.count >= 0
                        '''if action.order_action == 'buy':
                            #bid side limit order
                            print('bid')
                            print(self.bids)
                            print(self.resting_bids)
                        else:
                            #ask side limit order
                            print('ask')
                            print(self.asks)
                            print(self.resting_asks)
                        print(action.price, action.count, sep=', ')'''
                        self.adjust_orderbook(action.price, action.count, 'yes' if action.order_action=='buy' else 'no', True)
                        '''if action.order_action == 'buy':
                            #bid side limit order
                            print('bid')
                            print(self.bids)
                            print(self.resting_bids)
                        else:
                            #ask side limit order
                            print('ask')
                            print(self.asks)
                            print(self.resting_asks)
                        input()'''


                    elif action.type == 'market':
                        self.adjust_orderbook( -1, action.count,'yes' if action.order_action=='buy' else 'no', True )
                elif type(action) == Cancel:
                    self.handle_cancel_order(action)

            self.actions = self.actions[processed:]

class Order:
    def __init__(self, last_msg_ts, order_latency, order_type, order_action, yes_price = 0, qty = 0) -> None:
        '''
        last_msg_ts: timestamp, indicates the  timestamp of the last orderbook event after which you are placing this order
        order_latency: int, units of ms
        order_type: 'limit' or 'market'
        order_action: 'buy' or 'sell'
        qty:  int, indicates number of contracts order is for
        '''
        self.ts = last_msg_ts + timedelta(milliseconds=order_latency)
        self.type = order_type
        self.order_action = order_action
        self.price = yes_price
        self.count =  qty

class Cancel:
    def __init__(self, last_msg_ts, order_latency, order_action, yes_price) -> None:
        '''
        last_msg_ts: timestamp, indicates the  timestamp of the last orderbook event after which you are placing this cancellation
        order_latency: int, units of ms
        order_action: 'buy' or 'sell' where 'buy' inidicates that you want to cancel a bid and 'sell' indicates an ask 
        qty:  int, indicates number of contracts order is for
        '''
        self.ts = last_msg_ts + timedelta(milliseconds = order_latency)
        self.order_action = order_action #buy or sell
        self.price = yes_price

class Strategy:
    def __init__(self, state) -> None:
        self.agent_state = state
        self.last_action = None
    
    def run_naive_strat(self, ts):
        #place limit orders (1 contract each) at best bid and ask if no orders of agent are there already and cancel resting orders at other price levels
        if self.last_action != None and ts <= self.last_action:
            return 

        for p_lvl, order in self.agent_state.resting_bids.items():
                if p_lvl != self.agent_state.best_bid:
                    self.last_action = ts + timedelta(milliseconds= 200)
                    self.agent_state.actions.append(Cancel(ts, 200, 'buy', p_lvl))
        
        for p_lvl, order in self.agent_state.resting_asks.items():
                if p_lvl != self.agent_state.best_ask:
                    self.last_action = ts + timedelta(milliseconds = 200)
                    self.agent_state.actions.append(Cancel(ts, 200, 'sell', p_lvl))


        if self.agent_state.best_bid not in self.agent_state.resting_bids:
            self.last_action = ts + timedelta(milliseconds= 220)  
            self.agent_state.actions.append(Order(ts, 220, 'limit', 'buy', self.agent_state.best_bid, 1))
        
        if self.agent_state.best_ask not in self.agent_state.resting_asks:
            self.last_action = ts + timedelta(milliseconds= 220)
            self.agent_state.actions.append(Order(ts, 220, 'limit', 'sell', self.agent_state.best_ask, 1))

    def run_strat_1(self, ts):
        '''
        This strategy is similar to the naive strategy but it also factors in inventory risk in making the market
        '''
        if self.last_action != None and ts <= self.last_action:
            return 
        inv = self.agent_state.yes_contracts_bought - self.agent_state.no_contracts_bought
        if abs(inv) <= 15:
            #not greater than 15 contracts long or greater than 15 contracts short
            return self.run_naive_strat(ts)
        elif abs(inv) <= 35:
            #either long [16, 35] contracts or short [16, 35] contracts
            if inv > 0:
                #long [16, 35] contracts so we want to avoid our bids being hit
                for p_lvl, p_lvl_orders in self.agent_state.resting_bids.items(): #kill all resting  bids
                    if p_lvl < self.agent_state.best_bid - 3:
                        continue #cancel only resting bids within the following price lvl: [best bid - 3, best bid]
                    for p_lvl_order in p_lvl_orders:
                        self.last_action = ts + timedelta(milliseconds= 200)
                        self.agent_state.actions.append(Cancel(ts, 200, p_lvl_order[2], 'buy', p_lvl))
                
                for p_lvl, p_lvl_orders in self.agent_state.resting_asks.items(): #kill all resting asks
                    for p_lvl_order in p_lvl_orders:
                        self.last_action = ts + timedelta(milliseconds = 200)
                        self.agent_state.actions.append(Cancel(ts, 200, p_lvl_order[2], 'sell', p_lvl))
        else:
            #long > 36 contracts or short > 36 contracts
            
            for p_lvl, p_lvl_orders in self.agent_state.resting_bids.items(): #kill all resting  bids
                for p_lvl_order in p_lvl_orders:
                    self.last_action = ts + timedelta(milliseconds= 200)
                    self.agent_state.actions.append(Cancel(ts, 200, p_lvl_order[2], 'buy', p_lvl))
        
            for p_lvl, p_lvl_orders in self.agent_state.resting_asks.items(): #kill all resting asks
                    for p_lvl_order in p_lvl_orders:
                        self.last_action = ts + timedelta(milliseconds = 200)
                        self.agent_state.actions.append(Cancel(ts, 200, p_lvl_order[2], 'sell', p_lvl))

            if inv > 0:
                #long > 36 contracts, can occur when price is falling and bid is repeatedly hit
                #market sell inv - 10 contracts
                self.last_action = ts + timedelta(milliseconds= 200)
                self.agent_state.actions.append(Order(ts, 200, 'market', 'buy', -1, inv - 10))
            else:
                #short > 36 contracts
                #market buy inv - 10 contracts
                self.last_action = ts + timedelta(milliseconds= 200)
                self.agent_state.actions.append(Order(ts, 200, 'market', 'sell', -1, inv - 10))
            
    
        

    

