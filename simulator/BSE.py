# -*- coding: utf-8 -*-
#
# BSE: The Bristol Stock Exchange
#
# Version YYB; June 2024 Modified for generate data. by Israel Yang
# Version 1.8; March 2023 added ZIPSH
# Version 1.7; September 2022 added PRDE
# Version 1.6; September 2021 added PRSH
# Version 1.5; 02 Jan 2021 -- was meant to be the final version before switch to BSE2.x, but that didn't happen :-)
# Version 1.4; 26 Oct 2020 -- change to Python 3.x
# Version 1.3; July 21st, 2018 (Python 2.x)
# Version 1.2; November 17th, 2012 (Python 2.x)
#
# Copyright (c) 2012-2023, Dave Cliff
#
#
# ------------------------
#
# MIT Open-Source License:
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial
# portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT
# LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
#
# ------------------------
#
#
#
# BSE is a very simple simulation of automated execution traders
# operating on a very simple model of a limit order book (LOB) exchange
#
# major simplifications in this version:
#       (a) only one financial instrument being traded
#       (b) traders can only trade contracts of size 1 (will add variable quantities later)
#       (c) each trader can have max of one order per single orderbook.
#       (d) traders can replace/overwrite earlier orders, and/or can cancel
#       (d) simply processes each order in sequence and republishes LOB to all traders
#           => no issues with exchange processing latency/delays or simultaneously issued orders.
#
# NB this code has been written to be readable/intelligible, not efficient!

import csv
import sys
import math
import random
import os
import time as chrono
import src.config
from datetime import datetime

import numpy as np

from src.config import useInputFile

# a bunch of system constants (globals)
bse_sys_minprice = 1                    # minimum price in the system, in cents/pennies
bse_sys_maxprice = 500                  # maximum price in the system, in cents/pennies
# ticksize should be a param of an exchange (so different exchanges have different ticksizes)
ticksize = 1  # minimum change in price, in cents/pennies

from src.tbse.tbse_exchange import Exchange
from src.tbse.tbse_customer_orders import customer_orders
from src.tbse.tbse_trader_agents import (
    TraderGiveaway,
    TraderShaver,
    TraderSniper,
    TraderZic,
    TraderZip,
    TraderAA,
    TraderGdx,
    # DeepTrader,
)

def trade_stats(expid, traders, dumpfile, time, lob):

    # Analyse the set of traders, to see what types we have
    trader_types = {}
    for t in traders:
        ttype = traders[t].ttype
        if ttype in trader_types.keys():
            t_balance = trader_types[ttype]['balance_sum'] + traders[t].balance # accumulated balance
            n = trader_types[ttype]['n'] + 1
        else:
            t_balance = traders[t].balance
            n = 1
        trader_types[ttype] = {'n': n, 'balance_sum': t_balance}

    # first two columns of output are the session_id and the time
    dumpfile.write('%s, %06d, ' % (expid, time))

    # second two columns of output are the LOB best bid and best offer (or 'None' if they're undefined)
    if lob['bids']['best'] is not None:
        dumpfile.write('%d, ' % (lob['bids']['best']))
    else:
        dumpfile.write('None, ')
    if lob['asks']['best'] is not None:
        dumpfile.write('%d, ' % (lob['asks']['best']))
    else:
        dumpfile.write('None, ')

    # total remaining number of columns printed depends on number of different trader-types at this timestep
    # for each trader type we print FOUR columns...
    # TraderTypeCode, TotalProfitForThisTraderType, NumberOfTradersOfThisType, AverageProfitPerTraderOfThisType
    for ttype in sorted(list(trader_types.keys())):
        n = trader_types[ttype]['n']
        s = trader_types[ttype]['balance_sum']
        dumpfile.write('%s, %d, %d, %f, ' % (ttype, s, n, s / float(n)))

    dumpfile.write('\n')


# create a bunch of traders from traders_spec
# returns tuple (n_buyers, n_sellers)
# optionally shuffles the pack of buyers and the pack of sellers
def populate_market(traders_spec, traders, shuffle, verbose):
    # traders_spec is a list of buyer-specs and a list of seller-specs
    # each spec is (<trader type>, <number of this type of trader>, optionally: <params for this type of trader>)

    def trader_type(robottype, name, parameters):
        balance = 0.00
        time0 = 0
        if robottype == "GVWY":
            return TraderGiveaway("GVWY", name, 0.00, 0)
        if robottype == "ZIC":
            return TraderZic("ZIC", name, 0.00, 0)
        if robottype == "SHVR":
            return TraderShaver("SHVR", name, 0.00, 0)
        if robottype == "SNPR":
            return TraderSniper("SNPR", name, 0.00, 0)
        if robottype == "ZIP":
            return TraderZip("ZIP", name, 0.00, 0)
        if robottype == "AA":
            return TraderAA("AA", name, 0.00, 0)
        if robottype == "GDX":
            return TraderGdx("GDX", name, 0.00, 0)
        elif robottype == "AA":
            return TraderAA("AA", name, 0.00, 0)
        elif robottype == "GDX":
            return TraderGdx("GDX", name, 0.00, 0)
        else:
            sys.exit('FATAL: don\'t know robot type %s\n' % robottype)

    def shuffle_traders(ttype_char, n, traders):
        for swap in range(n):
            t1 = (n - 1) - swap
            t2 = random.randint(0, t1)
            t1name = '%c%02d' % (ttype_char, t1)
            t2name = '%c%02d' % (ttype_char, t2)
            traders[t1name].tid = t2name
            traders[t2name].tid = t1name
            temp = traders[t1name]
            traders[t1name] = traders[t2name]
            traders[t2name] = temp

    def unpack_params(trader_params, mapping):
        # unpack the parameters for those trader-types that have them
        parameters = None
        if ttype == 'ZIPSH' or ttype == 'ZIP':
            # parameters matter...
            if mapping:
                parameters = 'landscape-mapper'
            elif trader_params is not None:
                parameters = trader_params.copy()
                # trader-type determines type of optimizer used
                if ttype == 'ZIPSH':
                    parameters['optimizer'] = 'ZIPSH'
                else:   # ttype=ZIP
                    parameters['optimizer'] = None
        if ttype == 'PRSH' or ttype == 'PRDE' or ttype == 'PRZI':
            # parameters matter...
            if mapping:
                parameters = 'landscape-mapper'
            else:
                # params determines type of optimizer used
                if ttype == 'PRSH':
                    parameters = {'optimizer': 'PRSH', 'k': trader_params['k'],
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
                elif ttype == 'PRDE':
                    parameters = {'optimizer': 'PRDE', 'k': trader_params['k'],
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}
                else:   # ttype=PRZI
                    parameters = {'optimizer': None, 'k': 1,
                                  'strat_min': trader_params['s_min'], 'strat_max': trader_params['s_max']}

        return parameters

    landscape_mapping = False   # set to true when mapping fitness landscape (for PRSH etc).

    # the code that follows is a bit of a kludge, needs tidying up.
    n_buyers = 0
    for bs in traders_spec['buyers']:
        ttype = bs[0]
        for b in range(bs[1]):
            tname = 'B%02d' % n_buyers  # buyer i.d. string
            if len(bs) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(bs[2], landscape_mapping)
            else:
                params = unpack_params(None, landscape_mapping)
            traders[tname] = trader_type(ttype, tname, params)
            n_buyers = n_buyers + 1

    if n_buyers < 1:
        sys.exit('FATAL: no buyers specified\n')

    if shuffle:
        shuffle_traders('B', n_buyers, traders)

    n_sellers = 0
    for ss in traders_spec['sellers']:
        ttype = ss[0]
        for s in range(ss[1]):
            tname = 'S%02d' % n_sellers  # buyer i.d. string
            if len(ss) > 2:
                # third part of the buyer-spec is params for this trader-type
                params = unpack_params(ss[2], landscape_mapping)
            else:
                params = unpack_params(None, landscape_mapping)
            traders[tname] = trader_type(ttype, tname, params)
            n_sellers = n_sellers + 1

    if n_sellers < 1:
        sys.exit('FATAL: no sellers specified\n')

    if shuffle:
        shuffle_traders('S', n_sellers, traders)

    if verbose:
        for t in range(n_buyers):
            bname = 'B%02d' % t
            print(traders[bname])
        for t in range(n_sellers):
            bname = 'S%02d' % t
            print(traders[bname])
            

    return {'n_buyers': n_buyers, 'n_sellers': n_sellers}


# calculates the mid and micro price of the market after each time step
def lob_data_out(exchange, time, data_file, lobframes, limits):
    """
    This function is used to write the LOB data to a file.
    """

    lob = exchange.publish_lob(time, lobframes, False, data_out=True)
    t = 0

    if lob["bids"]["best"] is None:
        x = 0
    else:
        x = lob["bids"]["best"]

    if lob["asks"]["best"] is None:
        y = 0
    else:
        y = lob["asks"]["best"]
    if limits[0] == 0:
        t = 1

    if time == lob["trade_time"] and time != 0:
        data_file.write(
            "%f,%d,%d,%f,%f,%f,%d,%d,%d,%f,%d,%f,%f,%d\n"
            % (
                time,
                t,
                limits[t],
                lob["mid_price"],
                lob["micro_price"],
                lob["imbalances"],
                lob["spread"],
                x,
                y,
                lob["dt"],
                (lob["asks"]["n"] + lob["bids"]["n"]),
                lob["smiths_alpha"],
                lob["p_estimate"],
                lob["trade_price"],
            )
        )       


# one session in the market

# Demonstration of INPUTs (By Israel Yang):
# sess_id: string type, the order number of the current session. Used to mark the *.csv filename of recording.
# starttime & endtime: float type, the numbers of sec unit that mark the time interval. used in the marching of while loop.
# trader_spec: dictionary type, stores the spec_infos (?) in its index 'sellers' or 'buyers' respectively. Used in populate_market(trader_spec,...). Moreover, in each secp_info, the type of trader, the number of this type and the necessary parameters are recorded.

# order_schedule: dictionary type, {'sup': supply_schedule, 'dem': demand_schedule, 'interval': order_interval, 'timemode': 'drip-poisson'}. Used in customer_orders(...,os,...).
# dump_flags: dictionary type, a list of bool variables that mark if the corresponding files are needed to be written.
# verbose: bool type, marks if the display of detail information is needed.

def market_session(sess_id,
                   sess_length,
                   virtual_end,
                   trader_spec,
                   order_schedule,
                   dump_flags,
                   verbose,
                   schedule_n,
                   lob_out=True):

    def dump_strats_frame(time, stratfile, trdrs):
        # write one frame of strategy snapshot
        line_str = 't=,%.0f, ' % time

        best_buyer_id = None
        best_buyer_prof = 0
        best_buyer_strat = None
        best_seller_id = None
        best_seller_prof = 0
        best_seller_strat = None

        # loop through traders to find the best
        for t in traders:
            trader = trdrs[t]

            # print('PRSH/PRDE/ZIPSH strategy recording, t=%s' % trader)
            if trader.ttype == 'PRSH' or trader.ttype == 'PRDE' or trader.ttype == 'ZIPSH':
                line_str += 'id=,%s, %s,' % (trader.tid, trader.ttype)

                if trader.ttype == 'ZIPSH':
                    # we know that ZIPSH sorts the set of strats into best-first
                    act_strat = trader.strats[0]['stratvec']
                    act_prof = trader.strats[0]['pps']
                else:
                    act_strat = trader.strats[trader.active_strat]['stratval']
                    act_prof = trader.strats[trader.active_strat]['pps']

                line_str += 'actvstrat=,%s ' % trader.strat_csv_str(act_strat)
                line_str += 'actvprof=,%f, ' % act_prof

                if trader.tid[:1] == 'B':
                    # this trader is a buyer
                    if best_buyer_id is None or act_prof > best_buyer_prof:
                        best_buyer_id = trader.tid
                        best_buyer_strat = act_strat
                        best_buyer_prof = act_prof
                elif trader.tid[:1] == 'S':
                    # this trader is a seller
                    if best_seller_id is None or act_prof > best_seller_prof:
                        best_seller_id = trader.tid
                        best_seller_strat = act_strat
                        best_seller_prof = act_prof
                else:
                    # wtf?
                    sys.exit('unknown trader id type in market_session')

        if best_buyer_id is not None:
            line_str += 'best_B_id=,%s, best_B_prof=,%f, best_B_strat=, ' % (best_buyer_id, best_buyer_prof)
            line_str += traders[best_buyer_id].strat_csv_str(best_buyer_strat)

        if best_seller_id is not None:
            line_str += 'best_S_id=,%s, best_S_prof=,%f, best_S_strat=, ' % (best_seller_id, best_seller_prof)
            line_str += traders[best_seller_id].strat_csv_str(best_seller_strat)

        line_str += '\n'

        if verbose:
            print('line_str: %s' % line_str)
        print(1)    
        stratfile.write(line_str)
        stratfile.flush()
        os.fsync(stratfile)

    def blotter_dump(session_id, traders):
        bdump = open(trial_time + '/' + session_id+'_blotters.csv', 'w')
        for t in traders:
            bdump.write('%s, %d\n' % (traders[t].tid, len(traders[t].blotter)))
            for b in traders[t].blotter:
                bdump.write('%s, %s, %.3f, %d, %s, %s, %d\n'
                            % (traders[t].tid, b['type'], b['t'], b['price'], b['party1'], b['party2'], b['qty']))
        bdump.close()

    start_time = 0

    orders_verbose = False
    lob_verbose = False
    process_verbose = False
    respond_verbose = False
    bookkeep_verbose = False
    populate_verbose = False

    if dump_flags['dump_strats']:
        strat_dump = open(trial_time + '/' + sess_id + '_strats.csv', 'w')
    else:
        strat_dump = None

    if dump_flags['dump_lobs']:
        lobframes = open(trial_time + '/' + sess_id + '_LOB_frames.csv', 'w')
    else:
        lobframes = None

    if dump_flags['dump_avgbals']:
        avg_bals = open(trial_time + '/' + sess_id + '_avg_balance.csv', 'w')
    else:
        avg_bals = None

    # initialise the exchange
    exchange = Exchange()

    # create a bunch of traders
    traders = {}
    trader_stats = populate_market(trader_spec, traders, True, populate_verbose)

    lob_file_name = "lob_data/"+str(schedule_n) + "-" + str(chrono.time())
    data_file = None

    if lob_out:
        data_file = open(f"{lob_file_name}.csv", "w", encoding="utf-8")


    # timestep set so that can process all traders in one second
    # NB minimum interarrival time of customer orders may be much less than this!!
    timestep = 1.0 / float(trader_stats['n_buyers'] + trader_stats['n_sellers'])

    # duration = float(endtime - starttime)

    duration = virtual_end

    last_update = -1.0

    time = start_time

    pending_cust_orders = []

    if True:
        print('%s;  \n' % sess_id)

    # frames_done is record of what frames we have printed data for thus far
    frames_done = set()
    cuid = 0  # Customer order id
    while time < (start_time + virtual_end):

        # how much time left, as a percentage?
        time_left = (start_time + virtual_end - time) / duration

        # if verbose: print('\n\n%s; t=%08.2f (%4.1f/100) ' % (sess_id, time, time_left*100))

        trade = None

        [pending_cust_orders, kills, cuid] = customer_orders(
        time, 
        cuid, 
        traders, 
        trader_stats,
        order_schedule, pending_cust_orders, orders_verbose)
        
        # if any newly-issued customer orders mean quotes on the LOB need to be cancelled, kill them
        if len(kills) > 0:
            # if verbose : print('Kills: %s' % (kills))
            for kill in kills:
                # if verbose : print('lastquote=%s' % traders[kill].lastquote)
                if traders[kill].last_quote is not None:
                    # if verbose : print('Killing order %s' % (str(traders[kill].lastquote)))
                    exchange.del_order(time, traders[kill].last_quote)

        # get a limit-order quote (or None) from a randomly chosen trader
        tid = list(traders.keys())[random.randint(0, len(traders) - 1)]
        order = traders[tid].get_order(time, time_left, exchange.publish_lob(time, lobframes, lob_verbose))
        
        # if verbose: print('Trader Quote: %s' % (order))
        limits = [0, 0]
        if order is not None:
            if order.otype == 'Bid': limits[0] = traders[tid].orders[order.coid].price
            if order.otype == 'Ask': limits[1] = traders[tid].orders[order.coid].price

            if order.otype == 'Ask' and order.price < traders[tid].orders[order.coid].price:
                sys.exit('Bad ask')
            if order.otype == 'Bid' and order.price > traders[tid].orders[order.coid].price:
                sys.exit('Bad bid')
            # send order to exchange
            traders[tid].n_quotes = 1
            trade,lob = exchange.process_order2(time, order, process_verbose, None)
            if trade is not None:
                # trade occurred,
                # so the counterparties update order lists and blotters
                traders[trade['party1']].bookkeep(trade, order, bookkeep_verbose, time)
                traders[trade['party2']].bookkeep(trade, order, bookkeep_verbose, time)
                if dump_flags['dump_avgbals']:
                    trade_stats(sess_id, traders, avg_bals, time, exchange.publish_lob(time, lobframes, lob_verbose, False))
                if lob_out: lob_data_out(exchange, time, data_file, lobframes, limits)
            
        # traders respond to whatever happened
        lob = exchange.publish_lob(time, lobframes, lob_verbose, False)
        any_record_frame = False
        for t in traders:
            # NB respond just updates trader's internal variables
            # doesn't alter the LOB, so processing each trader in
            # sequence (rather than random/shuffle) isn't a problem
            record_frame = traders[t].respond(time, lob, trade, respond_verbose)
            if record_frame:
                any_record_frame = True

        # log all the PRSH/PRDE/ZIPSH strategy info for this timestep?
        
        if any_record_frame and dump_flags['dump_strats']:
            # print one more frame to strategy dumpfile
            dump_strats_frame(time, strat_dump, traders)
            # record that we've written this frame
            frames_done.add(int(time))

        time = time + timestep

    # session has ended

    # write trade_stats for this session (NB could use this to write end-of-session summary only)
    if dump_flags['dump_avgbals']:
        trade_stats(sess_id, traders, avg_bals, time, exchange.publish_lob(time, lobframes, lob_verbose, False))
        avg_bals.close()

    if dump_flags['dump_tape']:
        # dump the tape (transactions only -- not writing cancellations)
        exchange.tape_dump(trial_time + '/' + sess_id + '_tape.csv', 'w', 'keep')

    if dump_flags['dump_blotters']:
        # record the blotter for each trader
        blotter_dump(sess_id, traders)

    if dump_flags['dump_strats']:
        strat_dump.close()

    if dump_flags['dump_lobs']:
        lobframes.close()
    
    if lob_out:
        data_file.close()

#############################

def get_order_schedule():
    """
    Produces order schedule as defined in src.config file.
    :return: Order schedule representing the supply/demand curve of the market
    """
    range_max = random.randint(
        src.config.supply["rangeMax"]["rangeLow"],
        src.config.supply["rangeMax"]["rangeHigh"],
    )
    range_min = random.randint(
        src.config.supply["rangeMin"]["rangeLow"],
        src.config.supply["rangeMin"]["rangeHigh"],
    )

    if src.config.useInputFile:
        offset_function_event_list = get_offset_event_list()
        range_s = (
            range_min,
            range_max,
            [real_world_schedule_offset_function, [offset_function_event_list]],
        )
    elif src.config.useOffset:
        range_s = (range_min, range_max, schedule_offset_function)
    else:
        range_s = (range_min, range_max)

    supply_schedule = [
        {
            "from": 0,
            "to": src.config.virtualSessionLength,
            "ranges": [range_s],
            "stepmode": src.config.stepmode,
        }
    ]

    if not src.config.symmetric:
        range_max = random.randint(
            src.config.demand["rangeMax"]["rangeLow"],
            src.config.demand["rangeMax"]["rangeHigh"],
        )
        range_min = random.randint(
            src.config.demand["rangeMin"]["rangeLow"],
            src.config.demand["rangeMin"]["rangeHigh"],
        )

    if src.config.useInputFile:
        offset_function_event_list = get_offset_event_list()
        range_d = (
            range_min,
            range_max,
            [real_world_schedule_offset_function, [offset_function_event_list]],
        )
    elif src.config.useOffset:
        range_d = (range_min, range_max, schedule_offset_function)
    else:
        range_d = (range_min, range_max)

    demand_schedule = [
        {
            "from": 0,
            "to": src.config.virtualSessionLength,
            "ranges": [range_d],
            "stepmode": src.config.stepmode,
        }
    ]

    return {
        "sup": supply_schedule,
        "dem": demand_schedule,
        "interval": src.config.interval,
        "timemode": src.config.timemode,
    }


def schedule_offset_function(t):
    """
    schedule_offset_function returns t-dependent offset on schedule prices
    :param t: Time at which we are retrieving the offset
    :return: The offset
    """
    print(t)
    pi2 = math.pi * 2
    c = math.pi * 3000
    wavelength = t / c
    gradient = 100 * t / (c / pi2)
    amplitude = 100 * t / (c / pi2)
    offset = gradient + amplitude * math.sin(wavelength * t)
    return int(round(offset, 0))


def real_world_schedule_offset_function(t, params):
    """
    Returns offset based on real world data read in via CSV
    :param t: Time at which the offset is being calculated
    :param params: Parameters used to find offset
    :return: The offset
    """
    end_time = float(params[0])
    offset_events = params[1]
    # this is quite inefficient: on every call it walks the event-list
    # come back and make it better
    percent_elapsed = t / end_time
    offset = 0
    for event in offset_events:
        offset = event[1]
        if percent_elapsed < event[0]:
            break
    return offset


# pylint: disable:too-many-locals


def get_offset_event_list():
    """
    read in a real-world-data data-file for the SDS offset function
    having this here means it's only read in once
    this is all quite skanky, just to get it up and running
    assumes data file is all for one date, sorted in t order, in correct format, etc. etc.
    :return: list of offset events
    """
    with open(src.config.input_file, "r", encoding="utf-8") as input_file:
        rwd_csv = csv.reader(input_file)
        scale_factor = 80
        # first pass: get t & price events, find out how long session is, get min & max price
        min_price = None
        max_price = None
        first_time_obj = None
        price_events = []
        time_since_start = 0
        for line in rwd_csv:
            t = line[1]
            if first_time_obj is None:
                first_time_obj = datetime.strptime(t, "%H:%M:%S")
            time_obj = datetime.strptime(t, "%H:%M:%S")
            price = float(line[2])
            if min_price is None or price < min_price:
                min_price = price
            if max_price is None or price > max_price:
                max_price = price
            time_since_start = (time_obj - first_time_obj).total_seconds()
            price_events.append([time_since_start, price])
        # second pass: normalise times to fractions of entire t-series duration
        #              & normalise price range
        price_range = max_price - min_price
        end_time = float(time_since_start)
        offset_function_event_list = []
        for event in price_events:
            # normalise price
            normld_price = (event[1] - min_price) / price_range
            # clip
            normld_price = min(normld_price, 1.0)
            normld_price = max(0.0, normld_price)
            # scale & convert to integer cents
            price = int(round(normld_price * scale_factor))
            normld_event = [event[0] / end_time, price]
            offset_function_event_list.append(normld_event)
        return offset_function_event_list


# # Below here is where we set up and run a whole series of experiments


if __name__ == "__main__":
    if not src.config.parse_config():
        sys.exit()

    os.makedirs('lob_data', exist_ok=True)
    os.makedirs('tdump_data', exist_ok=True)

    # Input configuration
    USE_CONFIG = False
    USE_CSV = False
    USE_COMMAND_LINE = False

    NUM_ZIC = src.config.numZIC
    NUM_ZIP = src.config.numZIP
    NUM_GDX = src.config.numGDX
    NUM_AA = src.config.numAA
    NUM_GVWY = src.config.numGVWY
    NUM_SHVR = src.config.numSHVR
    NUM_DTR = src.config.numDTR

    NUM_OF_ARGS = len(sys.argv)
    if NUM_OF_ARGS == 1:
        USE_CONFIG = True
    elif NUM_OF_ARGS == 2:
        USE_CSV = True
    elif NUM_OF_ARGS == 8:
        USE_COMMAND_LINE = True
        try:
            NUM_ZIC = int(sys.argv[1])
            NUM_ZIP = int(sys.argv[2])
            NUM_GDX = int(sys.argv[3])
            NUM_AA = int(sys.argv[4])
            NUM_GVWY = int(sys.argv[5])
            NUM_SHVR = int(sys.argv[6])
            NUM_DTR = int(sys.argv[7])
        except ValueError:
            print("ERROR: Invalid trader schedule. Please enter seven integer values.")
            sys.exit()
    else:
        print("Invalid input arguements.")
        print("Options for running TBSE:")
        print("	$ python3 tbse.py  ---  Run using trader schedule from src.config.")
        print(
            " $ python3 tbse.py <string>.csv  ---  Enter name of csv file describing a series of trader schedules."
        )
        print(
            " $ python3 tbse.py <int> <int> <int> <int> <int> <int> <int>  ---  Enter 7 integer values representing \
        trader schedule."
        )
        sys.exit()
    # pylint: disable=too-many-boolean-expressions
    if (
        NUM_ZIC < 0
        or NUM_ZIP < 0
        or NUM_GDX < 0
        or NUM_AA < 0
        or NUM_GVWY < 0
        or NUM_SHVR < 0
        or NUM_DTR < 0
    ):
        print("ERROR: Invalid trader schedule. All input integers should be positive.")
        sys.exit()
    
    n_trials_recorded = 1

    verbose = True

    if USE_CONFIG or USE_COMMAND_LINE:

        print('Use Config.')
        order_sched = get_order_schedule()

        buyers_spec = [
            ("ZIC", NUM_ZIC),
            ("ZIP", NUM_ZIP),
            ("GDX", NUM_GDX),
            ("AA", NUM_AA),
            ("GVWY", NUM_GVWY),
            ("SHVR", NUM_SHVR),
            ("DTR", NUM_DTR),
        ]

        sellers_spec = buyers_spec
        traders_spec = {"sellers": sellers_spec, "buyers": buyers_spec}

        trial = 1
        dump_all = True

        trial_time: str = 'trial-'+ chrono.strftime("%Y%m%d-%H%M%S", chrono.localtime(chrono.time())) + f"-{n_trials_recorded}"

        directory_path = f"./{trial_time}"

        os.makedirs(directory_path, exist_ok=True)

        while trial < (src.config.numTrials + 1):
            trial_id = f"trial{str(trial).zfill(7)}"
            if trial > n_trials_recorded: 
                dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                        'dump_avgbals': False, 'dump_tape': False}
            else:
                dump_flags = {'dump_blotters': True, 'dump_lobs': True, 'dump_strats': True,
                        'dump_avgbals': True, 'dump_tape': True}
            #try:
            market_session(
                trial_id,
                src.config.sessionLength,
                src.config.virtualSessionLength,
                traders_spec,
                order_sched,
                dump_flags,
                verbose,
                0,
                True,
            )

            # except Exception as e:  # pylint: disable=broad-except
            #     print("Error: Market session failed, trying again.")
            #     print(e)
            #     trial = trial - 1
                
            chrono.sleep(0.5)
            trial = trial + 1


    elif USE_CSV:
        print('Using CSV. \n')

        trial_time: str = 'trial-'+ chrono.strftime("%Y%m%d-%H%M%S", chrono.localtime(chrono.time())) + f"-{n_trials_recorded}"

        directory_path = f"./{trial_time}"

        server = sys.argv[1]
        ratios = []
        try:
            with open(server, newline="", encoding="utf-8") as csv_file:
                reader = csv.reader(csv_file, delimiter=",")
                for row in reader:
                    ratios.append(row)
        except FileNotFoundError:
            print("ERROR: File " + server + " not found.")
            sys.exit()
        except IOError as e:
            print("ERROR: " + e)
            sys.exit()

        trial_number = 1

        for no_of_schedule, ratio in enumerate(ratios):
            try:
                NUM_ZIC = int(ratio[0])
                NUM_ZIP = int(ratio[1])
                NUM_GDX = int(ratio[2])
                NUM_AA = int(ratio[3])
                NUM_GVWY = int(ratio[4])
                NUM_SHVR = int(ratio[5])
                NUM_DTR = int(ratio[6])
            except ValueError:
                print(
                    "ERROR: Invalid trader schedule. Please enter seven, comma-separated, integer values. Skipping "
                    "this trader schedule."
                )
                continue
            except Exception as e:  # pylint: disable=broad-except
                print(
                    "ERROR: Unknown input error. Skipping this trader schedule."
                    + str(e)
                )
                continue
            # pylint: disable=too-many-boolean-expressions
            if (
                NUM_ZIC < 0
                or NUM_ZIP < 0
                or NUM_GDX < 0
                or NUM_AA < 0
                or NUM_GVWY < 0
                or NUM_SHVR < 0
                or NUM_DTR < 0
            ):
                print(
                    "ERROR: Invalid trader schedule. All input integers should be positive. Skipping this trader"
                    " schedule."
                )
                continue

            for _ in range(0, src.config.numSchedulesPerRatio):
                order_sched = get_order_schedule()

                buyers_spec = [
                    ("ZIC", NUM_ZIC),
                    ("ZIP", NUM_ZIP),
                    ("GDX", NUM_GDX),
                    ("AA", NUM_AA),
                    ("GVWY", NUM_GVWY),
                    ("SHVR", NUM_SHVR),
                    ("DTR", NUM_DTR),
                ]

                sellers_spec = buyers_spec
                traders_spec = {"sellers": sellers_spec, "buyers": buyers_spec}

                trader_count = 0
                for ttype in buyers_spec:
                    trader_count += ttype[1]
                for ttype in sellers_spec:
                    trader_count += ttype[1]

                if trader_count > 40:
                    print("WARNING: Too many traders can cause unstable behaviour.")

                trial = 1
                dump_all = False

                while trial <= src.config.numTrialsPerSchedule:
                    trial_id = f"trial{str(trial_number).zfill(7)}"

                    # if trial > n_trials_recorded: 
                        
                    #     dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                    #             'dump_avgbals': False, 'dump_tape': False}
                    # else:
                    #     dump_flags = {'dump_blotters': True, 'dump_lobs': True, 'dump_strats': True,
                    #             'dump_avgbals': True, 'dump_tape': True}
    
                    dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
                            'dump_avgbals': False, 'dump_tape': False}                    
                    try:
                        # NUM_THREADS = market_session(
                        #     trial_id,
                        #     src.config.sessionLength,
                        #     src.config.virtualSessionLength,
                        #     traders_spec,
                        #     order_sched,
                        #     start_session_event,
                        #     False,
                        #     tdump,
                        #     dump_all,
                        #     (no_of_schedule + 5*(int(server[-6:-4]))),
                        #     lob_out=True,
                        # )

                        market_session(
                            trial_id,
                            src.config.sessionLength,
                            src.config.virtualSessionLength,
                            traders_spec,
                            order_sched,
                            dump_flags,
                            verbose,
                            (no_of_schedule + 5*(int(server[-6:-4]))),
                            lob_out=True,
                        )                            

                        #if NUM_THREADS != trader_count + 2:

                    except Exception as e:  # pylint: disable=broad-except
                        print("Market session failed. Trying again. " + str(e))
                        trial = trial - 1
                        trial_number = trial_number - 1

                    trial = trial + 1
                    trial_number = trial_number + 1

        sys.exit("Done Now")

    else:
        print("ERROR: An unknown error has occurred. Something is very wrong.")
        sys.exit()



# Below is the old version. Jun.23 2024

    

#     t_prog_s = chrono.time()
#     # set up common parameters for all market sessions
#     # 1000 days is good, but 3*365=1095, so may as well go for three years.
    
#     if len(sys.argv) > 1:
#         n_days = float(sys.argv[1])
#     else :
#         n_days = 1

#     start_time = 0.0
#     # end_time = 60.0 * 60.0 * 24 * n_days
#     end_time = 5
#     duration = end_time - start_time

#     # schedule_offsetfn returns time-dependent offset, to be added to schedule prices
#     def schedule_offsetfn(t):
#         pi2 = math.pi * 2
#         c = math.pi * 3000
#         wavelength = t / c
#         gradient = 100 * t / (c / pi2)
#         amplitude = 100 * t / (c / pi2)
#         offset = gradient + amplitude * math.sin(wavelength * t)
#         return int(round(offset, 0))

#     # Here is an example of how to use the offset function
#     #
#     # range1 = (10, 190, schedule_offsetfn)
#     # range2 = (200, 300, schedule_offsetfn)

#     # Here is an example of how to switch from range1 to range2 and then back to range1,
#     # introducing two "market shocks"
#     # -- here the timings of the shocks are at 1/3 and 2/3 into the duration of the session.
#     #
#     # supply_schedule = [ {'from':start_time, 'to':duration/3, 'ranges':[range1], 'stepmode':'fixed'},
#     #                     {'from':duration/3, 'to':2*duration/3, 'ranges':[range2], 'stepmode':'fixed'},
#     #                     {'from':2*duration/3, 'to':end_time, 'ranges':[range1], 'stepmode':'fixed'}
#     #                   ]

#     range1 = (50, 150)
#     supply_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range1], 'stepmode': 'fixed'}]

#     range2 = (50, 150)
#     demand_schedule = [{'from': start_time, 'to': end_time, 'ranges': [range2], 'stepmode': 'fixed'}]

#     # new customer orders arrive at each trader approx once every order_interval seconds
#     order_interval = 2

#     order_sched = {'sup': supply_schedule, 'dem': demand_schedule,
#                    'interval': order_interval, 'timemode': 'periodic'}

#     # Use 'periodic' if you want the traders' assignments to all arrive simultaneously & periodically
#     #               'order_interval': 30, 'timemode': 'periodic'}

#     # buyers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]
#     # sellers_spec = [('GVWY',10),('SHVR',10),('ZIC',10),('ZIP',10)]

#     opponent = 'GVWY'
#     opp_N = 30
# #    sellers_spec = [('PRSH', 30),(opponent, opp_N-1)]
# #    buyers_spec = [(opponent, opp_N)]


#     # run a sequence of trials, one session per trial

#     verbose = True

#     # n_trials is how many trials (i.e. market sessions) to run in total
#     n_trials = 1

#     # n_recorded is how many trials (i.e. market sessions) to write full data-files for
#     n_trials_recorded = 1

#     trial = 1

#     trial_time: str = 'trial-'+ chrono.strftime("%Y%m%d-%H%M%S", chrono.localtime(chrono.time())) + f"-{n_trials}-{n_trials_recorded}"

#     directory_path = f"./{trial_time}"

#     os.makedirs(directory_path, exist_ok=True)

#     while trial < (n_trials+1):

#         # create unique i.d. string for this trial
#         trial_id = 'bse_d%03d_i%02d_%04d' % (n_days, order_interval, trial)

#         # buyers_spec = [('ZIPSH', 10, {'k': 4})] # ?
#         # sellers_spec = [('ZIPSH', 10, {'k': 4})] # ?

#         buyers_spec = [('SNPR', 5), ('GVWY', 5), ('ZIC', 5), ('ZIP', 5)]
#         # buyers_spec = [('ZIP', 20)]
#         sellers_spec = buyers_spec

#         traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}

#         n_traders = sum([tdr[1] for tdr in traders_spec['sellers']])+sum([tdr[1] for tdr in traders_spec['buyers']])

#         if trial > n_trials_recorded: 
#             dump_flags = {'dump_blotters': False, 'dump_lobs': False, 'dump_strats': False,
#                           'dump_avgbals': False, 'dump_tape': False}
#         else:
#             dump_flags = {'dump_blotters': True, 'dump_lobs': True, 'dump_strats': True,
#                           'dump_avgbals': True, 'dump_tape': True}

#         market_session(trial_id, start_time, end_time, traders_spec, order_sched, dump_flags, verbose, True)

#         trial = trial + 1

#     dt_prog = chrono.time()-t_prog_s

#     with open(trial_time +'/time_consumed.dat', 'w') as file:
#     # 写入数据到文件
#         file.write(f"{dt_prog:.4f} seconds have been spent for {n_trials} trials of {n_days} days duration with {n_traders} traders.")

    # run a sequence of trials that exhaustively varies the ratio of four trader types
    # NB this has weakness of symmetric proportions on buyers/sellers -- combinatorics of varying that are quite nasty
    #
    # n_trader_types = 4
    # equal_ratio_n = 4
    # n_trials_per_ratio = 50
    #
    # n_traders = n_trader_types * equal_ratio_n
    #
    # fname = 'balances_%03d.csv' % equal_ratio_n
    #
    # tdump = open(fname, 'w')
    #
    # min_n = 1
    #
    # trialnumber = 1
    # trdr_1_n = min_n
    # while trdr_1_n <= n_traders:
    #     trdr_2_n = min_n
    #     while trdr_2_n <= n_traders - trdr_1_n:
    #         trdr_3_n = min_n
    #         while trdr_3_n <= n_traders - (trdr_1_n + trdr_2_n):
    #             trdr_4_n = n_traders - (trdr_1_n + trdr_2_n + trdr_3_n)
    #             if trdr_4_n >= min_n:
    #                 buyers_spec = [('GVWY', trdr_1_n), ('SHVR', trdr_2_n),
    #                                ('ZIC', trdr_3_n), ('ZIP', trdr_4_n)]
    #                 sellers_spec = buyers_spec
    #                 traders_spec = {'sellers': sellers_spec, 'buyers': buyers_spec}
    #                 # print buyers_spec
    #                 trial = 1
    #                 while trial <= n_trials_per_ratio:
    #                     trial_id = 'trial%07d' % trialnumber
    #                     market_session(trial_id, start_time, end_time, traders_spec,
    #                                    order_sched, tdump, False, True)
    #                     tdump.flush()
    #                     trial = trial + 1
    #                     trialnumber = trialnumber + 1
    #             trdr_3_n += 1
    #         trdr_2_n += 1
    #     trdr_1_n += 1
    # tdump.close()
    #
    # print(trialnumber)
