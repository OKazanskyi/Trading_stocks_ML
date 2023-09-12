#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 19:05:37 2023

@author: olegkazanskyi
To allow multiprocessing I put several functions outside the main code 
otherwise those are called multiple times that is not the best scenario.
The idea is record data to the csv files and consume these after in the main script that is much faster
"""

import os
import pandas as pd #data manipulation and analysis package
#import matplotlib.pyplot as plt #if you want to plot your findings
import datetime
from datetime import date
import requests

if os.name == 'posix':
    #OS
    os.chdir("/Users/olegkazanskyi/Documents/GitHub/Trading")
else:
    #Windows
    os.chdir("C:\\Users\\oleg.kazanskyi\\OneDrive - Danaher\\Documents\\Trading")
    
import Federal_Reserve_Assets
import crisis_dataset
import VIX
import get_sp500_stocks

historical_number_of_days = 12000
future_number_of_days = 1000
#Tiingo_API = input()
EOD_APi ='61e713bb946d18.24351969'

script_location = os.getcwd()

def EOD (rqst, transpose = False, json = False):
    url = rqst
    headers = {
            'Content-Type': 'application/json'
            #'Authorization' : f'Token {EOD_APi}'
            }
    r = requests.get(url, headers=headers)
    response = r.json()
    if json:
        return response
    else:
        if response == 'NA':
            response = pd.DataFrame()
            return response
        else:
            response = pd.DataFrame(response)
            if transpose:
                response = response.transpose()
                return response
            else:
                return response

def get_dates_table(historical_days = 6000, future_days = 700):
    last_date = date.today()
    future_date = last_date + datetime.timedelta(days=future_days)
    historical_date = last_date-datetime.timedelta(days=historical_days)
    dates_df=pd.DataFrame()
    dates_df["date"] = pd.date_range(start=historical_date, end=future_date)
    dates_df.set_index(["date"], inplace = True)
    
    return dates_df

def get_fed_res(dates_df):
    fed_res_data = Federal_Reserve_Assets.download_tables()
    fed_res_data.rename({'Date':'date'}, inplace = True, axis = 1)
    fed_res_data.set_index('date', inplace = True)
    dates_df = dates_df.join(fed_res_data)
    dates_df.sort_index(axis = 0, ascending = False, inplace = True)
    dates_df["Total Assets"].bfill(inplace = True)
    #dates_df['Fed_Balance_MoM'] = 100*dates_df["Total Assets"].pct_change(periods = -30)
    #dates_df['Fed_Balance_YoY'] = 100*dates_df["Total Assets"].pct_change(periods = -360)
    #dates_df = dates_df[dates_df['Fed_Balance_YoY'].notnull()]
    
    
    #dates_df.sort_index(axis = 0, ascending = True, inplace = True)
    dates_df['Fed_SMA50'] = dates_df['Total Assets'].rolling(50).mean()
    #dates_df['Fed_EMA50'] = dates_df['Total Assets'].ewm(span=50).mean()
    dates_df['Fed_SMA200'] = dates_df['Total Assets'].rolling(200).mean()
    #dates_df['Fed_EMA200'] = dates_df['Total Assets'].ewm(span=200).mean()
    dates_df['Fed_SMA50_vector'] = 100*dates_df['Fed_SMA50'].pct_change(periods = 1)
    dates_df['Fed_SMA200_vector'] = 100*dates_df['Fed_SMA50'].pct_change(periods = 1)
    dates_df['Fed_SMA50_position'] = (dates_df['Fed_SMA50']-dates_df["Total Assets"])/dates_df["Total Assets"]
    dates_df['Fed_SMA200_position'] = (dates_df['Fed_SMA200']-dates_df["Total Assets"])/dates_df["Total Assets"]
    dates_df['Fed_SMA50_to_200_position'] = (dates_df['Fed_SMA50']-dates_df["Fed_SMA200"])/dates_df["Fed_SMA200"]
    dates_df.sort_index(axis = 0, ascending = False, inplace = True)
    
    dates_df.drop(["Total Assets","Fed_SMA50","Fed_SMA200"], axis = 1, inplace  = True)
    #dates_df.rename({"Total Assets":"Federal_Assets"}, axis = 1, inplace = True)
    
    return dates_df

def get_10Y_bond_yield(EOD_APi, dates_df, historical_days = 6000):
    rqst_bond = f'https://eodhistoricaldata.com/api/eod/US10Y.GBOND?api_token={EOD_APi}&fmt=json'
    bonds_10Y = EOD(rqst_bond, transpose = False, json = False)
    bonds_10Y["date"] = pd.to_datetime(bonds_10Y["date"], format='%Y-%m-%d',errors='coerce')
    bonds_10Y.set_index('date', inplace = True)
    bonds_10Y.rename({'close':'10YBond'}, axis = 1, inplace = True)
    bonds_10Y = bonds_10Y['10YBond']
    bonds_10Y.sort_index(ascending = False, inplace = True)
    
    dates_df.sort_index(ascending = False, inplace = True)
    
    bonds_df = dates_df.join(bonds_10Y)
    bonds_df = bonds_df.bfill()
    bonds_df['10YB_MoM'] = 100*bonds_df['10YBond'].pct_change(periods = -30)
    bonds_df['10YB_YoY'] = 100*bonds_df['10YBond'].pct_change(periods = -365)
    bonds_df.sort_index(ascending = True, inplace = True)
    bonds_df['10YB_30MA'] = bonds_df['10YBond'].rolling(30).mean()
    bonds_df['10YB_200MA'] = bonds_df['10YBond'].rolling(200).mean()
    bonds_df['10YB_30MA_Vector'] = 100*bonds_df['10YB_30MA'].pct_change(periods = 1)
    bonds_df['10YB_200MA_Vector'] = 100*bonds_df['10YB_200MA'].pct_change(periods = 1)
    bonds_df['10Y_Val_to_30MA'] = 100*(bonds_df['10YBond']-bonds_df['10YB_30MA'])/bonds_df['10YBond']
    bonds_df['10Y_Val_to_200MA'] = 100*(bonds_df['10YBond']-bonds_df['10YB_200MA'])/bonds_df['10YBond']
    bonds_df.drop(['10YB_30MA','10YB_200MA'], axis = 1, inplace = True)
    
    return bonds_df
    
path = os.path.join(script_location,'Script_intermediate_data')

symbols = get_sp500_stocks.get_sp500_symbols()
#Recording variable to a file
path_symbols = os.path.join(path,'symbols.txt')
with open(path_symbols, 'w') as fp:
    for item in symbols:
        # write each item on a new line
        fp.write("%s\n" % item)

#get VIX Volatility index
vix_df = VIX.get_vix(date.today(), historical_number_of_days)
#Recording variable to a file
path_vix = os.path.join(path,'VIX.csv')
vix_df.to_csv(path_vix, index = True)

#Get the dataframe with the daily historical and future days
full_dates_df = get_dates_table(historical_number_of_days)
#Recording variable to a file
path_dates = os.path.join(path,'full_dates.csv')
full_dates_df.to_csv(path_dates, index = True)

#Get the US Federal reserve balance Sheet
fed_reserve_movements = get_fed_res(full_dates_df)
#Recording variable to a file
path_Fed = os.path.join(path,'fed_reserve.csv')
fed_reserve_movements.to_csv(path_Fed, index = True)

#Get 10Y US Gov Bond Yield
bond_US = get_10Y_bond_yield(EOD_APi, full_dates_df)
#Recording variable to a file
path_bonds = os.path.join(path,'10Y_bonds.csv')
bond_US.to_csv(path_bonds, index = True)

#get last crisis date data
last_crisis = crisis_dataset.get_dates()
#Recording variable to a file
path_crisis = os.path.join(path,'crisises.csv')
last_crisis.to_csv(path_crisis, index = True)











