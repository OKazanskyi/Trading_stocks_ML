# -*- coding: utf-8 -*-
"""
Created on Wed Apr 13 09:53:10 2022
This script pulls VIX data from the open source

@author: oleg.kazanskyi
"""

import pandas as pd
import datetime
from datetime import date

def get_vix(last_date = date.today(), historical_days = 1450):
    historical_date = last_date-datetime.timedelta(days=historical_days)
    
    url="https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
    response=pd.read_csv(url)
    
    response["DATE"]=pd.to_datetime(response["DATE"])
    response.rename(columns = {"DATE":"date"}, inplace = True)
    response.set_index(["date"], inplace = True)
    response = response[historical_date:last_date]["HIGH"]
    response = response.rename("VIX_high")
    
    return response