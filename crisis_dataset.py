# -*- coding: utf-8 -*-
"""
Created on Mon May  9 18:24:50 2022
Creates a dataframe with dates and # of days after the last crysis(when stocks dropped for 15%)

@author: oleg.kazanskyi
"""
import datetime
from datetime import date
import pandas as pd
import numpy as np

def get_dates(historical_days = 6000):
    '''
    Creates a DF with dates when S&P dropped for 15% from the max point
    '''
    last_date = date.today()
    historical_date = last_date-datetime.timedelta(days=historical_days)
    
    dates_df=pd.DataFrame()
    dates_df["date"] = pd.date_range(start=historical_date, end=last_date)
    dates_df["last_crisis_day"] = np.nan
    dates_df.loc[(dates_df['date'] == pd.Timestamp(2008, 3, 14)) | (dates_df['date'] == pd.Timestamp(2018, 12, 21)) | (dates_df['date'] == pd.Timestamp(2020, 3, 13)) | (dates_df['date'] == pd.Timestamp(2022, 4, 29)), 'last_crisis_day'] = dates_df['date']
    dates_df.sort_values(by = 'date', axis = 0, ascending = True, inplace = True)
    dates_df.ffill(axis = 0, inplace = True)
    dates_df.sort_values(by = 'date', axis = 0, ascending = False, inplace = True)
    dates_df["days_after_crisis"] = dates_df["date"] - dates_df["last_crisis_day"] 
    dates_df.set_index(["date"], inplace = True)
    dates_df['days_after_crisis'] = pd.to_numeric(dates_df['days_after_crisis'].dt.days, downcast='integer')
    dates_df.drop(["last_crisis_day"], axis = 1, inplace = True)
    return dates_df
