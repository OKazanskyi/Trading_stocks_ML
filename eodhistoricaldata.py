# -*- coding: utf-8 -*-
"""
Created on Mon Nov 28 13:29:05 2022

@author: oleg.kazanskyi
"""
import os
import pandas as pd #data manipulation and analysis package
import numpy as np
#import matplotlib.pyplot as plt #if you want to plot your findings
import datetime
from datetime import date
from datetime import timedelta
import requests

#Ignore unnecessary warnings
#import warnings
#warnings.filterwarnings("ignore")

if os.name == 'posix':
    #OS
    os.chdir("/Users/olegkazanskyi/Documents/GitHub/Trading")
else:
    #Windows
    os.chdir("C:\\Users\\oleg.kazanskyi\\Documents\\Trading")
    
script_location = os.getcwd()

import matplotlib.pyplot as plt

#multiprocessing feature
import multiprocessing as mp
print("Number of cpu : ", mp.cpu_count())
cores = mp.cpu_count()

#%matplotlib qt

#Enter TIINGO
print("Enter 40 signs tiingo API: ")  
historical_number_of_days = 12000
future_number_of_days = 1000
#Tiingo_API = input()
EOD_APi ='61e713bb946d18.24351969'

def get_splits(splits, dates_df, historical_days = 6000):
    '''
    Creates a DF with dates when S&P dropped for 15% from the max point
    '''
    if splits.empty:
        splits = splits.append({'date' : datetime.datetime.now(), 'cum_coef':1}, ignore_index = True)
        splits.set_index(["date"], inplace = True)
        
        dates_df = dates_df.join(splits)
        dates_df.sort_index(axis = 0, ascending = False, inplace = True)
        dates_df["cum_coef"].ffill(inplace = True)
        dates_df["cum_coef"].fillna(1, inplace = True)
        
        
        return dates_df
    else:

        splits["date"] = splits["date"].apply(pd.to_datetime, errors='coerce')
        splits[['to','from']] = splits['split'].str.split('/',expand=True)
        splits[['to','from']] = splits[['to','from']].apply(pd.to_numeric, errors='coerce')
        splits['coef'] = splits['from'] /splits['to']
        splits['date_shifted'] = splits['date'] - timedelta(days=1)
        splits.sort_values(by=["date"], ascending = False, inplace = True)
        splits["cum_coef"] = splits['coef'].cumprod()
        splits.set_index(["date", "split", "to", "from", "coef"], inplace = True)
        splits.rename(columns={"date_shifted": "date"}, inplace = True)
        splits.set_index(["date"], inplace = True)
    
        dates_df = dates_df.join(splits)
        dates_df.sort_index(axis = 0, ascending = False, inplace = True)
        dates_df["cum_coef"].ffill(inplace = True)
        dates_df["cum_coef"].fillna(1, inplace = True)
    
        return dates_df

def mod_earn_trend (earn_trend, earn_history, splits_daily):
    '''
    Parameters
    ----------
    earn_trend : dataframe from earning_trend reques
        DESCRIPTION.
    historical_days : INT, optional
        DESCRIPTION. The default is 6000.
        How big range of dates for the earnings trend you want to have

    Returns
    -------
    earn_trend : dataframe
        DESCRIPTION.
        Function converts all earning to quarter basis (from mix of Year/Quarter) and keep only the most important numeric values

    '''
    #copy_df = earn_trend.copy()
    #earn_trend = copy_df.copy()
    #dropping the columns that look redundant for our purpose
    earn_trend_mod = earn_trend.copy()
    #earn_trend_mod_2 = earn_trend_mod.copy()
    earn_trend_mod.drop(["earningsEstimateNumberOfAnalysts","revenueEstimateAvg", "revenueEstimateLow", "revenueEstimateHigh", "revenueEstimateYearAgoEps",'revenueEstimateNumberOfAnalysts', 'epsRevisionsUpLast7days','epsRevisionsUpLast30days','epsRevisionsDownLast7days','epsRevisionsDownLast30days', 'earningsEstimateLow', 'earningsEstimateHigh','growth', 'earningsEstimateGrowth', 'epsTrendCurrent'],axis = 1,inplace = True)
    #earn_trend.drop(['epsTrendCurrent'],axis = 1,inplace = True)
    #convert date column to datetime
    earn_trend_mod["date"] = pd.to_datetime(earn_trend_mod["date"], format='%Y-%m-%d',errors='coerce')
    #adding financial year
    earn_trend_mod['financial_year'] = earn_trend_mod['date'].map(lambda x: x.year if x.month <= 9 else x.year+1)
    #adding the column with 1 values to count later
    earn_trend_mod['row'] = 1
    #Converting columns to numeric values
    num_cols = ['earningsEstimateAvg', 'earningsEstimateYearAgoEps', 'revenueEstimateGrowth', 'epsTrend7daysAgo', 'epsTrend30daysAgo', 'epsTrend60daysAgo', 'epsTrend90daysAgo']
    earn_trend_mod[num_cols] =  earn_trend[num_cols].apply(pd.to_numeric, errors='coerce')
    
    
    #Converting earn_history
    earn_history_df = earn_history.copy()
    earn_history_df["date"] = pd.to_datetime(earn_history_df["date"] , format='%Y-%m-%d',errors='coerce')
    earn_history_df["reportDate"] = pd.to_datetime(earn_history_df["reportDate"] , format='%Y-%m-%d',errors='coerce')
    #Convert numeric columns
    columns_num = ['epsActual','epsEstimate', 'epsDifference', 'surprisePercent']
    for i in columns_num:
        earn_history_df[i]= earn_history_df[i].astype(str).replace('None',np.nan)
        earn_history_df[i] = earn_history_df[i].apply(pd.to_numeric, errors='coerce')
    
    # Getting Report date for different periods in the table
    earn_trend_mod = earn_trend_mod.merge(earn_history_df, how='left', left_on = "date", right_on = "date")
    earn_trend_mod['report_date_7d_ago'] = earn_trend_mod['reportDate'] - pd.DateOffset(days=7)
    earn_trend_mod['report_date_30d_ago'] = earn_trend_mod['reportDate'] - pd.DateOffset(days=30)
    earn_trend_mod['report_date_60d_ago'] = earn_trend_mod['reportDate'] - pd.DateOffset(days=60)
    earn_trend_mod['report_date_90d_ago'] = earn_trend_mod['reportDate'] - pd.DateOffset(days=90)
    
    #Merging diffrent dates with the splits data so we get the value we can compare with the today's
    earn_trend_mod = earn_trend_mod.merge(splits_daily, how='left', left_on = "reportDate", right_on = "date")
    earn_trend_mod.rename(columns={"cum_coef": "cum_same_d", 'date_x': 'date'}, inplace = True)
    earn_trend_mod = earn_trend_mod.merge(splits_daily, how='left', left_on = "report_date_7d_ago", right_on = "date")
    earn_trend_mod.rename(columns={"cum_coef": "cum_7_d", 'date_x': 'date'}, inplace = True)
    earn_trend_mod = earn_trend_mod.merge(splits_daily, how='left', left_on = "report_date_30d_ago", right_on = "date")
    earn_trend_mod.rename(columns={"cum_coef": "cum_30_d", 'date_x': 'date'}, inplace = True)
    earn_trend_mod = earn_trend_mod.merge(splits_daily, how='left', left_on = "report_date_60d_ago", right_on = "date")
    earn_trend_mod.rename(columns={"cum_coef": "cum_60_d", 'date_x': 'date'}, inplace = True)
    earn_trend_mod = earn_trend_mod.merge(splits_daily, how='left', left_on = "report_date_90d_ago", right_on = "date")
    earn_trend_mod.rename(columns={"cum_coef": "cum_90_d", 'date_x': 'date'}, inplace = True)
    earn_trend_mod[["cum_same_d","cum_7_d","cum_30_d","cum_60_d","cum_90_d"]] = earn_trend_mod[["cum_same_d","cum_7_d","cum_30_d","cum_60_d","cum_90_d"]].fillna(1)
    
    #Converting Earnings estimate to the same value
    earn_trend_mod['earningsEstimateAvg'] = earn_trend_mod['earningsEstimateAvg'] * earn_trend_mod['cum_same_d']
    earn_trend_mod['earningsEstimateYearAgoEps'] = earn_trend_mod['earningsEstimateYearAgoEps'] * earn_trend_mod['cum_same_d']    
    earn_trend_mod['epsTrend7daysAgo'] = earn_trend_mod['epsTrend7daysAgo'] * earn_trend_mod['cum_7_d']
    earn_trend_mod['epsTrend30daysAgo'] = earn_trend_mod['epsTrend30daysAgo'] * earn_trend_mod['cum_30_d']
    earn_trend_mod['epsTrend60daysAgo'] = earn_trend_mod['epsTrend60daysAgo'] * earn_trend_mod['cum_60_d']
    earn_trend_mod['epsTrend90daysAgo'] = earn_trend_mod['epsTrend90daysAgo'] * earn_trend_mod['cum_90_d']
    
    #Calculating Projected Growth (PG)
    earn_trend_mod['PG_YoY'] = 100*(earn_trend_mod['earningsEstimateAvg'] - earn_trend_mod['earningsEstimateYearAgoEps']) / earn_trend_mod['earningsEstimateYearAgoEps']
    earn_trend_mod['PG_7d_YoY'] = 100*(earn_trend_mod['epsTrend7daysAgo'] - earn_trend_mod['earningsEstimateYearAgoEps']) / earn_trend_mod['earningsEstimateYearAgoEps']
    earn_trend_mod['PG_30d_YoY'] = 100*(earn_trend_mod['epsTrend30daysAgo'] - earn_trend_mod['earningsEstimateYearAgoEps']) / earn_trend_mod['earningsEstimateYearAgoEps']
    earn_trend_mod['PG_60d_YoY'] = 100*(earn_trend_mod['epsTrend60daysAgo'] - earn_trend_mod['earningsEstimateYearAgoEps']) / earn_trend_mod['earningsEstimateYearAgoEps']
    earn_trend_mod['PG_90d_YoY'] = 100*(earn_trend_mod['epsTrend90daysAgo'] - earn_trend_mod['earningsEstimateYearAgoEps']) / earn_trend_mod['earningsEstimateYearAgoEps']
    
    #Let's keep only the columns with % calculations
    earn_trend_mod = earn_trend_mod[["date","PG_YoY","PG_7d_YoY","PG_30d_YoY","PG_60d_YoY","PG_90d_YoY"]]
    # Creating a datraframe with future and historical dates with all htey days
        
    return earn_trend_mod

def mod_earn_history(earn_history, dates_df_t, historical_days = 6000, future_days = 400):
    '''
    Creates a historical earnings DF with daily data 
    '''
    
    ##MAKE CORRECTION ON STOCKS DIVISION ??
    dates_df = dates_df_t.copy()
    #dates_df = full_dates_df.copy()
    dates_df.reset_index(inplace = True)
    
    #copy_df = earn_history.copy()
    #earn_history = copy_df.copy()
    
    #Convert numeric columns
    columns_num = ['epsActual','epsEstimate', 'epsDifference', 'surprisePercent']
    for i in columns_num:
        earn_history[i]= earn_history[i].astype(str).replace('None',np.nan)
        earn_history[i] = earn_history[i].apply(pd.to_numeric, errors='coerce')
    
    #convert Dates Columns
    earn_history["date"] = pd.to_datetime(earn_history["date"], format='%Y-%m-%d',errors='coerce')
    earn_history["reportDate"] = pd.to_datetime(earn_history["reportDate"], format='%Y-%m-%d',errors='coerce')
    
    #Deal with duplicate reportDate, when at the same date company reports about several quarters
    earn_history['new_reportDate'] = earn_history.reportDate.duplicated()
    earn_history.loc[earn_history['new_reportDate'] == True, 'reportDate'] = earn_history['reportDate'] + timedelta(days=1)
    earn_history.drop('new_reportDate', axis = 1, inplace = True)
    
    #Fill all nan values with 0
    earn_history.sort_values(by=["date"], ascending = False, inplace = True)
    earn_history.loc[earn_history['epsEstimate'].notnull() & earn_history['epsActual'].isnull(),'epsActual'] = earn_history['epsEstimate']
    earn_history.loc[earn_history['epsActual'].notnull() & earn_history['epsEstimate'].isnull(),'epsEstimate'] = earn_history['epsActual']
    earn_history['epsDifference'] = earn_history['epsActual'] - earn_history['epsEstimate']
    earn_history['surprisePercent'] = 100*earn_history['epsDifference'] / earn_history['epsEstimate']
    
    earn_history = earn_history[earn_history['epsEstimate'].notnull()]
    earn_history = earn_history[['reportDate', 'date', 'surprisePercent','epsActual']]
    
    earn_history_with_dates = dates_df.merge(earn_history, how = 'left', left_on="date", right_on="reportDate")
    earn_history_with_dates.rename(columns={"date_x": "date","date_y": "Q_end_date"}, inplace = True)
    earn_history_with_dates.sort_values(by=["date"], ascending = False, inplace = True)
    earn_history_with_dates[['reportDate','epsActual','surprisePercent']] = earn_history_with_dates[['reportDate','epsActual','surprisePercent']].bfill()
    #earn_history_with_dates['surprisePercent'] = earn_history_with_dates['surprisePercent'].bfill()
    earn_history_with_dates['days_after_report'] = earn_history_with_dates['date'] - earn_history_with_dates['reportDate']
    earn_history_with_dates['days_after_report'] = earn_history_with_dates['days_after_report'].dt.days
    
    return earn_history_with_dates

def earnings_df_prep(earn_history, earn_trend, splits_daily, dates_df, historical_days = 6000, future_days = 750):
    '''
    Combines earnings trend and earnings historical data in one dataframe
    '''
    earn_trend_df = mod_earn_trend (earn_trend, earn_history, splits_daily)
    earn_history_with_dates = mod_earn_history(earn_history, dates_df, historical_days, future_days)
    
    earn_history_with_dates = earn_history_with_dates.merge(earn_trend_df, how = 'left', left_on="Q_end_date", right_on="date")
    earn_history_with_dates.rename(columns={"date_x": "date"}, inplace = True)
    earn_history_with_dates.drop("date_y", axis = 1, inplace = True)
    earn_history_with_dates.sort_values(by=["date"], ascending = True, inplace = True)
    earn_history_with_dates['PG_7d_YoY_shift'] = earn_history_with_dates['PG_7d_YoY'].shift(-7)
    earn_history_with_dates['PG_30d_YoY_shift'] = earn_history_with_dates['PG_30d_YoY'].shift(-30)
    earn_history_with_dates['PG_60d_YoY_shift'] = earn_history_with_dates['PG_60d_YoY'].shift(-60)
    earn_history_with_dates['PG_90d_YoY_shift'] = earn_history_with_dates['PG_90d_YoY'].shift(-89)
    earn_history_with_dates.drop(['PG_7d_YoY','PG_30d_YoY','PG_60d_YoY','PG_90d_YoY'],axis = 1,inplace = True)
    earn_history_with_dates.rename(columns={"PG_7d_YoY_shift": "PG_7d_YoY", "PG_30d_YoY_shift": "PG_30d_YoY", "PG_60d_YoY_shift": "PG_60d_YoY", "PG_90d_YoY_shift": "PG_90d_YoY"}, inplace = True)
    earn_history_with_dates.sort_values(by=["date"], ascending = False, inplace = True)
    #earn_history_with_dates['financial_year'] = earn_history_with_dates['date'].map(lambda x: x.year if x.month <= 9 else x.year+1)
    
    # Creating an additional column with all th
    conditions = [
        earn_history_with_dates['date'] >= datetime.datetime.now(),
        earn_history_with_dates['PG_90d_YoY'].notnull(),
        earn_history_with_dates['PG_60d_YoY'].notnull(),
        earn_history_with_dates['PG_30d_YoY'].notnull(),
        earn_history_with_dates['PG_7d_YoY'].notnull(),
        earn_history_with_dates['PG_YoY'].notnull()
        ]
    
    choices = [
        earn_history_with_dates['PG_YoY'],
        earn_history_with_dates['PG_90d_YoY'],
        earn_history_with_dates['PG_60d_YoY'],
        earn_history_with_dates['PG_30d_YoY'],
        earn_history_with_dates['PG_7d_YoY'],
        earn_history_with_dates['PG_YoY']
        ]
    
    choices_date = [
        np.nan,
        earn_history_with_dates['date'].values.astype(int),
        earn_history_with_dates['date'].values.astype(int),
        earn_history_with_dates['date'].values.astype(int),
        earn_history_with_dates['date'].values.astype(int),
        earn_history_with_dates['date'].values.astype(int)
        ]
    
    earn_history_with_dates["YoY_Proj_Growth_calc"] = np.select(conditions, choices, default=np.nan)
    earn_history_with_dates["forecast_date"] = np.select(conditions, choices_date, default=np.nan)
    earn_history_with_dates["forecast_date"] = earn_history_with_dates["date"][earn_history_with_dates["forecast_date"].notnull()]
    
    earn_history_with_dates['forecast_date'] = earn_history_with_dates['forecast_date'].bfill()
    earn_history_with_dates["forecast_date"] = pd.to_datetime(earn_history_with_dates["forecast_date"], format='%Y-%m-%d',errors='coerce')
    earn_history_with_dates["days_after_forecast_changed"] = earn_history_with_dates["date"] - earn_history_with_dates["forecast_date"]
    earn_history_with_dates["days_after_forecast_changed"] = earn_history_with_dates["days_after_forecast_changed"].dt.days
    earn_history_with_dates['YoY_Proj_Growth_calc'] = earn_history_with_dates['YoY_Proj_Growth_calc'].ffill()
    earn_history_with_dates.rename(columns={"YoY_Proj_Growth_calc": "YoY_Proj_Growth"}, inplace = True)
    
    earn_history_with_dates['prev_growth_forecast_d'] =  earn_history_with_dates[earn_history_with_dates['days_after_forecast_changed'] == 0]['date']
    earn_history_with_dates['prev_growth_forecast_d'] =  earn_history_with_dates['prev_growth_forecast_d'].shift(1)
    earn_history_with_dates['prev_growth_forecast'] = earn_history_with_dates[earn_history_with_dates['prev_growth_forecast_d'].notnull()]['YoY_Proj_Growth']
    earn_history_with_dates['prev_growth_forecast'] =  earn_history_with_dates['prev_growth_forecast'].bfill()
    earn_history_with_dates['forecasts_diff'] = earn_history_with_dates['YoY_Proj_Growth'] - earn_history_with_dates['prev_growth_forecast']
    
    earn_history_with_dates = earn_history_with_dates[['date', 'reportDate', 'Q_end_date', 'surprisePercent','days_after_report','YoY_Proj_Growth','days_after_forecast_changed','forecasts_diff','epsActual']]
    earn_history_with_dates = earn_history_with_dates[earn_history_with_dates['YoY_Proj_Growth'].notnull()]
    earn_history_with_dates["date"] = pd.to_datetime(earn_history_with_dates["date"], format='%Y-%m-%d',errors='coerce')
    
    #copy_df = earn_history_with_dates.copy()
    #Merging long term forecast
    #earn_history_with_dates = earn_history_with_dates.merge(long_forecast, how = 'left', left_on="date", right_on="date")
    
    earn_history_with_dates = earn_history_with_dates[earn_history_with_dates["date"] <= datetime.datetime.now()]
    earn_history_with_dates[earn_history_with_dates['forecasts_diff'].isnull()] = 0
    earn_history_with_dates = earn_history_with_dates[earn_history_with_dates['date'] != 0]
    earn_history_with_dates["date"] = pd.to_datetime(earn_history_with_dates["date"], format='%Y-%m-%d',errors='coerce')
    
    return earn_history_with_dates

def earnings_long_forecast(rqst_earn_trends, splits_daily, earn_trend, earn_history, dates_df_t, historical_days = 6000, future_days = 1000):
    long_forecast = EOD(rqst_earn_trends, transpose = False, json = True)
    long_forecast = long_forecast['trends'][0]
    long_forecast = pd.DataFrame(long_forecast)
    long_forecast = long_forecast[long_forecast['period'] == "+1y"]
    long_forecast.drop(["code","period","earningsEstimateNumberOfAnalysts","revenueEstimateAvg", "revenueEstimateLow", "revenueEstimateHigh", "revenueEstimateYearAgoEps",'revenueEstimateNumberOfAnalysts', 'epsRevisionsUpLast7days','epsRevisionsUpLast30days','epsRevisionsDownLast30days', 'earningsEstimateLow', 'earningsEstimateHigh','growth', 'earningsEstimateGrowth', 'epsTrendCurrent'],axis = 1,inplace = True)
    long_forecast["date"] = long_forecast["date"].apply(pd.to_datetime, errors='coerce')   
    
    columns_num = long_forecast.columns.to_list()
    columns_num.remove("date")
    for i in columns_num:
        long_forecast[i]= long_forecast[i].astype(str).replace('None',np.nan)
        long_forecast[i] = long_forecast[i].apply(pd.to_numeric, errors='coerce')
    
    # thus every record shows a projection that was created a year ago let's chenge the date to the period when projection was made
    long_forecast['date'] = long_forecast['date'] - pd.DateOffset(years=1)
    long_forecast.drop(["earningsEstimateYearAgoEps","revenueEstimateGrowth"], axis = 1, inplace = True)
    
    #We need to use stocks splits to make estamations close in values
    #Creating a dates df
    dates_df = dates_df_t.copy()
    dates_df.reset_index(inplace = True)
    
    splits_daily_df = splits_daily.copy().reset_index()
    
    #Convert earn_history date type
    earn_history[['date','reportDate']] = earn_history[['date','reportDate']].apply(pd.to_datetime, errors='coerce') 
    
    earn_trend_df = earn_trend[earn_trend.period.str.contains("y")][["date","earningsEstimateYearAgoEps"]]
    earn_trend_df['date'] = earn_trend_df['date'] - pd.DateOffset(years=1)
    earn_trend_df = earn_trend_df.merge(earn_history, how = 'left', left_on = "date", right_on = "date")
    earn_trend_df = earn_trend_df[["earningsEstimateYearAgoEps","reportDate"]]
    #flipping years because the split for the historical dates splits are counted
    earn_trend_df['reportDate'] = earn_trend_df['reportDate'] + pd.DateOffset(years=1)
    earn_trend_df.rename(columns={"earningsEstimateYearAgoEps": "real_EPS"}, inplace = True)
    earn_trend_df = earn_trend_df.merge(splits_daily_df, how = 'left', left_on = "reportDate", right_on = "date")
    earn_trend_df['cum_coef'].fillna(1, axis = 0, inplace = True)
    earn_trend_df['reportDate'] = earn_trend_df['reportDate'] - pd.DateOffset(years=1)
    earn_trend_df['real_EPS']  = earn_trend_df['real_EPS'].apply(pd.to_numeric, errors='coerce')
    earn_trend_df['real_EPS'] = earn_trend_df['real_EPS'] * earn_trend_df['cum_coef']
    earn_trend_df = earn_trend_df[["real_EPS","reportDate"]]
    
    #Creating a big table
    long_forecast_conversion = dates_df.merge(splits_daily_df, how = 'left', left_on = "date", right_on = "date")
    long_forecast_conversion = long_forecast_conversion.merge(long_forecast, how = 'left', left_on = "date", right_on = "date")
    long_forecast_conversion = long_forecast_conversion.merge(earn_trend_df, how = 'left', left_on = "date", right_on = "reportDate")
    #long_forecast_conversion_2 = long_forecast_conversion[(long_forecast_conversion.date <= '2020-12-12') & (long_forecast_conversion["real_EPS"].notnull())]
    
    long_forecast_conversion['cum_coef'].fillna(1, axis = 0, inplace = True)
    long_forecast_conversion.sort_values(by=["date"], ascending = True, inplace = True)
    long_forecast_conversion['1Y_frcst_7d_shift'] = long_forecast_conversion['epsTrend7daysAgo'].shift(-7)
    long_forecast_conversion['1Y_frcst_30d_shift'] = long_forecast_conversion['epsTrend30daysAgo'].shift(-30)
    long_forecast_conversion['1Y_frcst_60d_shift'] = long_forecast_conversion['epsTrend60daysAgo'].shift(-60)
    long_forecast_conversion['1Y_frcst_90d_shift'] = long_forecast_conversion['epsTrend90daysAgo'].shift(-90)
    long_forecast_conversion.drop(["epsTrend7daysAgo","epsTrend30daysAgo","epsTrend60daysAgo","epsTrend90daysAgo"], axis = 1, inplace = True)
    long_forecast_conversion.rename(columns={"earningsEstimateAvg": "1Y_frcst_AVG"}, inplace = True)
    
    #Conversion
    long_forecast_conversion['1Y_frcst_AVG'] = long_forecast_conversion['1Y_frcst_AVG'] * long_forecast_conversion['cum_coef']
    long_forecast_conversion['1Y_frcst_7d_shift'] = long_forecast_conversion['1Y_frcst_7d_shift'] * long_forecast_conversion['cum_coef']
    long_forecast_conversion['1Y_frcst_30d_shift'] = long_forecast_conversion['1Y_frcst_30d_shift'] * long_forecast_conversion['cum_coef']
    long_forecast_conversion['1Y_frcst_60d_shift'] = long_forecast_conversion['1Y_frcst_60d_shift'] * long_forecast_conversion['cum_coef']
    long_forecast_conversion['1Y_frcst_90d_shift'] = long_forecast_conversion['1Y_frcst_90d_shift'] * long_forecast_conversion['cum_coef']
    
    conditions = [
        long_forecast_conversion['date'] >= datetime.datetime.now(),
        long_forecast_conversion['1Y_frcst_90d_shift'].notnull(),
        long_forecast_conversion['1Y_frcst_60d_shift'].notnull(),
        long_forecast_conversion['1Y_frcst_30d_shift'].notnull(),
        long_forecast_conversion['1Y_frcst_7d_shift'].notnull(),
        long_forecast_conversion['1Y_frcst_AVG'].notnull()
        ]
    
    choices = [
        long_forecast_conversion['1Y_frcst_AVG'],
        long_forecast_conversion['1Y_frcst_90d_shift'],
        long_forecast_conversion['1Y_frcst_60d_shift'],
        long_forecast_conversion['1Y_frcst_30d_shift'],
        long_forecast_conversion['1Y_frcst_7d_shift'],
        long_forecast_conversion['1Y_frcst_AVG']
        ]
    
    long_forecast_conversion["1Y_frcst"] = np.select(conditions, choices, default=np.nan)
    long_forecast_conversion["1Y_frcst"] = long_forecast_conversion["1Y_frcst"].bfill()
    long_forecast_conversion["real_EPS"] = long_forecast_conversion["real_EPS"].ffill()
    long_forecast_conversion["1Y_expect_Change"] = 100*(long_forecast_conversion["1Y_frcst"] - long_forecast_conversion["real_EPS"])/long_forecast_conversion["real_EPS"]
    long_forecast_conversion = long_forecast_conversion[["1Y_expect_Change","date"]]
    long_forecast_conversion = long_forecast_conversion[long_forecast_conversion["1Y_expect_Change"].notnull()]
    
    return long_forecast_conversion

def clear_prices (prices, splits_daily):
    '''
    Replace the stock trading prices with the adjusted in regards to the splits only.
    Avoiding reevaluation by dividends, buy backs and so on
    '''
    prices["date"] = prices["date"].apply(pd.to_datetime, errors='coerce')
    prices.set_index(["date"], inplace = True)
    prices_updated = splits_daily.join(prices)
    prices_updated["open_adj"] = prices_updated["open"]*prices_updated["cum_coef"]
    prices_updated["close_adj"] = prices_updated["close"]*prices_updated["cum_coef"]
    prices_updated["low_adj"] = prices_updated["low"]*prices_updated["cum_coef"]
    prices_updated["high_adj"] = prices_updated["high"]*prices_updated["cum_coef"]
    prices_updated = prices_updated[["open_adj", "close_adj","low_adj","high_adj", "volume"]]
    prices_updated.rename(columns={"open_adj": "open","close_adj":"close","low_adj":"low","high_adj":"high"}, inplace = True)
    
    prices_updated.sort_index(axis = 0, ascending = True, inplace = True)
    prices_updated['Price_SMA50'] = prices_updated['close'].rolling(50).mean()
    prices_updated['Price_SMA200'] = prices_updated['close'].rolling(200).mean()
    prices_updated['Price_SMA50_vector'] = 100*prices_updated['Price_SMA50'].pct_change(periods = 1)
    prices_updated['Price_SMA200_vector'] = 100*prices_updated['Price_SMA50'].pct_change(periods = 1)
    prices_updated['Price_SMA50_position'] = (prices_updated['Price_SMA50']-prices_updated["close"])/prices_updated["close"]
    prices_updated['Price_SMA200_position'] = (prices_updated['Price_SMA200']-prices_updated["close"])/prices_updated["close"]
    prices_updated['Price_MA50_to_200_position'] = (prices_updated['Price_SMA50']-prices_updated["Price_SMA200"])/prices_updated["Price_SMA200"]
    prices_updated.sort_index(axis = 0, ascending = False, inplace = True)
    
    prices_updated.drop(["Price_SMA50","Price_SMA200"], axis = 1, inplace  = True)
    
    return prices_updated

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

def get_data(ticker, rqst_prices, rqst_splits, rqst_erng_history, rqst_erng_trend, rqst_fin_balance, rqst_fin_cashflow, rqst_fin_income, rqst_general, rqst_stocks_number):
    #all_query = EOD(rqst_all,transpose = False, json = True)

    # EARNINGS
    earn_history = EOD(rqst_erng_history,transpose = True, json = False)
    
    earn_trend = EOD(rqst_erng_trend,transpose = True, json = False)
    if earn_trend.empty:
        print (ticker, " doesn't have the earnings Trend")
    else:
        earn_trend["date"] = pd.to_datetime(earn_trend["date"], format='%Y-%m-%d',errors='coerce')

    # BALANCE
    balance = EOD(rqst_fin_balance, transpose = True, json = False)
    if balance.empty:
        print(f"Stock {ticker} doesn't have any financial data")
        prices, splits, earn_history, earn_trend, balance, cashflow, income, general, stocks_amount = 1, 2, 3, 4, balance, 6,7,8,9,10
        return prices, splits, earn_history, earn_trend, balance, cashflow, income, general, stocks_amount
    else:
        balance['totalLiabilities'] = balance['totalCurrentLiabilities']+balance['nonCurrentLiabilitiesTotal']
    balance[["date","filing_date"]] = balance[["date","filing_date"]].apply(pd.to_datetime, errors='coerce')
    balance = balance[balance.filing_date.notnull()]
    balance.sort_index(axis = 0, inplace = True, ascending = False)
    balance.drop_duplicates('filing_date', keep = 'last', inplace = True)
    
    #CASHFLOW
    cashflow = EOD(rqst_fin_cashflow, transpose = True, json = False)
    #dropping the duplicate columns with the balance
    cashflow.drop(["currency_symbol"], axis = 1, inplace = True)
    cashflow[["date","filing_date"]] = cashflow[["date","filing_date"]].apply(pd.to_datetime, errors='coerce')
    cashflow = cashflow[cashflow.filing_date.notnull()]
    cashflow.sort_index(axis = 0, inplace = True, ascending = False)
    cashflow.drop_duplicates('filing_date', keep = 'last', inplace = True)
    #INCOME
    income = EOD(rqst_fin_income, transpose = True, json = False)
    #dropping the duplicate columns with the cashflow
    income.drop(["currency_symbol", "netIncome"], axis = 1, inplace = True)
    income[["date","filing_date"]] = income[["date","filing_date"]].apply(pd.to_datetime, errors='coerce')
    income = income[income.filing_date.notnull()]
    income.sort_index(axis = 0, inplace = True, ascending = False)
    income.drop_duplicates('filing_date', keep = 'last', inplace = True)
    # OTHER
    general = EOD(rqst_general, transpose = True, json = False)
    general = general.reset_index().iloc[:,:2]
    general.set_index("index", inplace = True)
    
    #insiders_trade = EOD(rqst_insiders_trade, transpose = True, json = False)
    stocks_amount = EOD(rqst_stocks_number, transpose = True, json = False)
    
    #Get market prices
    prices = EOD(rqst_prices, transpose = False, json = False)
    #Get shares splits
    splits = EOD(rqst_splits, transpose = False, json = False)
    return prices, splits, earn_history, earn_trend, balance, cashflow, income, general, stocks_amount

def build_general_df(prices_updated_t, income, balance, cashflow, earnings, general):
    
    
    
    
    #prices_updated_t = prices_updated.copy()
    prices_updated = prices_updated_t.copy()
    prices_updated.reset_index(inplace = True)
    general_df = pd.DataFrame()
    general_df = prices_updated.merge(cashflow, how = 'left', left_on = "date", right_on = "filing_date")
    general_df.rename(columns={"date_x": "date","date_y":"date_cashflow","filing_date":"filing_date_cashflow"}, inplace = True)
    general_df = general_df.merge(income, how = 'left', left_on = "date", right_on = "filing_date")
    general_df.rename(columns={"date_x": "date","date_y":"date_income","filing_date":"filing_date_income"}, inplace = True)
    general_df = general_df.merge(balance, how = 'left', left_on = "date", right_on = "filing_date")
    general_df.rename(columns={"date_x": "date","date_y":"date_balance","filing_date":"filing_date_balance"}, inplace = True)
    remove_columns = prices_updated.columns.to_list()
    columns = general_df.columns.tolist()
    columns.sort()
    for i in remove_columns:
        columns.remove(i)
    general_df.sort_values(by = "date", axis = 0, ascending = False, inplace = True)
    general_df[columns] = general_df[columns].fillna(method="bfill")
    
    general_df.fillna(0, inplace = True)
    remove_columns = ["date","filing_date_cashflow","filing_date_income","filing_date_balance","date_cashflow","date_income","date_balance","currency_symbol"]
    columns = general_df.columns.tolist()
    columns.sort()
    for i in remove_columns:
        columns.remove(i)
    #general_df.convert_dtypes()
    general_df[columns] = general_df[columns].apply(pd.to_numeric, errors='coerce')
    del columns
    
    general_df[["date","filing_date_cashflow"]] = general_df[["date","filing_date_cashflow"]].apply(pd.to_datetime, errors='coerce')
    general_df["days_after_report"] = general_df['date'] - general_df["filing_date_cashflow"]    
    general_df["days_after_report"] = general_df["days_after_report"].dt.days
    
    #Categorical Vars
    general_df["sector"] = general.loc["GicSector"].Street
    general_df["industry"] = general.loc["GicGroup"].Street
    #Return on Equity
    general_df["ROE"] = general_df["netIncome"].divide(general_df["totalAssets"], fill_value=0)

    #Long-Term Debt to Equity
    general_df["LTDE"] = general_df["longTermDebt"]/(general_df["totalAssets"]-general_df["shortLongTermDebtTotal"])

    
    #Debt-to-Equity Ratio
    general_df["DE"] = (general_df["shortTermDebt"]+general_df["longTermDebt"])/(general_df["totalAssets"]-general_df["shortLongTermDebtTotal"])

    
    #Current Ratio
    general_df["CR"] = general_df["totalCurrentAssets"]/general_df["totalCurrentLiabilities"]
    
    #Gross Margin
    general_df["GM"] = (general_df["totalRevenue"] - general_df["costOfRevenue"])/ general_df["totalRevenue"]
    #Return on Asset
    general_df["ROA"] = general_df["netIncome"]/general_df["totalAssets"]
    #Dividend Payout Ratio
    general_df["DPR"] = np.absolute(general_df["dividendsPaid"])/general_df["netIncome"]
    #Accounts Receivable Turnover Ratio
    general_df["Acc_Rec_Pay_Ration"] = general_df["netReceivables"]/general_df["accountsPayable"]
    #Earnings per dollar of stock
    general_df["ES"] =  100*general_df["netIncome"]/general_df['commonStockSharesOutstanding']/ general_df["close"]
    #Dividends Yield
    general_df["DY"] = 4*100*np.absolute(general_df["dividendsPaid"])/general_df['commonStockSharesOutstanding']/ general_df["close"]
    #Piotrovski_Score
    general_df["P_netIncome"] = np.where(general_df['netIncome'] >= 0, 1, 0)
    general_df["P_ROA"] = np.where(general_df['ROA'] >= 0, 1, 0)
    general_df["P_totalCashFromOperatingActivities"] = np.where(general_df['totalCashFromOperatingActivities'] >= 0, 1, 0)
    general_df["P_tcp"] = np.where(general_df['totalCashFromOperatingActivities'] >= general_df['netIncome'], 1, 0)
    general_df.set_index("date", inplace = True)
    general_df["P_longTermDebt"] = np.where(general_df['longTermDebt'] <= general_df["longTermDebt"].shift(-365), 1, 0)
    general_df["P_CR"] = np.where(general_df['CR'] >= general_df["CR"].shift(-365), 1, 0)
    general_df["P_GM"] = np.where(general_df['GM'] >= general_df["GM"].shift(-365), 1, 0)
    general_df["asset_turnover"] = general_df["totalRevenue"] / general_df["totalAssets"]
    general_df["P_asset_turnover"] = np.where(general_df['asset_turnover'] >= general_df["asset_turnover"].shift(-365), 1, 0)
    general_df["P_issuanceOfCapitalStock"] = np.where(general_df['issuanceOfCapitalStock'] > 0, 0, 1)
    general_df["Piotroski_Score"] = general_df["P_netIncome"] + general_df["P_ROA"] +general_df["P_totalCashFromOperatingActivities"] +general_df["P_tcp"] +general_df["P_longTermDebt"] +general_df["P_CR"] +general_df["P_GM"] +general_df["P_asset_turnover"] +general_df["P_issuanceOfCapitalStock"]
    general_df.drop(["P_netIncome","P_ROA", "P_totalCashFromOperatingActivities", "P_tcp", "P_longTermDebt", "P_CR", "P_GM", "P_asset_turnover","P_issuanceOfCapitalStock"], axis = 1, inplace = True)
    #PE
    earnings.set_index('date', inplace = True)
    earnings.rename(columns={"days_after_report": "days_after_earnings_report"}, inplace = True)
    general_df = general_df.join(earnings)
    
    general_df['PE'] = general_df['close']/(4*general_df['epsActual'])

    #PB
    general_df['Book_Val_Share'] = (general_df['totalAssets'] - general_df['totalLiab']) / general_df['commonStockSharesOutstanding'] #
    general_df['PB'] = general_df['close'] / general_df['Book_Val_Share']
    #PEG backwards and forward
    general_df['PEG_Forward'] = general_df['PE']*(1+general_df['1Y_expect_Change'])
    general_df['PEG_Backwards'] = general_df['PE']*(1+general_df['YoY_Proj_Growth'])
    
    general_df.rename(columns={"surprisePercent": "EPS_surprise",
                               "YoY_Proj_Growth": "EPS_YoY_Growth",
                               "forecasts_diff": "EPS_QoQ_frcst_diff",
                               "1Y_expect_Change": "EPS_1Y_exp_Change"}, inplace = True)
    
    columns_to_keep = ['days_after_earnings_report',
                       'open',
                       'close',
                       'sector',
                       'industry',
                       'ROE',
                       'LTDE',
                       'DE',
                       'CR',
                       'GM',
                       'ROA',
                       'DPR',
                       'Acc_Rec_Pay_Ration',
                       'ES',
                       'DY',
                       'Piotroski_Score',
                       'PE',
                       'PB',
                       'PEG_Forward',
                       'PEG_Backwards',
                       'EPS_surprise',
                       'EPS_YoY_Growth',
                       'EPS_QoQ_frcst_diff',
                       'EPS_1Y_exp_Change']
    
    general_df = general_df[columns_to_keep]
    
    remove_cols_from_list = ['days_after_earnings_report','open','close','EPS_surprise','EPS_YoY_Growth','EPS_QoQ_frcst_diff','EPS_1Y_exp_Change']
    for i in remove_cols_from_list:
        columns_to_keep.remove(i)
    
    general_df[columns_to_keep] = general_df[columns_to_keep].replace([np.inf, 0, -np.inf], np.nan)
    general_df[columns_to_keep] = general_df[columns_to_keep].bfill()
       
    #Year over Year change of the kritical KPIs to undesrstand how a company performs in time
    general_df['YoY_ROE'] = 100*general_df['ROE'].pct_change(periods = -365)
    general_df['YoY_LTDE'] = 100*general_df['LTDE'].pct_change(periods = -365)
    general_df['YoY_DE'] = 100*general_df['DE'].pct_change(periods = -365)
    general_df['YoY_CR'] = 100*general_df['CR'].pct_change(periods = -365)
    general_df['YoY_GM'] = 100*general_df['GM'].pct_change(periods = -365)
    general_df['YoY_ROA'] = 100*general_df['ROA'].pct_change(periods = -365)
    general_df['YoY_DPR'] = 100*general_df['DPR'].pct_change(periods = -365)
    general_df['YoY_AR_Ration'] = 100*general_df['Acc_Rec_Pay_Ration'].pct_change(periods = -365)
    general_df['YoY_ES'] = 100*general_df['ES'].pct_change(periods = -365)
    general_df['YoY_Piotroski'] = 100*general_df['Piotroski_Score'].pct_change(periods = -365)
    general_df['YoY_PE'] = 100*general_df['PE'].pct_change(periods = -365)
    general_df['YoY_PB'] = 100*general_df['PB'].pct_change(periods = -365)
    general_df['YoY_PEGF'] = 100*general_df['PEG_Forward'].pct_change(periods = -365)
    general_df['YoY_PEGB'] = 100*general_df['PEG_Backwards'].pct_change(periods = -365)
    general_df['YoY_ROE'] = 100*general_df['ROE'].pct_change(periods = -365)
    general_df['YoY_DY'] = 100*general_df['DY'].pct_change(periods = -365)
    
    general_df = general_df[general_df.index > '2000-01-01']
    
    #A change of predicted earnings 1 year ahead
    general_df['EPS_1Y_exp_Change_QoQ1'] = general_df['EPS_1Y_exp_Change'].shift(1)
    general_df.loc[general_df['EPS_1Y_exp_Change_QoQ1'].isnull(),'EPS_1Y_exp_Change_QoQ1'] = general_df['EPS_1Y_exp_Change']
    general_df['EPS_1Y_exp_Change_QoQ'] = general_df['EPS_1Y_exp_Change'] - general_df['EPS_1Y_exp_Change_QoQ1']
    general_df.loc[general_df['EPS_1Y_exp_Change_QoQ'] == 0,'EPS_1Y_exp_Change_QoQ'] = np.nan
    general_df['EPS_1Y_exp_Change_QoQ'] = general_df['EPS_1Y_exp_Change_QoQ'].bfill()
    general_df.drop('EPS_1Y_exp_Change_QoQ1', axis = 1, inplace = True)
    
    general_df = general_df[general_df['PE'].notnull()]
    general_df = general_df[general_df['close'] != 0]
      
    general_df["future_15dprice_change"] = 100*(general_df["close"].shift(11) / general_df["open"] - 1)
    general_df["future_30dprice_change"] = 100*(general_df["close"].shift(22) / general_df["open"] - 1)
    general_df["future_60dprice_change"] = 100*(general_df["close"].shift(44) / general_df["open"] - 1)
    general_df["future_90dprice_change"] = 100*(general_df["close"].shift(66) / general_df["open"] - 1)
    general_df["future_120dprice_change"] = 100*(general_df["close"].shift(88) / general_df["open"] - 1)
    general_df["future_150dprice_change"] = 100*(general_df["close"].shift(110) / general_df["open"] - 1)
    
    return general_df

path = os.path.join(script_location,'Script_intermediate_data')

def read_one_time_data(path):
    symbols = []
    #Recording variable to a file
    path_symbols = os.path.join(path,'symbols.txt')
    with open(path_symbols, 'r') as fp:
        for line in fp:
            # remove linebreak from a current name
            # linebreak is the last character of each line
            x = line[:-1]
            # add current item to the list
            symbols.append(x)

    #get VIX Volatility index
    path_vix = os.path.join(path,'VIX.csv')
    vix_df = pd.read_csv(path_vix)
    vix_df["date"] = vix_df["date"].apply(pd.to_datetime, errors='coerce')
    vix_df.set_index("date", inplace = True)  
    vix_df.squeeze(axis=0)

    #Get the dataframe with the daily historical and future days
    path_dates = os.path.join(path,'full_dates.csv')
    full_dates_df = pd.read_csv(path_dates)
    full_dates_df["date"] = full_dates_df["date"].apply(pd.to_datetime, errors='coerce')
    full_dates_df.set_index("date", inplace = True)  

    #Get the US Federal reserve balance Sheet
    path_Fed = os.path.join(path,'fed_reserve.csv')
    fed_reserve_movements = pd.read_csv(path_Fed)
    fed_reserve_movements["date"] = fed_reserve_movements["date"].apply(pd.to_datetime, errors='coerce')
    fed_reserve_movements.set_index("date", inplace = True)  

    #Get 10Y US Gov Bond Yield
    path_bonds = os.path.join(path,'10Y_bonds.csv')
    bond_US = pd.read_csv(path_bonds)
    bond_US["date"] = bond_US["date"].apply(pd.to_datetime, errors='coerce')
    bond_US.set_index("date", inplace = True)  

    #get last crisis date data
    path_crisis = os.path.join(path,'crisises.csv')
    last_crisis = pd.read_csv(path_crisis)
    last_crisis["date"] = last_crisis["date"].apply(pd.to_datetime, errors='coerce')
    last_crisis.set_index("date", inplace = True)  
    
    return symbols, vix_df, full_dates_df, fed_reserve_movements, bond_US, last_crisis
          
symbols, vix_df, full_dates_df, fed_reserve_movements, bond_US, last_crisis = read_one_time_data(path)

def send_email():
    # Sending email
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    import smtplib
    import sys
    
    
    #The mail addresses and password
    sender_address = 'oleg.kazanskyi@gmail.com'
    sender_pass = 'mjmchacmlqapfgii'#input()
    receiver_address = 'oleg.kazanskyi@gmail.com'
    #Setup the MIME

    part1 = "VIX Problem Again"
    part1 = MIMEText(part1)
    message = MIMEMultipart()
    message['From'] = sender_address
    message['To'] = receiver_address
    message['Subject'] = 'Go and check the last date in the VIX table' 
    
    
    #The subject line
    #The body and the attachments for the mail
    message.attach(part1)
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, receiver_address, text)
    session.quit()
    print('Mail Sent')

#symbols = ["SWKS",'SPG','SBNY','SHW','NOW','SRE','SEE']
def execution (stock):
    #stock = 'ABBV'
    stock = stock.replace(".","-")
    rqst_all = f'https://eodhistoricaldata.com/api/fundamentals/{stock}.US?api_token={EOD_APi}'
    rqst_prices = f'https://eodhistoricaldata.com/api/eod/{stock}.US?period=d&fmt=json&api_token={EOD_APi}'
    rqst_splits = f'https://eodhistoricaldata.com/api/splits/{stock}.US?fmt=json&from=2000-01-01&api_token={EOD_APi}'
    rqst_erng_history = rqst_all+'&filter=Earnings::History'
    rqst_erng_trend = rqst_all+'&filter=Earnings::Trend'
    rqst_fin_balance = rqst_all+'&filter=Financials::Balance_Sheet::quarterly'
    rqst_fin_cashflow = rqst_all+'&filter=Financials::Cash_Flow::quarterly'
    rqst_fin_income = rqst_all+'&filter=Financials::Income_Statement::quarterly'
    rqst_general = rqst_all+'&filter=General'
    rqst_insiders_trade = rqst_all+'&filter=InsiderTransactions'
    rqst_stocks_number = rqst_all+'&filter=outstandingShares::quarterly'
    
    rqst_earn_trends = f'https://eodhistoricaldata.com/api/calendar/trends?api_token={EOD_APi}&fmt=json&symbols={stock}.US'
        
    prices, splits, earn_history, earn_trend, balance, cashflow, income, general, stocks_amount = get_data(stock, rqst_prices, rqst_splits, rqst_erng_history, rqst_erng_trend, rqst_fin_balance, rqst_fin_cashflow, rqst_fin_income, rqst_general, rqst_stocks_number)
    if balance.empty:
        return
    del rqst_all, rqst_prices, rqst_splits, rqst_erng_history, rqst_erng_trend, rqst_fin_balance, rqst_fin_cashflow, rqst_fin_income, rqst_general, rqst_insiders_trade, rqst_stocks_number
    
    if earn_trend.empty:
        return
    
    splits_daily = get_splits(splits, full_dates_df, historical_days = historical_number_of_days)
    earnings = earnings_df_prep(earn_history, earn_trend, splits_daily, full_dates_df, historical_days = historical_number_of_days, future_days = future_number_of_days)
    long_forecast_conversion = earnings_long_forecast(rqst_earn_trends, splits_daily, earn_trend, earn_history, full_dates_df, historical_days = 6000, future_days = 1000)
    earnings = earnings.merge(long_forecast_conversion, how = 'left', left_on = "date", right_on = "date")
    
    #CHECK THIS LINE
    earnings['1Y_expect_Change'] = earnings['1Y_expect_Change'].bfill()
    
    # surprise percent - EPS estimates compared to real EPS
    # days after report - Number of dates after real EPS report was published
    # YoY Proj Growth - forecasted Quarter results changes compared to the previous year
    # days after forecast changed - Number of days after forecast for a quarter changed 
    # forecasts diff - a difference between the latest forecast for the next quarter and the previous one
    # epsActual - quarter results EPS
    # 1Y expect Change - forecasted EPS YoY 1 year ahead
    #del rqst_earn_trends, earn_history, earn_trend
    
    prices_updated = clear_prices (prices, splits_daily)
    #del splits_daily, prices
    
    general_df = build_general_df(prices_updated, income, balance, cashflow, earnings, general)
    del balance, income, cashflow, prices_updated
    del earnings, general, long_forecast_conversion, splits, stocks_amount
    
    #Adding External Factors to the dataset
    general_df = general_df.join(vix_df)
    general_df = general_df.join(last_crisis)
    general_df['VIX_DoD'] =  100*general_df['VIX_high'].pct_change(periods = -1)
    general_df['VIX_WoW'] =  100*general_df['VIX_high'].pct_change(periods = -5)
    general_df['VIX_MoM'] =  100*general_df['VIX_high'].pct_change(periods = -22)
    
    general_df['stock'] =  stock
    
    general_df = general_df.join(bond_US)
    
    general_df = general_df.join(fed_reserve_movements)
    
    csv_file = stock + ".csv"
    path_to_all_data = os. path. join(script_location, "SP500_CSVs")
    path_to_all_data = os. path. join(path_to_all_data, csv_file)
    general_df.to_csv(path_to_all_data)
     
def multiprocess_run(symbols):
    __spec__ = None
    # Step 1: Init multiprocessing.Pool()
    pool = mp.Pool(cores)
    # Step 2: `pool.apply` the `howmany_within_range()`
    res = pool.map(execution, symbols)
    # Step 3: Don't forget to close
    pool.close()  

#symbols = ["AAPL"]
        
                         
def dual_axis(df, col1_name, col2_name, col3_name):
    COLOR_1 = "#69b3a2"
    COLOR_2 = "#3399e6"
    COLOR_3 = "#e29b8c"
    
    fig, ax1 = plt.subplots(figsize=(60, 30))
    ax2 = ax1.twinx()
    
    if len(col3_name)>0:
        ax1.plot(df.index, df[col1_name], color=COLOR_1, lw=3)
        ax1.plot(df.index, df[col3_name]/20000, color=COLOR_3, lw=3)
    else:
        ax1.plot(df.index, df[col1_name], color=COLOR_1, lw=3)
        
    ax2.plot(df.index, np.zeros(len(df)), color='red', lw=2)
    ax2.plot(df.index, df[col2_name], color=COLOR_2, lw=4)
    
    ax1.set_xlabel("Date", fontsize=20)
    ax1.set_ylabel(col1_name, color=COLOR_1, fontsize=14)
    ax1.tick_params(axis="y", labelcolor=COLOR_1)
    ax1.tick_params(axis="x", labelsize=25)
    
    ax2.set_ylabel(col2_name, color=COLOR_2, fontsize=14)
    ax2.tick_params(axis="y", labelcolor=COLOR_2)
    
    fig.suptitle(f"{col1_name} and {col2_name}", fontsize=20)
    fig.autofmt_xdate()

#plotting a sample to see how data performs
'''df = pd.read_csv("C:\\Users\\oleg.kazanskyi\\Documents\\Trading\\SP500_CSVs_01032023\\AAPL.csv")
df["date"] = df["date"].apply(pd.to_datetime, errors='coerce')
df.set_index("date", inplace = True)
df[["close","Price_SMA200"]].plot()'''

#dual_axis(df.iloc[:-300,:], 'close', 'EPS_1Y_exp_Change')

if __name__ == '__main__':  
    
    #This part checks if the latest available VIX date equal to the last trading date
    #We need only the latest date to get evaluation by our model
    if date.today().weekday() == 0:
        delta = 3
    else:
        delta = 1
        
    yesterday = date.today() - timedelta(days=delta)
    if vix_df.index.date.max() != yesterday:
        send_email()
    else:
        '''for stock in symbols[:]:
            execution(stock)'''
            
        mp.freeze_support()
        __spec__ = None
        multiprocess_run(symbols)
