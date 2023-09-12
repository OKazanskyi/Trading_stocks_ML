# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 14:37:16 2023

@author: oleg.kazanskyi
THis script lauchens 3 stages of the stocks "buys" evaluation
1. Get all the data that is not related to a single stock, and store it incsv
2. Get all the data related to stocks, process and transform data and save to csvs
3. Read the final CSVs, 
"""
import os

#Set the proper location
if os.name == 'posix':
    #OS
    os.chdir("/Users/olegkazanskyi/Documents/GitHub/Trading")
else:
    #Windows
    os.chdir("C:\\Users\\oleg.kazanskyi\\OneDrive - Danaher\\Documents\\Trading")

script_location = os.getcwd()

# Run step 1
import Dates_FedRes_Bonds

#Run step 2
import eodhistoricaldata

path = os.path.join(script_location,'Script_intermediate_data')
symbols, vix_df, full_dates_df, fed_reserve_movements, bond_US, last_crisis = eodhistoricaldata.read_one_time_data(path)

'''import multiprocessing as mp
if __name__ == '__main__':
    mp.freeze_support()
    __spec__ = None
    eodhistoricaldata.multiprocess_run(symbols)
    
for stock in symbols[:]:
    eodhistoricaldata.execution(stock)

#Run step 3
import Models.Apply_ML_Send_Email'''
