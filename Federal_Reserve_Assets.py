# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 10:07:00 2023

@author: oleg.kazanskyi
This is to get Federal Reserve Total Assets
"""

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup


def download_tables():
    '''
    Downloads 10-Y bond historical daily data
    '''
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('https://www.federalreserve.gov/monetarypolicy/bst_recenttrends_accessible.htm')
    time.sleep(5)
    page_source = driver.page_source
    
    #Calculating current and historical periods
    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(page_source, 'html.parser')
    indiatable=soup.find('table',{'class':"pubtables"})
    
    df=pd.read_html(str(indiatable))
    # convert list to dataframe
    df=pd.DataFrame(df[0])
    df.Date = pd.to_datetime(df.Date, format='%d-%b-%Y')
    #df.Date = df.Date.dt.date
    driver.close()
    driver.quit()
   
    return df
    
