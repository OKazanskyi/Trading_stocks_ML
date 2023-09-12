# -*- coding: utf-8 -*-
"""
Created on Mon Jul  4 14:07:24 2022
This script pulls the list of S&P500 companies

@author: oleg.kazanskyi
"""

import pandas as pd # library for data analysis
import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents

def get_sp500_symbols():
    # get the response in the form of html
    wikiurl="https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks"
    table_class="wikitable sortable jquery-tablesorter"
    response=requests.get(wikiurl)
    print(response.status_code)
    
    # parse data from the html into a beautifulsoup object
    soup = BeautifulSoup(response.text, 'html.parser')
    indiatable=soup.find('table',{'class':"wikitable"})
    
    df=pd.read_html(str(indiatable))
    # convert list to dataframe
    df=pd.DataFrame(df[0])
    print(df.head())
    
    symbols = df.Symbol.tolist()
    return symbols
