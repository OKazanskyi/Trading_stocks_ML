#!/usr/bin/env python
# coding: utf-8

# Import the required libraries

import warnings
warnings.filterwarnings('ignore')

import sys
import numpy as np
import pandas as pd
import os
import time
import lightgbm as lgb # Our ML library
from datetime import date
from datetime import timedelta
import joblib

SET PYTHONPATH = "C:/Users/oleg.kazanskyi/Documents/Trading"

if os.name == 'posix':
    #OS
    sys.path.insert(1,"/Users/olegkazanskyi/Documents/GitHub/Trading")
else:
    #Windows
    sys.path.insert(1, "C:/Users/oleg.kazanskyi/Documents/Trading")


# Read all CSVs with stocks data and append to one big file
# I've uploaded the latest SP500 data. Let's read it

def get_latest_data():
    #os.chdir("/Users/olegkazanskyi/Documents/GitHub/Trading/CSVs")
    if os.name == 'posix':
        #OS
        os.chdir("/Users/olegkazanskyi/Documents/GitHub/Trading/SP500_CSVs_01032023")
    else:
        #Windows
        os.chdir("C:\\Users\\oleg.kazanskyi\\Documents\\Trading\\SP500_CSVs_01032023")
        
    filepaths = [f for f in os.listdir("./") if f.endswith('.csv')]
    df = pd.DataFrame()
    for i in filepaths:
        iterate_df = pd.DataFrame()
        iterate_df = pd.read_csv(i, encoding= 'unicode_escape')
        iterate_df["stock"] = i[:-4]
        df = pd.concat([df,iterate_df])
    
    df = df[df.close.notna()]
    
    #Convert date to datetime type
    df["date"] = pd.to_datetime(df["date"])
    
    #We need only the latest date to get evaluation by our model
    if date.today().weekday() == 0:
        delta = 3
    else:
        delta = 1
        
    #delta = 4
        
    today = date.today()
    yesterday = today - timedelta(days=delta)
    yesterday = yesterday.strftime(format = '%Y-%m-%d')
    #today = date.today().strftime(format = '%Y-%m-%d')
    df = df[df.date == yesterday]
    df.drop(["date"], axis = 1, inplace = True)
    
    print("1. Number of stocks: ",len(df.stock.unique()))
    # Dealing with "EPS" column
    df = df[df.EPS_surprise.notnull()]
    print("2. Number of stocks: ",len(df.stock.unique()))
    
    # Dealing with the null values in the "dividends" column
    #This companies do not pay dividends, we can replace payments to "0"
    df['DY'].fillna(0, inplace=True)
    df['YoY_DPR'].fillna(0, inplace=True)
    df['DPR'].fillna(0, inplace=True)
    df['YoY_DY'].fillna(0, inplace=True)      
    
    # Dealing with YoY
    #The YoY variables with null values are caused by YoY calculation of the rows without historical data.
    #Let's drop these
    df = df[df.YoY_CR.notnull()]
    print("3. Number of stocks: ",len(df.stock.unique()))
    
    # Dealing with debt ratio
    #It seems these companies had zero debt for some quarters.
    #Let's replace with "0"
    df['DE'].fillna(0, inplace=True)
    df['LTDE'].fillna(0, inplace=True)
    
    # Dealing with VIX
    #Let's remove null values for VIX as these are related to the historical calculations
    df = df[df.VIX_MoM.notnull()]
    print("4. Number of stocks: ",len(df.stock.unique()))
    
    #Dealing with Accounts Payable
    #It seems these companies had zero Accounts receivables for some quarters
    df['Acc_Rec_Pay_Ration'].fillna(0, inplace=True)
    
    
    # Dealing with PEG
    df = df[df.PEG_Forward.notnull()]
    print("5. Number of stocks: ",len(df.stock.unique()))
    df = df[df.PEG_Backwards.notnull()]
    print("6. Number of stocks: ",len(df.stock.unique()))
    df = df[df.EPS_1Y_exp_Change.notnull()]
    print("7. Number of stocks: ",len(df.stock.unique()))
    
    # Dealing with Industry
    df.loc[df['stock'] == 'GEN','sector'] = 'Information Technology'
    df.loc[df['stock'] == 'GEN','industry'] = 'Software & Services'
    
    df = df[df.YoY_AR_Ration.notnull()]
    print("8. Number of stocks: ",len(df.stock.unique()))
    
    #Let's also check if there are any infinite numbers that can cause same trublesas nan
    df.replace([np.inf, -np.inf], np.nan, inplace = True)
    df = df[df.EPS_1Y_exp_Change.notnull() & df.EPS_YoY_Growth.notnull() & df.EPS_QoQ_frcst_diff.notnull()]
    print("9. Number of stocks: ",len(df.stock.unique()))
    
    df = df[df['Fed_Balance_YoY'].notnull()]
    print("10. Number of stocks: ",len(df.stock.unique()))
    
    df.drop(['close','open'], axis = 1, inplace = True)
    
    df.drop(["industry","days_after_crisis"], axis = 1, inplace = True)
    
    df.drop(["future_15dprice_change","future_30dprice_change","future_60dprice_change","future_90dprice_change","future_120dprice_change","future_150dprice_change"], axis = 1, inplace = True)
    
    #Let's reset index to understand the earliest and the latest records
    df.reset_index(inplace = True)
    df.drop('index', axis = 1, inplace = True)
    
    return df

def ML_classification(df_input, stocks, period = '30'):
    #Load the model
    df = df_input.copy()
    #df = df_usage.copy()
    if os.name == 'posix':
        #OS
        model_path = f"/Users/olegkazanskyi/Documents/GitHub/Trading/Models/{period}d_model.joblib"
    else:
        #Windows
        model_path = f"C:/Users/oleg.kazanskyi/Documents/Trading/Models/{period}d_model.joblib"
    #ddrt
    lgbm = joblib.load(open(model_path, 'rb'))
    
    if os.name == 'posix':
        #OS
        label_encoder_path = f"/Users/olegkazanskyi/Documents/GitHub/Trading/Models/label_encoder_012023.joblib"
    else:
        #Windows
        label_encoder_path = "C:/Users/oleg.kazanskyi/Documents/Trading/Models/label_encoder_012023.joblib"   
    
    le = joblib.load(open(label_encoder_path, 'rb'))
    
    df["sector"] = le.transform(df["sector"])
    
    #predicting on test set
    ypred_prob = lgbm.predict_proba(df)[:, 1]
    
    #Adding the predicted probability to the dataframe
    df['predicted_proba'] = ypred_prob
    df['stock'] = stocks_array
    df = df[df['predicted_proba'] >= 0.5]
    
    return df

#Get the dataframe with 1 date data
df_core = get_latest_data()

#Creating a new dataset without the stock column. We can use it to get our classification
df_usage = df_core.drop('stock', axis = 1, inplace = False)
stocks_array = df_core.stock

df_1 = ML_classification(df_usage, stocks_array, '30')
df_1["period"] = 30

# If you want to add a ML results with another time period prediction use the commented part
'''df_2 = ML_classification(df_usage, stocks_array, '15')
df_2["period"] = 15

df_total = pd.concat([df_1,df_2])  '''
    

# Sending email
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import smtplib
import sys

def send_email(df_input):
    df = df_input.copy()
    
    #The mail addresses and password
    sender_address = 'oleg.kazanskyi@gmail.com'
    sender_pass = 'mjmchacmlqapfgii'#input()
    receiver_address = 'oleg.kazanskyi@gmail.com'
    send_copy_to = 'ivanna.kazanska@gmail.com'
    #Setup the MIME
    if df.empty:
        part1 = "Sorry, there are no buys this time"
        part1 = MIMEText(part1)
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Cc'] = send_copy_to
        message['Subject'] = 'S&P 500. No bets today' 
    else:
        df = df[['stock','predicted_proba','period']]
        html = """<html>
          <head></head>
          <body>
            {0}
          </body>
        </html>
        """.format(df.to_html())
        part1 = MIMEText(html, 'html')
        message = MIMEMultipart()
        message['From'] = sender_address
        message['To'] = receiver_address
        message['Cc'] = send_copy_to
        message['Subject'] = 'S&P 500 Classifcation Profit Model' 
    
    #The subject line
    #The body and the attachments for the mail
    message.attach(part1)
    #Create SMTP session for sending the mail
    session = smtplib.SMTP('smtp.gmail.com', 587) #use gmail with port
    session.starttls() #enable security
    session.login(sender_address, sender_pass) #login with mail_id and password
    text = message.as_string()
    session.sendmail(sender_address, [receiver_address,send_copy_to], text)
    session.quit()
    print('Mail Sent')
    
send_email(df_1)

def relearn_model(period = '30'):
    '''
    Relearn the model if it's older than 30 days
    '''
      
    if os.name == 'posix':
        #OS
        os.chdir("/Users/olegkazanskyi/Documents/GitHub/Trading")
        model_path = f"/Users/olegkazanskyi/Documents/GitHub/Trading/Models/{period}d_model.joblib"
    else:
        #Windows
        os.chdir("C:\\Users\\oleg.kazanskyi\\Documents\\Trading")
        model_path = f"C:/Users/oleg.kazanskyi/Documents/Trading/Models/{period}d_model.joblib"
    
    last_mod_time = datetime.datetime.fromtimestamp(os.path.getctime(model_path) ).date()
    delta = date.today() - last_mod_time
    delta = delta.days
    
    if delta >= 30:
        import importlib  
        cleaning = importlib.import_module("ML_part.EOD.Data_cleaning")
        ML_relearn = importlib.import_module("ML_part.EOD.ML_Classifiction")
        
relearn_model(period = '30')
