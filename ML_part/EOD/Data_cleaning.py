#!/usr/bin/env python
# coding: utf-8

# Import the required libraries

# In[1]:


from jupyter_core.paths import jupyter_path
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import os
import glob
import matplotlib.pyplot as plt
from datetime import date
from datetime import datetime
from datetime import timedelta


# In[2]:


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from sklearn import metrics
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import precision_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
import lightgbm as lgb # Our ML library


# In[3]:


#Cross validation libraries
from sklearn.model_selection import TimeSeriesSplit


# In[4]:


import joblib


# Read all CSVs with stocks data and append to one big file

# In[5]:


#os.chdir("/Users/olegkazanskyi/Documents/GitHub/Trading/CSVs")
os.chdir("C:/Users/oleg.kazanskyi/OneDrive - Danaher/Documents/Trading/SP500_CSVs_01032023")
filepaths = [f for f in os.listdir("./") if f.endswith('.csv')]
df = pd.DataFrame()
for i in filepaths:
    iterate_df = pd.DataFrame()
    iterate_df = pd.read_csv(i, encoding= 'unicode_escape')
    iterate_df["stock"] = i[:-4]
    df = pd.concat([df,iterate_df])
#df = pd.concat(map(pd.read_csv, filepaths))

df = df[df.close.notna()]
os.chdir("C:/Users/oleg.kazanskyi/OneDrive - Danaher/Documents/Trading/ML_part/EOD")
df.to_csv("THE_FINAL_DATASET_2023.csv")


# record the dataframe to speed up the future reading process

# os.chdir("C:/Users/oleg.kazanskyi/Personal-oleg.kazanskyi/Trading Python/")
# df.to_csv("THE_FINAL_DATASET_2023.csv")
os.chdir("C:/Users/oleg.kazanskyi/OneDrive - Danaher/Documents/Trading/ML_part/EOD")
df= pd.read_csv("THE_FINAL_DATASET_2023.csv")
#df.drop(["Unnamed: 0","dataCode","name","description","statementType","units","Unnamed: 0.2", "Unnamed: 0.1"], axis = 1, inplace = True)
df = df[df.close.notna()]
# In[6]:


#df['YoY_DY'] = 100*df['DY'].pct_change(periods = -365)


# In[7]:


df.shape


# In[8]:


df.duplicated().sum()


# Set up Date column as an index columns

# In[9]:


df["date"] = pd.to_datetime(df["date"])


# Let's check how much empty values we have by column

# In[10]:


zeroes = df.isnull().sum()
print(zeroes[zeroes>0])
del zeroes


# ### Dealing with "EPS" column

# There are many null values in the "current_ratio" column. Let's see the stocks where it happens and then decide what to do

# In[11]:


df[df.EPS_surprise.isnull()].stock.unique()


# After checking the original files, it's apparently the null values comes from the older data.
# 
# Before 2017 our API did not provide us info about earinings. 
# 
# We can drop the reocrds with 'nan' values

# we will remove only 10k records 

# In[12]:


df = df[df.EPS_surprise.notnull()]


# Let's check the dataset again

# In[13]:


def check_null_cols(df):
    zeroes = df.isnull().sum()
    print(zeroes[zeroes>0])
    
check_null_cols(df)   


# ### Dealing with the null values in the "dividends" column

# About 30% of data is affected by null values in the dividends columns.
# Let's check how many companies are in the list

# In[14]:


print("Number of companies with zero values in dividends",len(df[df['DY'].isnull()].stock.unique()))


# This companies do not pay dividends, we can replace payments to "0"

# In[15]:


df['DY'].fillna(0, inplace=True)
df['YoY_DPR'].fillna(0, inplace=True)
df['DPR'].fillna(0, inplace=True)
df['YoY_DY'].fillna(0, inplace=True)                   


# ### Dealing with YoY

# The YoY variables with null values are caused by YoY calculation of the rows without historical data.
# Let's drop these

# In[16]:


df = df[df.YoY_CR.notnull()]


# Let's check the dataset again

# In[17]:


check_null_cols(df)   


# In[18]:


df.shape


# ### Dealing with debt ratio

# Let's check the companies that contain null values

# In[19]:


df[df['DE'].isnull()].stock.unique()


# It seems these companies had zero debt for some quarters.
# 
# Let's replace with "0"

# In[20]:


df['DE'].fillna(0, inplace=True)
df['LTDE'].fillna(0, inplace=True)


# In[21]:


check_null_cols(df)   


# ### Dealing with VIX

# Let's remove null values for VIX as these are related to the historical calculations

# In[22]:


df = df[df.VIX_MoM.notnull()]


# In[23]:


check_null_cols(df) 


# ### Dealing with Accounts Payable

# In[24]:


df[df['Acc_Rec_Pay_Ration'].isnull()].stock.unique()


# It seems these companies had zero Accounts receivables for some quarters

# In[25]:


df['Acc_Rec_Pay_Ration'].fillna(0, inplace=True)


# ### Dealing with PEG

# In[26]:


df[df['PEG_Forward'].isnull()].stock.unique()


# We can drop the rows as the effect would be unsignificant

# In[27]:


df = df[df.PEG_Forward.notnull()]


# In[28]:


df[df['PEG_Backwards'].isnull()].stock.unique()


# In[29]:


df = df[df.PEG_Backwards.notnull()]


# In[30]:


df = df[df.EPS_1Y_exp_Change.notnull()]


# In[31]:


check_null_cols(df) 


# ### Dealing with Industry

# In[32]:


df[df['sector'].isnull()].stock.unique()


# Let's fill the sector and industry values for the GEN company

# In[33]:


df.loc[df['stock'] == 'GEN','sector'] = 'Information Technology'
df.loc[df['stock'] == 'GEN','industry'] = 'Software & Services'


# In[34]:


check_null_cols(df) 


# In[35]:


df = df[df.YoY_AR_Ration.notnull()]


# ### Working with data types

# Now we can keep only the necessery columns

# In[36]:


categoric_columns = df.select_dtypes(include='object').columns
for col in categoric_columns:
    if col == "stock":
        continue
    print(f"column {col}, data: \n {df[col].unique()}")


# In[37]:


#Number of stocks per industry
df[['sector','industry','stock']].groupby('industry').stock.nunique()


# In[38]:


#Number of stocks per sector
df[['sector','industry','stock']].groupby('sector').stock.nunique()


# We can drop "Stocks" and "Industry" columns as there are too many unique values that block us from generalizing the data. 

# In[39]:


#df.drop(["industry", "stock"], axis = 1, inplace = True)
#df.drop(["industry"], axis = 1, inplace = True)


# Let's also check if there are any infinite numbers that can cause same trublesas nan

# In[40]:


df.replace([np.inf, -np.inf], np.nan, inplace = True)

check_null_cols(df) 


# In[41]:


df = df[df.EPS_1Y_exp_Change.notnull() & df.EPS_YoY_Growth.notnull() & df.EPS_QoQ_frcst_diff.notnull()]


# In[42]:


check_null_cols(df)


# In[43]:


df = df[df['Fed_Balance_YoY'].notnull()]


# ### Data Cleaning is Over!

# ## Trimming the data to avoid overfitting

#  Our dataset have the range of dates that are very close to each other. 
# 
# The changes in trading value are rarely very different between today and yesterday. 
# 
# There are exemptions but mostly these are connected with big surprises, and in any case we would notice the change even if we track every other day or every 4th day.
# 
# It means that removing part of the dataset should help us in building more generalized model since we will be looking for trend but not for matching values.
# 
# 
# Let's do this, let's remove 4/5 of the dataset.
# 
# That means we will keep only one day of data per week. It will help us to generalize the dataset

# print("Old dataframe shape:", df.shape)
# df_compact = df.iloc[::5, :]
# print("New dataframe shape:", df_compact.shape)

# The final dataset is 131K rows long with 53 variables and 6 targets(price after 15 days, 30 days, 60 days, 90 days, 120 days and 150 days)
# 
# We can remove the bigger dataset, but let's save it to the file before doing so.

# In[44]:


os.chdir("C:/Users/oleg.kazanskyi/OneDrive - Danaher/Documents/Trading/ML_Part/EOD")
df.to_csv("full_cleaned_dataframe_2023.csv", index = False, header = True)
del df


# Lets' save the shorter version of the dataframe as well so we can get it faster when required

# os.chdir("C:/Users/oleg.kazanskyi/Personal-oleg.kazanskyi/Trading Python/ML_Part/EOD")
# df_compact.to_csv("shorter_cleaned_dataframe_2023.csv", index = False, header = True)
