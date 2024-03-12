Hello Friend,

This is my personal project that attemps selecting stock 
based on financial data for mid term (30-60 days) trading.

In this file I'm putting a short description of how everything works,
while to see the visual structure please check the image "Full_Pipeline.png".

I worked on this project on my personal laptop and I hardcoded the folder path.
These scripts are not polished and you may need to change the address in all *.py files.  
I know this is far from the best practice, sorry.


1. File Dates_FedRes_Bonds.py collects:
- 10Y of data about US Federal Reserve Balance 
- 10Y of data of U.S. 10-years bonds yield
- 10Y of data of the daily values for VIX (fear indicator)
- History of last stock crises with dates

2. The file "eodhistoricaldata.py" combine data from the previous file with data coming from API eodhd.com .
What data do we get from eodhd.com:
- Daily trade values for 500 SP stocks
- Information about stock split from SP500 stocks
- Financial data for the last 10y for 500 SP stocks
- Expert expectations about futuree earnings

After loading the data the same script convert it into financial KPIs: 
    ROE - Return n Equity
    LTDE - Long term to Debt Equity
    DE - Debt to Equity
    CR - Current Liquidity Ratio
    GM - Gross Margin
    RQoQ - Revenue Quarter over Quarter
    ROA - Return on Asset
    ES - Earnings per stock
    PER - Price to Earning Ratio
    PERG - Price / Earnings Growth Ratio
    DPR = Divident Payot Ratio
    DY - dividends Yield
    ARTR - Accounts Receivable Turnover Ratio
    Industry of activities

3. The Script ML Part/EOD/Data_cleaning.py contains the data cleaning process

4. The Script ML Part/EOD/ML_Classifiction.py contains model selection, model training and evaluation part.
It also save the model to the joblib file.


5. The final file is called "Apply_ML_Send_Email.py".
It checks if the model is older than 30 days aand relearn it if it's too old.
Otherwise it applies the model to the latest data and sends an email if any stocke below market value are found.

If you have any questions or would like to cooperate with this project message me
oleg.kazanskyi@gmail.com

