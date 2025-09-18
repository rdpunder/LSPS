#!/usr/bin/env python3# -*- coding: utf-8 -*-## Imports# Fundamentalsimport numpy as np  # Systemimport pandas as pd# Download data from Yahoo Financeimport yfinance as yf            ## Magic numbers
sStart = "1980-01-01"
sEnd = '2022-12-31' 
timeStart = pd.to_datetime(sStart, format='%Y-%m-%d') 
timeEnd = pd.to_datetime(sEnd, format='%Y-%m-%d') 
sFileName = 'SP500andRealVol1995'
    
# Use: SPDR S&P 500 ETF Trust (SPY)
dfClosingPrices = yf.download("SPY", start=timeStart, end=timeEnd, auto_adjust=False)['Adj Close'] # SPY is tradeable ETF
dfReturns = np.log(dfClosingPrices[1:]) - np.log(dfClosingPrices.shift(1)[1:])
dfReturns.rename(columns={'SPY':'Adj Close'}, inplace=True)
    
## Add RV data of RiskLab Xiu
dfRMAllFullPeriod = pd.read_csv('RealisedVolatilityFullPeriodTrade.csv', sep=',')
dfRMAllFullPeriod.index = dfRMAllFullPeriod['Date']
dfRMAllFullPeriod.index = pd.to_datetime( dfRMAllFullPeriod.index, format='%Y-%m-%d')
dfRMFullPeriod = dfRMAllFullPeriod['Volatility'] 
dfRMa  = dfRMFullPeriod.loc[sStart:sEnd]
dfRM= pd.DataFrame(data=(dfRMa.values**2)/252, index=dfRMa.index, columns=['rv5'])

## Merge data
dfData = dfReturns.merge(dfRM, left_index=True, right_index=True, how='inner') 
dfData.to_csv(sFileName + 'Xiu'+ '.csv')
