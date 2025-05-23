#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Empiricial data
    
Version:
    1   Copy relevant functions from RollingWindowv12 
    2  
Date:
    2021/08/04

Author:
    Ramon de Punder 
"""

## Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration

# System
import os
from os import path
import pandas as pd
from datetime import datetime
from datetime import date

# Visualisation

# Plotting
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
import matplotlib.dates as mdates
import matplotlib.ticker as mtick

# Download data from Yahoo Finance
import yfinance as yf



os.environ["PATH"] += os.pathsep + '/Users/ramondepunder/bin' # add MikTex to PATH
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':'15'})
rc('text', usetex=True)
lFigureSettings= {'figsize':(10,6), 'dpi':70, 'titlefontsize':16, 'axisfontsize':14} 


###########################################################  
def FigureData(dfData, sNameFig): 
    """
    Purpose:
        Generate subplots Figure 1 

    Inputs:
        dfData      dataframe, contains data with atetimeobject as index        

    Output:
        Subplots Figure 1
    
    """
    

    f = plt.figure(figsize=(13,4))
    #plt.axvspan(*mdates.datestr2num(['8/1/2001', '11/30/2001']), color=sns.xkcd_rgb['grey'], alpha=0.5, lw=0)
    plt.axvspan(*mdates.datestr2num(['12/1/2007', '6/30/2009']), color=sns.xkcd_rgb['grey'], alpha=0.5, lw=0)
    plt.axvspan(*mdates.datestr2num(['2/1/2020', '4/30/2020']), color=sns.xkcd_rgb['grey'], alpha=0.5, lw=0)
    plt.plot(pd.DataFrame(0.0, index=dfData.index, columns=[0]), color='k', linewidth=1)
    plt.plot(dfData, color='C0')
    plt.margins(x=0)
    plt.show()
    f.savefig(sNameFig+'.pdf', bbox_inches='tight')
        
###########################################################  
def main():    
    
    ## Cleaning
    plt.close('all')                        # close all figures
    
    ###########################################################  
    ## Magic numbers
    sStart = "1980-01-01"
    sEnd = '2022-12-31' #"2008-03-14"
    sFileName = 'SP500andRealVol1980full'
    
    
    ### Xiu Data
    # Have to download SP500 ourselves
    #dfClosingPrices = yf.download("^GSPC", start=sStart, end=sEnd)['Adj Close']
    # Use: SPDR S&P 500 ETF Trust (SPY)
    dfClosingPrices = yf.download("SPY", start=sStart, end=sEnd)['Adj Close'] # SPY is tradeable ETF
    dfReturns = np.log(dfClosingPrices[1:]) - np.log(dfClosingPrices.shift(1)[1:])
    FigureData(dfReturns, "SP500retadjclosing")
    plt.plot(dfReturns)
    
    ## Add RV data
    dfRMAllFullPeriod = pd.read_csv('RealisedVolatilityFullPeriodTrade.csv', sep=',')
    dfRMAllFullPeriod.index = dfRMAllFullPeriod['Date']
    dfRMAllFullPeriod.index = pd.to_datetime( dfRMAllFullPeriod.index, format='%Y-%m-%d')
    dfRMFullPeriod = dfRMAllFullPeriod['Volatility'] 
    dfRMa  = dfRMFullPeriod.loc[sStart:sEnd]
    dfRM= pd.DataFrame(data=(dfRMa.values**2)/252, index=dfRMa.index, columns=['rv5'])
    
    ### Oxford data
    dfRMAllFullPeriodOxfordAll = pd.read_csv('oxfordmanrealizedvolatilityindicesPB.csv', sep=',')
    dfRMAllFullPeriodOxfordAll = dfRMAllFullPeriodOxfordAll.rename(columns={'Unnamed: 0': 'Date'})
    dfRMAllFullPeriodOxfordAll['Date'] = pd.to_datetime(dfRMAllFullPeriodOxfordAll['Date'], format='%Y-%m-%d', utc=True).dt.date
    dfRMAllFullPeriodOxfordAll.index = dfRMAllFullPeriodOxfordAll['Date']

    dfRMAllFullPeriodOxford = dfRMAllFullPeriodOxfordAll[dfRMAllFullPeriodOxfordAll['Symbol']=='.SPX']
    dfClosingPricesOxford = dfRMAllFullPeriodOxford['close_price']
    dfReturnsOxford = np.log(dfClosingPricesOxford[1:]) - np.log(dfClosingPricesOxford.shift(1)[1:])
    FigureData(dfReturnsOxford, "SP500retclosingoxford")
    plt.plot(dfReturnsOxford)
    dfRMOxford = dfRMAllFullPeriodOxford['rv5']

    ## Merge data
    dfData = dfReturns.to_frame().merge(dfRM, left_index=True, right_index=True, how='inner') 
    dfData.to_csv(sFileName + 'Xiu'+ '.csv')
    dfDataOxford = dfReturns.to_frame().merge(dfRMOxford, left_index=True, right_index=True, how='inner') 
    dfDataOxford.to_csv(sFileName + 'Oxford' + '.csv')
    
    ### Data MATLAB
    dfDataMatlab = pd.read_csv('MatlabData.csv', sep=',', header=None)
    dfDataMatlab.columns = ['Date', 'Returns', 'RV5', 'vix'] 
    dfDataMatlab.to_csv(sFileName + 'Matlab' + '.csv')
    
   