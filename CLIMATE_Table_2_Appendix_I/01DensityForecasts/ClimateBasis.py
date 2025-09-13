#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Helper functions for climate application
"""

## Imports

# Fundamentals
import numpy as np  

# Pandas
import pandas as pd

###############################################################################################
def fWeekNumbers(dfData, bWeekly=False, bDayNumbers=False):
    """
    Purpose
    ----------
    Generate time indices: weekly or day-of-year numbers
    
    Parameters
    ----------
    dfData :         object, DataFrame with datetime index
    bWeekly :        boolean, use modulo week numbers (1–52)
    bDayNumbers :    boolean, return sequential day numbers per year
    
    Returns
    -------
    Vector, contains week numbers or day numbers
    """
    
    if bWeekly:
        return np.array([((i % 52) + 1) for i, _ in enumerate(dfData.values)])
    
    else:
        if not bDayNumbers:
            # Offset all dates by one month so that February 1st is the new "beginning of the year"
            dfOffsetDates = dfData.index - pd.DateOffset(months=1)
            
            # Calculate the "relative" day of the year for each shifted date
            dDayOfYear = dfOffsetDates.dayofyear
            
            # Convert day_of_year to week numbers
            vWeekNumbers = ((dDayOfYear - 1) // 7) + 1
            
            # Replace any week numbers greater than 52 with 52 (this will primarily affect January dates)
            vWeekNumbers = vWeekNumbers.where(vWeekNumbers <= 52, 52)
            
            return vWeekNumbers
        
        else:
            
            #  Initialize an empty array to store day numbers
            vT = np.array([])
            
            # Loop through unique years in the DataFrame
            for year in pd.Series(dfData.index).dt.year.unique():
                
                # Identify whether the year is a leap year or not
                bLeap = (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)
    
                # Generate day numbers depending on leap year status
                if bLeap:
                    vDayNumbers = np.arange(1, 367)
                else:
                    vDayNumbers = np.arange(1, 366)
    
                # Append the day numbers to the main array
                vT = np.concatenate((vT, vDayNumbers))
            
            #vT = np.array([((i % 365) + 1) for i, _ in enumerate(dfData.values)])    
            return vT   
        
###############################################################################################
def fWeeklyData(df, iYearStart=1961, iYearEnd=1990):
    """
    Purpose
    ----------
    Aggregate daily time series to custom weekly format
    
    Parameters
    ----------
    df :             object, DataFrame with datetime index
    iYearStart :     integer, start year
    iYearEnd :       integer, end year
    
    Returns
    -------
    Series, contains weekly averages with custom calendar
    """
    
    lResampledData = []
    lWeekStartDates = []

    for year in range(iYearStart, iYearEnd + 1):

        # Start date is always February 1st of the current year
        sStartDate = pd.Timestamp(f"{year}-02-01")

        # The end date is January 31st of the next year
        sEndDate = pd.Timestamp(f"{year+1}-01-31")

        # Extract subset for the "year" duration
        subset = df.loc[sStartDate:sEndDate]

        # Compute the number of days for 51 weeks
        iNumDays51wks = 51 * 7

        # Resample data for the first 51 weeks
        vWeeksFirst51 = [subset.iloc[i:i + 7] for i in range(0, iNumDays51wks, 7)]

        # The remaining days of the year are considered as the 52nd week
        vWeek52 = subset.iloc[iNumDays51wks:]

        # Combine the lists
        vWeeks = vWeeksFirst51 + [vWeek52]

        for week in vWeeks:
            lResampledData.append(week.mean())
            lWeekStartDates.append(week.index[0])

    return pd.Series(lResampledData, index=lWeekStartDates)


###########################################################
def MPITaskDistributor(iRank, iProc, iTotal, bOrder=True):
    """
    Purpose
    ----------
    Construct vector of indices [integers] for which process with rank iRank should do calculations

    Inputs
    ----------
    iRank :     integer, rank running process
    iProc :     integer, total number of processes in MPI program
    iTotal :    integer, total number of tasks
    bOrder :    boolean, use standard ordering of integers if True

    Output
    ----------
    vInt :      vector, part of total indices 0 to [not incl.] iTotal that must be calculated by process
                with rank iRank
    """
    
    # Print error if less than one task per process
    if iTotal/iProc < 1: print('Error: Number of tasks smaller than number of processes')
    
    # If iTotal/iProc is not an integer, then all processes take on one additional task, except for the last process.
    iCeil = int(np.ceil(iTotal/iProc))
    lInt = []
    
    # Standard ordering from 0 to iTotal-1
    if bOrder:
        for i in range(iTotal):
            lInt.append(iRank * iCeil + i)
            if len(lInt) == iCeil or iRank * iCeil + i == iTotal-1:
                break
        
    # Using iProc steps
    else:
        for i in range(iTotal):
            lInt.append(iRank  + iProc * i)
            if iRank  + iProc * (i+1) > iTotal -1:
                break    
    
    return  np.array(lInt)

###########################################################  
def Rhat(vY, lRq, iTest, iNumEstWindows):
    """
    Purpose
    ----------
    Calculate rhats, quantiles of estimation windows
    
    Parameters
    ----------
    vY :                vector, data
    lRq :               list, quantiles for r
    iTest :             integer, length estimation window
    iNumEstWindows :    integer, number of estimation windows
    
    Returns
    -------
    mRhat :             matrix, empirical quantiles              
    """
    
    mRhat = np.ones((len(lRq),iNumEstWindows)) * np.nan
    
    for i in range(iNumEstWindows):
        vYWindow = vY[i:i+iTest]
        mRhat[:,i] = np.quantile(vYWindow, lRq) 
  
    return mRhat

###########################################################  
def GammaHat(vY, lR, dC, iTest, iNumEstWindows):
    """
    Purpose
    ----------
    Calculate gammahats
    
    Parameters
    ----------
    vY :                vector, data
    lR :                list, quantiles for r
    dC                  double, center region of intererst
    iTest :             integer, length estimation window
    iNumEstWindows :    integer, number of estimation windows
    
    Returns
    -------
    mGammaHat :         matrix, gammahat per threshold r
    """
   
    mGammaHat = np.ones((len(lR),iNumEstWindows)) * np.nan
     
    for i in range(iNumEstWindows):
        
        vYWindow = vY[i:i+iTest]
    
        # Create an index array with the same shape as vY that contains the corresponding day of the year for each element
        vDayIndices = np.arange(iTest) % 366 # vY contains at least one year of data 
        for r in range(len(lR)):
            dR = lR[r]
            dA1 = dC - dR
            dA2 = dC + dR
            
            # Calculate daily gammas [use 365 average for day 366] [inefficient, but takes no computation time]
            vDailyGammasA1 = np.array([np.sum(vYWindow[vDayIndices == np.min((iDay, 365))] <  dA1) for iDay in range(366)])
            vDailyGammasA1andA2 = np.array([np.max((1,np.sum(vYWindow[vDayIndices == np.min((iDay, 365))] <  dA1) + np.sum(vYWindow[vDayIndices == np.min((iDay, 365))] > dA2))) for iDay in range(366)])
            
            mGammaHat[r,i]  = vDailyGammasA1[vDayIndices[-1]]/np.max((vDailyGammasA1andA2[vDayIndices[-1]],1))

    return mGammaHat
    
###########################################################  
def PredDistr(dictMethods, vH, vY, vT, vTh, bWarm):
    """
    Purpose
    ----------
    Parameters predictive distribution
    
    Parameters
    ----------
    dictMethods :   object, dictionary of forecasting methods
    vH :            vector, forecast horizons
    vY :            vector, data
    vT :            vector, in-sample time values
    vTh :           vector, out-of-sample time values
    bWarm :         boolean, warm start option
    
    Returns
    -------
    mParamForecast : array, predictive distribution parameters
    """

    lMethodsKeys = list(dictMethods.keys())
    iH = len(vH)
    iM = len(lMethodsKeys)
    mParamForecast = np.ones((iH,iM,3)) * np.nan
    
    for i in range(iM):

        # Select method
        dictMethod = dictMethods[lMethodsKeys[i]]
        mParamForecastMethod =dictMethod['Model'](vY, vT, vTh, dictMethod['sDistr'], bWarm)
        iHMethod, iNumParamMethod = mParamForecastMethod.shape
        mParamForecast[:,i,0:iNumParamMethod] = mParamForecastMethod
        
    return mParamForecast

###########################################################  
def PredDistrWindowMPI(dictMethods, vH, vY, vT, iTest, vIntWindows, bWarm):
    """
    Purpose
    ----------
    Parameters predictive distribution MPI
    
    Parameters
    ----------
    dictMethods :   object, dictionary of forecasting methods
    vH :            vector, forecast horizons
    vY :            vector, data
    vT :            vector, time values
    iTest :         integer, length estimation window
    vIntWindows :   vector, selection of indices for calculation
    bWarm :         boolean, warm start option
    
    Returns
    -------
    mParamsDF : array, contains predictive distribution parameters
    """
    
    mParamsDF = np.zeros((len(vH), len(list(dictMethods.keys())), 3, len(vIntWindows)))
    for i in range(len(vIntWindows)):
        idxStart = vIntWindows[i]
        mParamsDF[:,:,:,i] = PredDistr(dictMethods, vH, vY[idxStart:idxStart+iTest], vT[idxStart:idxStart+iTest],vT[idxStart+iTest:idxStart+iTest+vH.max()], bWarm)
        
    return mParamsDF
