#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Elementary functions: RiskManMainMV
"""

## Imports

# Fundamentals
import numpy as np  

###########################################################
def MPITaskDistributor(iRank, iProc, iTotal, bOrder= True):
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
def RhatMV(mY, lRq, iTest, iNumEstWindows):
    """
    Purpose
    ----------
    Calculate rhats, quantiles of estimation windows
    
    Parameters
    ----------
    mY :                matrix, data with two columns
    lRq :               vector, quantile levels
    iTest :             integer, window size
    iNumEstWindows :    integer, number of estimation windows
    
    Returns
    -------
    mRhat :             matrix, empirical quantiles
    """
    
    mRhat = np.ones((2,len(lRq),iNumEstWindows)) * np.nan
    for j in range(2):
        for i in range(iNumEstWindows):
            vYWindow = mY[i:i+iTest,j]
            mRhat[j,:,i] = np.quantile(vYWindow, np.sqrt(lRq)) 
            
    return mRhat

###########################################################  
def PredDistr(dictMethods, vH, mY, mRV, iPredParams=3):
    """
    Purpose
    ----------
    Estimate method and return h-stap ahead forecast 
    
    Parameters
    ----------
    dictMethods :       dictionary, model specifications with estimation and prediction functions
    vH :                vector, forecast horizons
    mY :                matrix or vector, data
    mRV :               matrix, realized volatility or auxiliary variables
    iPredParams :       integer, number of prediction parameters per method
    
    Parameters
    ----------
    mParamForecast :    array, parameter forecasts
    """
        
    lMethodsKeys = list(dictMethods.keys())
    iH = len(vH)
    iM = len(lMethodsKeys)
    mParamForecast = np.ones((iH,iM,iPredParams)) * np.nan
    
    for i in range(iM):

        # Select method
        dictMethod = dictMethods[lMethodsKeys[i]]
        iP = dictMethod['iP']
        mParamForecastMethod =dictMethod['Model'](mY, iP, vH, mRV)
        iHMethod, iNumParamMethod = mParamForecastMethod.shape
        mParamForecast[:,i,0:iNumParamMethod] = mParamForecastMethod

    return mParamForecast

###########################################################  
def PredDistrWindowMPI(dictMethods, vH, mY, mRV, dRq, iTest, vIntWindows, bMultivariate=False):
    """
    Purpose
    ----------
    Pred distribution window wrapper MPI
    
    Parameters
    ----------
    dictMethods :       dictionary, model specifications with estimation and prediction functions
    vH :                vector, forecast horizons
    mY :                matrix, data
    mRV :               matrix, realized volatility or auxiliary variables
    dRq :               double, quantile level (not used in function but required for compatibility)
    iTest :             integer, length estimation window
    vIntWindows :       vector, start indices of estimation windows
    bMultivariate :     boolean, use multivariate input structure if True
   
    Returns
    -------
    mParamsDF :         array, contains parameter forecasts of size [len(vH) x #methods x #params x #windows]
   
    """
    
    mParamsDF = np.zeros((len(vH), len(list(dictMethods.keys())), 3 + int(3*bMultivariate), len(vIntWindows)))
    for i in range(len(vIntWindows)):
        idxStart = vIntWindows[i]
        #print(i)
        if bMultivariate:
            mParamsDF[:,:,:,i] = PredDistr(dictMethods, vH, mY[idxStart:idxStart+iTest,:], mRV[idxStart:idxStart+iTest,:], 6)
        else:
            mParamsDF[:,:,:,i] = PredDistr(dictMethods, vH, mY[idxStart:idxStart+iTest], mRV[idxStart:idxStart+iTest], 3)
        
    return mParamsDF