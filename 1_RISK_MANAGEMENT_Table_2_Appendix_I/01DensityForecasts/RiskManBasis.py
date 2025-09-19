#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Elementary functions: RiskManMain
"""

## Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration

# Optimisation
import scipy.optimize as opt

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
def PredDistr(dictMethods, vH, vY, vRV):
    """
    Purpose
    ----------
    Compute h step ahead forecasts for conditional mean and variance based on specified methods.
 
    Parameters
    ----------
    dictMethods :       object, dictionary containing model specifications and estimation functions
    vH :                vector, forecast horizons
    vY :                vector, observed data
    vRV :               vector, realized variance or auxiliary volatility measure

    Returns
    -------
    mParamForecast      array, parameter forecasts
    """
  
    lMethodsKeys = list(dictMethods.keys())
    iH = len(vH)
    iM = len(lMethodsKeys)
    mParamForecast = np.ones((iH,iM,3)) * np.nan
    
    for i in range(iM):

        # Select method
        dictMethod = dictMethods[lMethodsKeys[i]]
        iP = dictMethod['iP']
    
        mParamForecastMethod =dictMethod['Model'](vY, iP, vH, vRV)
        iHMethod, iNumParamMethod = mParamForecastMethod.shape
        mParamForecast[:,i,0:iNumParamMethod] = mParamForecastMethod

    return mParamForecast

###########################################################  
def PredDistrWindowMPI(dictMethods, vH, vY, vRV, dRq, iTest, vIntWindows):
    """
    Purpose
    ----------
    Compute rolling h step ahead parameter forecasts across multiple windows using MPI.

    Parameters
    ----------
    dictMethods :       object, dictionary containing model specifications and estimation functions
    vH :                vector, forecast horizons
    vY :                vector, observed data
    vRV :               vector, realized variance or auxiliary volatility measure
    dRq :               double, unused input (reserved for compatibility or future use)
    iTest :             integer, length of the test window
    vIntWindows :       vector, start indices of the forecast evaluation windows

    Returns
    -------
    mParamsDF :         array, parameter forecasts with shape (len(vH), number of methods, 3, number of windows)
    """
    
    mParamsDF = np.zeros((len(vH), len(list(dictMethods.keys())), 3, len(vIntWindows)))
    for i in range(len(vIntWindows)):
        idxStart = vIntWindows[i]
        mParamsDF[:,:,:,i] = PredDistr(dictMethods, vH, vY[idxStart:idxStart+iTest], vRV[idxStart:idxStart+iTest])
        
    return mParamsDF