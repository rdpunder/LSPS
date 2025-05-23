#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Elementary functions: RiskManMain

Version:
    7 Add Tailindicator
    8 Add tpnorm
    
Date:
    2021/09/04

Author:
    Ramon de Punder 
"""

## Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration
import pandas as pd
from twopiece.scale import tpnorm

from ScoringRules import * # scoring rules

###########################################################
def MPITaskDistributor(iRank, iProc, iTotal, bOrder= True):
    """
    Purpose:
        Construct vector of indices [integers] for which process with rank iRank should do calculations

    Inputs:
        iRank :     integer, rank running process
        iProc :     integer, total number of processes in MPI program
        iTotal :    integer, total number of tasks
        bOrder :    boolean, use standard ordering of integers if True

    Output:
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
def GammaHat(mYFull, lR, dC, iHorizonInflation, iT):
    """
    Purpose
    ----------
    Calculate gammahats
    
    Parameters
    ----------
    mYFull :            array, full sample of observations
    lR :                array, list of radius parameters
    dC :                float, center of interval
    iHorizonInflation : integer, forecast horizon
    iT :                integer, number of evaluation periods
    
    Returns
    -------
    mGammaHat :         matrix, gammahat per threshold r
    """

    mGammaHat = np.ones((len(lR),iT)) * np.nan
    iTFull = mYFull.shape[0]    

    iWindow =iTFull-iT
    for i in range(iT):
            vY = mYFull[i:i+iWindow]
            vYWindow =  vY[~np.isnan(vY)]
            
            for r in range(len(lR)):
                dR = lR[r]
                dA1 = dC - dR
                dA2 = dC + dR
                
                if np.sum(vYWindow <  dA1) == 0:
                    dGamma = 0
                else:
                    dGamma = np.sum(vYWindow <  dA1)/(np.sum(vYWindow <  dA1) + np.sum(vYWindow > dA2))
                
                mGammaHat[r,i]  = dGamma

    return mGammaHat

###########################################################  
def RollingScoresMPI(dictMethods, dictScores, dictW, lY, lParamsDF, mRhat, vIntScoresByProc, mGammaHat=None):
    """
    Purpose
    ----------
    Calculate scores based on time-varying density forecasts 
    
    
    Parameters
    ----------
    dictMethods :        dictionary, method definitions
    dictScores :         dictionary, scores definitions
    dictW :              dictionary, weight function definition
    lYh :                list, shifted data per horizon: y[t] = y_{t+h} [useful for f[t]=f_{t+h|t} evaluations]
    lParamsDM :          list, h-step ahead forecasts time-varying parameters 
                             mParamsDFh :  iM x {mu_{t+h|t}, sig2_{t+h|t}, theta_{t+h|t}} x iT 
    mRhat :              array,  iQnum x iT, empirical quantiles Rhat[t]=r_{t+h|t}                        
    vIntScoresByProc :   array, time indices for which scores need to be calculated by running process
    mGammaHat :          matrix, gammahat per threshold r
    
    Returns
    -------
    lScores:              array, size iNumH x iQnum x iM x iS x len(vIntScoresByProc) with scores
    """

    # Initialization
    lMethodsKeys = list(dictMethods.keys())
    lScoreNames = list(dictScores.keys())
    iM = len(lMethodsKeys)
    iS = len(lScoreNames)
    iNumH = len(lY)
    iQnum = mRhat.shape[0]
    mScores = np.ones((iNumH, iQnum, iM, iS, len(vIntScoresByProc))) * np.nan

    for h in range(iNumH):
        vYh = lY[h]
        for q in range(iQnum):
            vR=mRhat[q,:] 
            dictW['vR'] = vR
                       
            for m in range(iM):
       
                # Select method
                dictMethod = dictMethods[lMethodsKeys[m]]
                # Note: A for-loop over t is required in this case, since the density forecast are time-varying
                for i in range(len(vIntScoresByProc)):
                    t = vIntScoresByProc[i]
                    dMu, dParam2, dParam3 = lParamsDF[h][m,0:3,t] 
                    
                    #dNu = np.min((dNu,1e4))
                    # Define time-varying density forecasts
                    if dictMethod['sDistr'] == 'Normal':
                        dictDistr = {'randDistr' : stats.norm(dMu,np.sqrt(dParam2)), 'sDistr': 'Normal'}
                    elif dictMethod['sDistr'] == 'Student-t': 
                         dictDistr =  {'randDistr' : stats.t(dParam3,loc=dMu, scale=np.sqrt(dParam2)/np.sqrt(dParam3/(dParam3-2))), 'sDistr' :'Student-t'}
                    elif dictMethod['sDistr'] == 'tpnorm': 
                         dictDistr =  {'randDistr' : tpnorm(loc=dMu, sigma1=dParam2, sigma2 =dParam3), 'sDistr' :'tpnorm'}     
                    elif dictMethod['sDistr'] == 'SStudent-t': 
                        dictRand = {'pdf' : lambda x: PdfSStd(x, lParamsDF[h][m,:,t]),'cdf' : lambda x: CdfSStd(x, lParamsDF[h][m,:,t])}
                        dictDistr = {'randDistr' : dictRand, 'sDistr' :'SStudent-t'}
                    
                    # Precalculated quantities of time-varying density forecast
                    dictPreCalc = {'AlphaNormsF': {dictDistr['sDistr']: AlphaNormsF(dictDistr, dictW, np.array([2]))},
                                   'AlphaNormsFw': {dictDistr['sDistr']: AlphaNormsFw(dictDistr, dictW, np.array([2]), np.array([t]))},
                                   'DistrBar': {dictDistr['sDistr']: DistrBar(dictDistr, dictW, np.array([t]))},
                                   'NumIntSettings': {'dMinInf':-20, 'dPlusInf': 20, 'dDeltaZ' : 0.001}
                                   }
                    
                    if  dictW['fW'].__name__ == 'fWIndicatorC':
                        dictPreCalc['Gamma'] = mGammaHat[q,t]
                    for s in range(iS):
                        mScores[h,q,m,s,i]=dictScores[lScoreNames[s]]['fScore'](dictDistr, np.array(vYh[t]).reshape((1)), t, dictW, dictScores[lScoreNames[s]]['vParamsS'], dictPreCalc)
             
    return mScores

###########################################################  
def InflationParamsDFandmRhat(lMeanNames, sDistr, dRTarget, lRq, iHorizon):
    """
    Purpose
    ----------
    Wrapper for construction mParamsDFh1 for inflation example
    Time varying parameters have been estimated in R.

    Purpose
    ----------
    Compute parameter array and R-hat grid for inflation forecast example
    
    Parameters
    ----------
    lMeanNames :       list, contains mean part filenames
    sDistr :           string, name of distribution
    dRTarget :         float, target radius (not used in current implementation)
    lRq :              list, grid of Rhat values
    iHorizon :         integer, forecast horizon
    
    Returns
    -------
    mParamsDFh1 :      array, contains estimated distribution parameters
    mRhat :            array, grid of Rhat values
    iNumEstWindows :   integer, number of estimation windows
    """

    mParams = np.load('mParamsDF/'+lMeanNames[0]+sDistr+'h'+str(iHorizon)+'.npy', allow_pickle=True) 
    iNumEstWindows, iParams = mParams.shape 
    iM = len(lMeanNames)
    mParamsDFh1 = np.ones((iM, int(3+int(sDistr=='sstd')), iNumEstWindows)) * np.nan
    
    for m in range(iM):
        sName = lMeanNames[m]
        mParams = np.load('mParamsDF/'+sName+sDistr+'h'+str(iHorizon)+'.npy', allow_pickle=True) 
        mParamsDFh1[m,] = mParams[:,:int(3+int(sDistr=='sstd'))].T
           
    mRhat = np.ones((len(lRq), iNumEstWindows))  * np.nan   
    for r in range(len(lRq)):    
        #mRhat[r,] = dRTarget - lRq[r]  
        mRhat[r,] = lRq[r]  
        
    return mParamsDFh1, mRhat, iNumEstWindows