#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Elementary functions: RiskManScoreCalcMain
"""

## Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables

from ScoringRules import * # scoring rules

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
    lRq :               double, quantiles for r
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
def RollingScoresMPI(dictMethods, dictScores, dictW, lY, lParamsDF, mRhat, vIntScoresByProc):
    """
    Purpose
    ----------
    Calculate scores based on time-varying denisty forecasts 
    
    
    Parameters
    ----------
    dictMethods :        dictionary, method definitions
    dictScores :         dictionary, scores definitions
    dictW :              dictionary, weight function definition
    lY :                 list, shifted data per horizon: y[t] = y_{t+h} [useful for f[t]=f_{t+h|t} evaluations]
    lParamsDM :          list, h-step ahead forecasts time-varying parameters 
                             mParamsDFh :  iM x {mu_{t+h|t}, sig2_{t+h|t}, theta_{t+h|t}} x iT 
    mRhat :              array,  iQnum x iT, empirical quantiles Rhat[t]=r_{t+h|t}                        
    vIntScoresByProc :   array, time indices for which scores need to be calculated by running process
    
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
                    dMu, dSig2, dNu = lParamsDF[h][m,:,t] 
                    #dNu = np.min((dNu,1e4))
                    # Define time-varying density forecasts
                    if dictMethod['sDistr'] == 'Normal':
                        dictDistr = {'randDistr' : stats.norm(dMu,np.sqrt(dSig2)), 'sDistr': 'Normal'}
                    elif dictMethod['sDistr'] == 'Student-t': 
                         dictDistr =  {'randDistr' : stats.t(dNu,loc=dMu, scale=np.sqrt(dSig2)/np.sqrt(dNu/(dNu-2))), 'sDistr' :'Student-t'}
                    # Precalculated quantities of time-varying density forecast
                    dictPreCalc = {'AlphaNormsF': {dictDistr['sDistr']: AlphaNormsF(dictDistr, dictW, np.array([2]))},
                                   'AlphaNormsFw': {dictDistr['sDistr']: AlphaNormsFw(dictDistr, dictW, np.array([2]), np.array([t]))},
                                   'DistrBar': {dictDistr['sDistr']: DistrBar(dictDistr, dictW, np.array([t]))},
                                   'NumIntSettings': {'dMinInf':-20, 'dPlusInf': 20, 'dDeltaZ' : 0.001}
                                   }
                    for s in range(iS):
                        mScores[h,q,m,s,i]=dictScores[lScoreNames[s]]['fScore'](dictDistr, np.array(vYh[t]).reshape((1,1)), t, dictW, dictScores[lScoreNames[s]]['vParamsS'], dictPreCalc)
   
    return mScores
 