#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Estimation and prediction LocalMeanSinGARCHII
"""

###########################################################
### Imports
import numpy as np

from scipy import stats # pre-programmed random variables
from scipy.optimize import minimize
from numpy import roots

from scipy.special import logit, expit, loggamma

from ClimateLocMeanSinGARCH import * 

###########################################################
def TempQGARCHIILink(vParams, sDistr='normal'):
    """
    Purpose
    ----------
    Function of parameters, freely floating to transformed (tilde) parameters

    Parameters
    ----------
    vParams         vector, parameters for TempQGARCHII model
    sDistr          string, distribution for the innovations (normal or t)

    Returns
    ----------
    vParamsTrOut    vector, transformed parameters
    """

    vParamsCopy = np.copy(vParams)
    vParamsTrOut = np.zeros(len(vParamsCopy))
    
    # dMu
    vParamsTrOut[0] = vParamsCopy[0] 
    
    # dPhi
    vParamsTrOut[1] = np.arctanh(vParamsCopy[1])
    
    # dOmega0
    vParamsTrOut[2] = np.log(vParamsCopy[2]) 
    
    # dOmega1
    vParamsTrOut[3] = np.log(vParamsCopy[3])
    
    # dAlpha                             
    vParamsTrOut[4] = np.log(vParamsCopy[4])                               

    # dGamma0
    vParamsTrOut[5] = vParamsCopy[5]

    # dBeta [logit]
    vParamsTrOut[6] = logit(vParamsCopy[6])  
    
    if sDistr == 't':
        # dNu
        dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
        vParamsTrOut[-1] = logit((vParamsCopy[-1]-2.1)/dNuMax)
    
    return vParamsTrOut
    
###########################################################
def TempQGARCHIILinkInverse(vParamsTr, sDistr='normal'):
    """
    Purpose
    ----------
    Function of Transformed (tilde) parameters, freely doubleing to untransformed

    Parameters
    ----------
    vParamsTr       transformed parameters for TempQGARCHII model
    sDistr          distribution for the innovations (normal or t)

    Returns
    ----------
    vParamsOut      untransformed parameters
    """
    
    vParamsTrCopy = np.copy(vParamsTr)
    vParamsOut = np.zeros(len(vParamsTrCopy))
    
    # dMu
    vParamsOut[0] = vParamsTrCopy[0] 
    
    # dPhi
    vParamsOut[1] = np.tanh(vParamsTrCopy[1])
    
    # dOmega0
    vParamsOut[2] = np.exp(vParamsTrCopy[2]) 
    
    # dOmega1 
    vParamsOut[3] = np.exp(vParamsTrCopy[3])
    
    # dAlpha                             
    vParamsOut[4] = np.exp(vParamsTrCopy[4])                               

    # dGamma0
    vParamsOut[5] = vParamsTrCopy[5]

    # dBeta [logit]
    vParamsOut[6] = expit(vParamsTrCopy[6])  
    
    if sDistr == 't':
        # dNu
        dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
        vParamsOut[-1] =  dNuMax*expit(vParamsTrCopy[-1]) + 2.1
    
    return vParamsOut

###########################################################
def TempQGARCHIIAvgNLLTr(vParamsTr, vY, vT, sDistr):
    """
    Purpose
    ----------
    Wrapper for calculating the average negative log-likelihood for the TQGARCHII model

    Parameters
    ----------
    vParamsTr       vector, transformed parameters for TempQGARCHII model
    vY              vector, time series data
    vT              vector, time values
    sDistr          string, distribution for the innovations (normal or t)

    Returns
    ----------
    dNLL            double, negative average log-likelihood
    """

    vParamsTrCopy = np.copy(vParamsTr)
    vParamsCopy = TempQGARCHIILinkInverse(vParamsTrCopy, sDistr)
    return TempQGARCHIIAvgNLL(vParamsCopy, vY, vT, sDistr)

###########################################################
def TempQGARCHIIAvgNLL(vParams, vY, vT, sDistr='normal'):
    """
    Purpose
    ----------
    Calculate the average negative log-likelihood for the TempQGARCHII model

    Parameters
    ----------
    vParams         vector, all parameters for AR(p)-TQGARCHII model
    vY              vector, time series data
    vT              vector, time values
    sDistr          string, distribution for the innovations (normal or t)

    Returns
    ----------
    dNLL            double, negative average log-likelihood
    """
    
    # Initialization
    iT = len(vY)
    dMu, dPhi, dOmega0, dOmega1, dAlpha, dGamma0, dBeta =  vParams[:7]
    if sDistr == 't':
        dNu = vParams[-1]

    vEps = np.zeros_like(vY)
    vEps[0] = vY[0]
    vSig2 = np.zeros_like(vEps)
    vSig2[0] = vY.var() 
    vTCap = np.copy(vT)
    vTCap[vTCap==366] = 365
    vSinT = np.abs(np.sin(np.pi/365 * vTCap))
   
    # Filter
    for t in range(1, iT):
        vEps[t] = vY[t] - dMu - dPhi * vY[t-1]
        vSig2[t] = dOmega0  + dOmega1 * vSinT[t] + dAlpha * (vEps[t - 1] - dGamma0)**2 + dBeta * vSig2[t - 1]
        
    if sDistr == 'normal':
        dLL =  -0.5 * (np.log(2 * np.pi) + np.log(vSig2) + vEps**2 / vSig2).mean()
    elif sDistr == 't':
        dLL =np.mean( loggamma((dNu+1)/2) - loggamma(dNu/2) - 1/2* np.log((dNu-2)*np.pi) - 1/2 * np.log(vSig2) - ((dNu+1)/2) * np.log(1+(vEps**2)/((dNu-2)*vSig2)) )

    return -dLL

###########################################################
def ConditionalMeanRegression(vY, vT):
    """
    Purpose
    ----------
    Estimate the TempAR model via Linear Regression 

    Parameters
    ----------
    vY              vector, time series data
    vT              vector, time values
    
    Returns
    ----------
    vParamsOpt      vector, OLS estimates
    vRes            vector, residuals
    """   
    
    iT=len(vY)
    vYLag = np.roll(vY, shift=1) # Construct y_{t-1}
    vYLag[0] = 0 
    mX = np.column_stack((np.ones(iT), vYLag))

    # Estimate the parameters using OLS
    mInvXX = np.linalg.inv(mX.T.dot(mX))
    vXY = mX.T.dot(vY)
    vParamsOpt = mInvXX.dot(vXY)
    vYHat = mX.dot(vParamsOpt)
    vRes = vY - vYHat
    
    return vParamsOpt, vRes

###########################################################
def LocalMeanDaily(vY):
    """
    Purpose
    ----------
    Normalize the vector vY by subtracting the daily average value for each day.
    
    Parameters
    ----------
    vY :            vector, time series data with length iT > 365
    
    Returns
    -------
    vNormalizedY :              vector, normalized time series vector  
    vDailyAveragesExpanded :    vector, daily averages aligned with time series
    """
    
    iT = len(vY)
    
    # Create an index array with the same shape as vY that contains the corresponding day of the year for each element
    vDayIndices = np.arange(iT) % 366 # vY contains at least one year of data 
    
    # Calculate daily averages [use 365 average for day 366]
    vDailyAverages = np.array([np.mean(vY[vDayIndices == np.min((iDay, 365))]) for iDay in range(366)])
    
    # Create an array of daily averages with the same shape as vY using the day index array
    vDailyAveragesExpanded = vDailyAverages[vDayIndices]
    
    # Subtract the daily averages from vY to normalize it
    vNormalizedY = vY - vDailyAveragesExpanded
    
    return vNormalizedY, vDailyAveragesExpanded

###########################################################
def TempQGARCHIIEstim(vY, vT, sDistr='normal', bWarm=False):
    """
    Purpose
    ----------
    Estimate the AR(p)-TQGARCHII model parameters: 
        - Use restricted optimisation, no link functions
        - Use ARCH estimates as starting values unless outside bounds

    Parameters
    ----------
    vY :            vector, time series data
    vT :            vector, time values
    sDistr :        string, distribution for innovations ('normal' or 't')
    bWarm :         boolean, warm start option (placeholder)

    Returns
    -------
    vParamsOpt :             vector, estimated MLE parameters
    dLLOpt :                 double, optimized log-likelihood value
    vYNormalized :           vector, normalized time series
    vDailyAveragesExpanded : vector, daily seasonal mean aligned with time series
    """

    # Initial parameters 
    vYNormalized, vDailyAveragesExpanded =  LocalMeanDaily(vY) # use local means to account for seasonality
    
    vParamsMean, vRes = ConditionalMeanRegression(vYNormalized, vT)
    dMuInit, dPhiInit = vParamsMean
    
    dOmega0Init = 0.1 
    dOmega1Init = 0.3
    dAlphaInit = 0.01
    dGamma0Init = 1
    dBetaInit = 0.9
    dNuInit = 5
    
    # Use TempGARCH for starting values
    if bWarm:
      vParamsInit, dLLGARCH, vYNormalized, vDailyAveragesExpanded =  TempARGARCHEstim(vYNormalized, vT, sDistr)
      dMuInit,dPhiInit,dOmega0Init,dOmega1Init,dAlphaInit,dBetaInit = vParamsInit[:6]
      if sDistr == 't':
        dNuInit = vParamsInit[-1]
        
    if sDistr == 'normal':
        vParamsInit = np.array([dMuInit,dPhiInit,dOmega0Init,dOmega1Init,dAlphaInit,dGamma0Init,dBetaInit])
    elif sDistr == 't':
        vParamsInit = np.array([dMuInit,dPhiInit,dOmega0Init,dOmega1Init,dAlphaInit,dGamma0Init,dBetaInit,dNuInit])

    vParamsInitTr = TempQGARCHIILink(vParamsInit, sDistr)
                    
    oResult = minimize(lambda x: TempQGARCHIIAvgNLLTr(x, vYNormalized, vT, sDistr), vParamsInitTr, method='L-BFGS-B') 
    vParamsOptTr = np.copy(oResult.x)
    
    vParamsOpt = TempQGARCHIILinkInverse(vParamsOptTr, sDistr)
    dLLOpt = -TempQGARCHIIAvgNLL(vParamsOpt, vYNormalized, vT, sDistr) * (len(vY)-1)
    
    return vParamsOpt, dLLOpt, vYNormalized, vDailyAveragesExpanded

###########################################################
######################## FORECASTS ########################
###########################################################

###########################################################
def ARTempForecast(vY, vDailyAveragesExpanded, vTh, vParamsMean):
    """
    Purpose
    ----------
    Produce forecasts for conditional mean

    Parameters
    ----------
    vY :                       vector, time series data
    vDailyAveragesExpanded :   vector, daily averages aligned with time series
    vTh :                      vector, time values for T+h | h = 1, 2, ..., iHorizon
    vParamsMean :              vector, parameters mu and phi

    Returns
    -------
    vMeanPred : vector, mean forecasts for horizons h = 1, ..., iHorizon
    """

    iHorizon = len(vTh)    
    vMeanPredWithoutLocMeans = np.ones(iHorizon+1) * np.nan
    vMeanPredWithoutLocMeans[0] =vY[-1]
    
    dMu, dPhi = vParamsMean
    
    for h in range(1,iHorizon+1):
        vMeanPredWithoutLocMeans[h] =  dMu + dPhi * vMeanPredWithoutLocMeans[h-1]
    
    # Add local means
    vMeanPred = vMeanPredWithoutLocMeans[1:] + vDailyAveragesExpanded[len(vDailyAveragesExpanded)-1-365+1:len(vDailyAveragesExpanded)-1-365+iHorizon+1] 
    
    return vMeanPred

###########################################################
def QGARCHIITempForecast(vY, vT, vTh, vParams, sDistr):
    """
    Purpose
    ----------
    Produce forecasts for conditional variance
    
    Parameters
    ----------
    vY :        vector, time series data
    vT :        vector, time values
    vTh :       vector, time values for T+h | h = 1, 2, ..., iHorizon
    vParams :   vector, parameters of Temp-AR-GARCH model
    sDistr :    string, distribution for innovations ('normal' or 't')
    
    Returns
    -------
    vSig2Pred : vector, conditional variance forecasts for horizons h = 1, ..., iHorizon
    """

    # Initialisation
    iHorizon = len(vTh)   
    iT = len(vY)
    dMu, dPhi, dOmega0, dOmega1, dAlpha, dGamma0, dBeta =  vParams[:7]
    vEps = np.zeros_like(vY)
    vEps[0] = vY[0]
    vSig2 = np.zeros_like(vEps)
    vSig2[0] = vY.var()/10 
    vTCap = np.copy(vT)
    vTCap[vTCap==366] = 365
    vSinT = np.abs(np.sin(np.pi/365 * vTCap))
   
    # Filter
    for t in range(1, iT):
        vEps[t] = vY[t] - dMu - dPhi * vY[t-1]
        vSig2[t] = dOmega0 + dOmega1 * vSinT[t] + dAlpha * (vEps[t - 1] - dGamma0)**2 + dBeta * vSig2[t - 1]

    # Prediction
    vSig2Pred = np.ones(iHorizon+1) * np.nan
    vSig2Pred[0] = vSig2[-1]
        
    for h in range(1,iHorizon+1):
        vSig2Pred[h] = dOmega0 + dOmega1 * np.abs(np.sin(np.pi/365 * np.min((vTh[h-1], 365)))) + (dAlpha + dBeta) * vSig2Pred[h-1] + dAlpha * dGamma0**2
    
    return vSig2Pred[1:]

###########################################################
def QGARCHIIEstimAndForecast(vY, vT, vTh, sDistr, bWarm=False):
    """
    Purpose
    ----------
    Produce all density forecast parameters
    
    Parameters
    ----------
    vY :        vector, time series data
    vT :        vector, time values
    vTh :       vector, time values for T+h | h = 1, 2, ..., iHorizon
    sDistr :    string, distribution for innovations ('normal' or 't')
    bWarm :     boolean, warm start option 
    
    Returns
    -------
    mParamsDF : matrix, predictive parameters [mu, sig2, nu] for each horizon
    """

    # Initialisation
    iHorizon = len(vTh)    
    mParamsDF = np.ones((iHorizon, 3)) * np.nan # horizon x [mu, sig2, nu] 
    
    # Estimate parameters
    vParamsHat, dLL, vYNormalized, vDailyAveragesExpanded = TempQGARCHIIEstim(vY, vT, sDistr, bWarm)
    
    # Collect forecasts
    mParamsDF[:,0] = ARTempForecast(vYNormalized, vDailyAveragesExpanded, vTh, vParamsHat[:2]) 
    mParamsDF[:,1] = QGARCHIITempForecast(vYNormalized, vT, vTh, vParamsHat, sDistr)
    
    if sDistr == 't':
        mParamsDF[:,2] = vParamsHat[-1] * np.ones(iHorizon) 
      
    return mParamsDF
