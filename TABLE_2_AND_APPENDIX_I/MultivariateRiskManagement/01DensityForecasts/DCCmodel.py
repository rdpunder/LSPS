#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: DCC Model
"""

###########################################################
### Imports
import numpy as np

from scipy import stats # pre-programmed random variables
from scipy.optimize import minimize
from numpy import roots
from arch import arch_model

from scipy.special import logit, expit, loggamma, gammaln
from statsmodels.tsa.stattools import acf # for starting values vPhi

# Dependencies
from TGARCHmodel import *
from BivariateT import *

###########################################################
def EnsurePSD(mMatrix):
    """
    Purpose
    ----------
    Ensure a matrix is symmetric positive semidefinite
    
    Parameters
    ----------
    mMatrix :   matrix, input matrix
    
    Returns
    ----------
    mPSD :      symmetric positive semidefinite matrix
    """
    # Symmetrize the matrix
    mSymmetrized = (mMatrix + mMatrix.T) / 2

    # Eigenvalue decomposition
    vEigenvalues, mEigenvectors = np.linalg.eigh(mSymmetrized)
    
    # Set negative eigenvalues to zero
    vEigenvalues[vEigenvalues < 0] = 0

    # Reconstruct the matrix
    mPSD = mEigenvectors @ np.diag(vEigenvalues) @ mEigenvectors.T

    return mPSD

###########################################################
def DCCAvgNLL(vParams, mResiduals, sDistr='normal'):
    """
    Purpose
    ----------
    Compute average negative log-likelihood for the DCC model

    Parameters
    ----------
    vParams :       vector, DCC parameters
    mResiduals :    matrix, GARCH residuals
    sDistr :        string, distribution type ('normal' or 't')

    Returns
    -------
    dNLL :          double, negative average log-likelihood
    """
    
    iT = mResiduals.shape[1]
    mS =np.cov(mResiduals, rowvar=True) # (1/iT) *  mResiduals @ mResiduals.T
    mQ = np.zeros((iT, 2,2))
    mQ[0,:,:] = mS
    vLL = np.zeros(iT)
    dAlphaQ = vParams[0]
    dBetaQ = vParams[1]
    
    if sDistr == 't':
        dNu = vParams[2]

    for t in range(1, iT):
        mQ[t,:,:] = (1 - dAlphaQ - dBetaQ) * mS + dAlphaQ * np.outer(mResiduals[:, t-1], mResiduals[:, t-1]) + dBetaQ * mQ[t-1,:,:]
        mR = np.diag(1/np.sqrt(np.diag(mQ[t,:,:]))) @ mQ[t,:,:] @ np.diag(1 / np.sqrt(np.diag(mQ[t,:,:])))
        if np.linalg.det(mR) == 0:
            return 1e3
        if sDistr == 'normal':
            vLL[t] = -1/2 * np.log(np.linalg.det(mR)) -1/2 * (mResiduals[:, t].reshape((1,2)) @ np.linalg.inv(mR) @ mResiduals[:, t].reshape((2,1))).item() - np.log(2*np.pi)
        elif sDistr == 't':
            vLL[t] = gammaln((dNu + 2)/2) - gammaln(dNu/2) - np.log(np.pi*(dNu - 2)) - 0.5*np.log(np.linalg.det(mR)) - ((dNu + 2)/2)*np.log(1 + ((mResiduals[:, t].reshape(1, 2) @ np.linalg.inv(mR) @ mResiduals[:, t].reshape(2, 1)).item() / (dNu - 2)))
            
    return -vLL.mean()

###########################################################
def DCCLink(vParams, sDistr='normal'):
    """
    Purpose
    ----------
    Transform parameters from unconstrained to constrained space

    Parameters
    ----------
    vParams :    vector, unconstrained parameters
    sDistr :     string, distribution type ('normal' or 't')

    Returns
    -------
    vParamsTrOut :    vector, constrained parameters
    """
    
    vParamsTrOut = np.zeros(len(vParams))
    # Reparameterize dAlphaQ and dBetaQ
    vParamsTrOut[0] = np.log(vParams[0] / (1 - vParams[0] - vParams[1]))
    vParamsTrOut[1] = np.log(vParams[1] / (1 - vParams[0] - vParams[1]))
    # Transform degrees of freedom parameter if t-distribution is used
    if sDistr == 't':
        vParamsTrOut[2] = logit((vParams[-1] - 2.1) / 1e2)  # dNu transformation
    return vParamsTrOut
    
###########################################################
def DCCLinkInverse(vParamsTr, sDistr='normal'):
    """
    Purpose
    ----------
    Transform parameters from constrained to unconstrained space

    Parameters
    ----------
    vParamsTr :    vector, constrained parameters
    sDistr :    string, distribution type ('normal' or 't')

    Returns
    -------
    vParamsOut :    vector, unconstrained parameters
    """
    
    vParamsOut = np.zeros(len(vParamsTr))
    # Transform back dAlphaQ and dBetaQ
    vExpParams = np.exp(vParamsTr[0:2])
    dSumExp = 1 + vExpParams.sum()
    vParamsOut[0] = vExpParams[0] / dSumExp
    vParamsOut[1] = vExpParams[1] / dSumExp
    # Transform back degrees of freedom parameter if t-distribution is used
    if sDistr == 't':
        vParamsOut[2] = 1e2 * expit(vParamsTr[2]) + 2.1
    return vParamsOut

###########################################################
def DCCAvgNLLTr(vParamsTr, mResiduals, sDistr='normal'):
    """
    Purpose
    ----------
    Wrapper to evaluate average negative log-likelihood in transformed space

    Parameters
    ----------
    vParamsTr :    vector, constrained parameters
    mResiduals :    matrix, GARCH residuals
    sDistr :    string, distribution type ('normal' or 't')

    Returns
    -------
    dNLL :    double, negative average log-likelihood
    """

    vParamsTrCopy = np.copy(vParamsTr)
    vParamsCopy = DCCLinkInverse(vParamsTrCopy, sDistr=sDistr)
    return DCCAvgNLL(vParamsCopy, mResiduals, sDistr)

###########################################################
def DCCEstimAndForecast(mY, vH, iP, bGamma=False, bRealized=False, sDistr='normal', mRV=None):
    """
    Purpose
    ----------
    Compute average negative log-likelihood for the bivariate DCC model

    Parameters
    ----------
    vParams :       vector, DCC parameters
    mResiduals :    matrix, GARCH residuals
    sDistr :        string, distribution type ('normal' or 't')

    Returns
    -------
    dNLL :    double, negative average log-likelihood
    """

    # Two-step procedure, first step is to estimate and predict univariate (T/R)GARCH models
    if bRealized:
        # Note: Function not flexible enough for general vH
        mForecastMu1Var1, vResiduals1 = RGARCHEstimAndForecast(mY[:,0], iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, vH=np.array([1,5]), bQML=False, bQMLFull=True, vRV=mRV[:,0], bImpStat=False, bResiduals=True)
        mForecastMu2Var2, vResiduals2 = RGARCHEstimAndForecast(mY[:,1], iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, vH=np.array([1,5]), bQML=False, bQMLFull=True, vRV=mRV[:,1], bImpStat=False, bResiduals=True)

    else:
        mForecastMu1Var1, vResiduals1 = TGARCHEstimAndForecast(mY[:,0], iP, bGamma, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, vH=vH, bQML=False, bQMLFull=False, bImpStat=False, bResiduals=True)
        mForecastMu2Var2, vResiduals2 = TGARCHEstimAndForecast(mY[:,1], iP, bGamma, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, vH=vH, bQML=False, bQMLFull=False, bImpStat=False, bResiduals=True)
    iT = len(vResiduals1)
    mResiduals = np.stack((vResiduals1, vResiduals2))
    mS = np.cov(mResiduals)
   
    # Initial parameters
    if sDistr == 'normal':
        vInitialParams = np.array([0.05, 0.7])
    elif sDistr == 't':
        vInitialParams = np.array([0.05, 0.7, 8])
    
    # Estimation
    vInitialParamsTr = DCCLink(vInitialParams, sDistr=sDistr)
    oResult = minimize(lambda x: DCCAvgNLLTr(x, mResiduals, sDistr=sDistr), vInitialParamsTr, method='L-BFGS-B') 
    vResultTr = np.copy(oResult.x)
    vResult = DCCLinkInverse(vResultTr, sDistr=sDistr)

    # Estimated parameters
    dAlphaQ = vResult[0]
    dBetaQ = vResult[1]
    
    if sDistr == 'normal':
        dNu = np.nan
    elif  sDistr == 't':
        dNu = vResult[2]
    
    mQ = np.zeros((iT+1, 2,2))
    mQ[0,:,:] = mS
    mR = np.zeros_like(mQ)
    mR[0,:,:] = np.diag(1 / np.sqrt(np.diag(mQ[0,:,:]))) @ mQ[0,:,:] @ np.diag(1 / np.sqrt(np.diag(mQ[0,:,:])))
    
    for t in range(1, iT+1):

        mQ[t,:,:] = (1 - dAlphaQ - dBetaQ) * mS + dAlphaQ * np.outer(mResiduals[:, t-1], mResiduals[:, t-1]) + dBetaQ * mQ[t-1,:,:]
        mR[t,:,:] = np.diag(1 / np.sqrt(np.diag(mQ[t,:,:]))) @ mQ[t,:,:] @ np.diag(1 / np.sqrt(np.diag(mQ[t,:,:])))
    
    mQTplus1 = mQ[iT,:,:]
    
    # Assumption: Q is approximately R [see Engle and Sheppard (2001)]
    mSigmaForecast = np.zeros((len(vH),2,2))
    for i in range(len(vH)):  
        iH = vH[i]  
        mQForecast =  mS * (1-(dAlphaQ + dBetaQ)**(iH-1))/(1-(dAlphaQ + dBetaQ)) +  (dAlphaQ + dBetaQ)**(iH-1) * mQTplus1
        mRForecast=  np.diag(1/np.sqrt(np.diag(mQForecast))) @ mQForecast @ np.diag(1 / np.sqrt(np.diag(mQForecast)))
        mD = np.diag(np.sqrt(np.array([mForecastMu1Var1[i,1], mForecastMu2Var2[i,1]])))
        mSigmaForecast[i, :,:] = mD @ mRForecast @ mD

    # Save for each h forecast in one dimension [mu1, mu2, sig11, sig12, sig22, nu]
    mForecast = np.ones((len(vH),6)) * np.nan
    for i in range(len(vH)):
        mForecast[i,:] = np.array([mForecastMu1Var1[i,0],mForecastMu2Var2[i,0],mSigmaForecast[i, 0,0],mSigmaForecast[i, 1,0], mSigmaForecast[i, 1,1], dNu])

    return mForecast
