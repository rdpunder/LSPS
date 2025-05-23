#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: TGARCH Models
"""

###########################################################
### Imports
import numpy as np

from scipy import stats # pre-programmed random variables
from scipy.optimize import minimize
from numpy import roots
from arch import arch_model # used for starting values

from scipy.special import logit, expit, loggamma
from statsmodels.tsa.stattools import acf # for starting values vPhi

###########################################################
def AROLSEstim(vY, iP, iHorizon):
    """
    Purpose
    ----------
    Estimate AR(p) model by OLS

    Parameters
    ----------
    vY :            vector, data
    iP :            integer, number of lags
    iHorizon :      integer, forecast horizon

    Returns
    -------
    vPhihatFull :   vector, estimated parameters including intercept
    vRes :          vector, residuals
    vYPred :        vector, h-step ahead predictions
    """

    # Create the lag matrix X with the constant as the first column
    mX = np.zeros((len(vY)-iP, iP+1))
    mX[:, 0] = np.ones(len(vY)-iP)
    for i in range(iP):
        mX[:, i+1] = vY[iP-i-1:-i-1]
    
    # Extract dependent variable
    vYrem = vY[iP:]
    
    # Estimate AR(p) coefficients by OLS
    vPhihatFull = np.linalg.inv(mX.T @ mX) @ mX.T @ vYrem
    dPhi0 = vPhihatFull[0]
    vPhihat = vPhihatFull[1:]
    
    # Calculate residuals
    vRes = vYrem - mX @ vPhihatFull
    iT=len(vY)

    # Generate h-step ahead forecasts
    vYPred = np.zeros(iT + iHorizon)

    vYPred[:iT] = vY #np.hstack((np.copy(vY),np.zeros(iHorizon)))
    # First prediction y_{T+1|T} [index T is time T+1]
    vYPred[iT-1+1] = dPhi0 + np.dot(vPhihat,  np.flip(vYPred[iT-len(vPhihat):iT]))
    for h in np.arange(2,iHorizon+1):
        vYPred[iT-1+h] = dPhi0 + np.dot(vPhihat,  np.flip(vYPred[iT-len(vPhihat)-1+h:iT-1+h]))

    vYPred = vYPred[iT:]
    
    return vPhihatFull, vRes, vYPred

###########################################################
def ARResiduals(dPhi0, vPhi, vY, iP):
    """
    Purpose
    ----------
    Compute the residuals for the AR(p) model
    
    Inputs
    ----------
    dPhi0 :       double, constant term for AR(p) model
    vPhi :        vector, AR(p) coefficients
    vY :          vector, time series data
    iP :          integer, order of the AR(p) model
    
    Returns
    vEps :        vector, AR(p) residuals
    """
    
    vEps = np.zeros_like(vY)
    vEps[:iP] = vY[:iP]
    for t in range(iP, len(vY)):
        vEps[t] = vY[t] - dPhi0 - np.dot(vPhi, vY[t - iP:t][::-1])
        
    return vEps

###########################################################
def ARTGARCHAvgNLL(vParams, vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal'):
    """
    Purpose
    ----------
    Compute average negative log-likelihood for the AR(p)-TGARCH model

    Parameters
    ----------
    vParams :      array, model parameters
    vY :           array, time series data
    iP :           integer, AR order
    bGamma :       boolean, include gamma term if True
    bRobust :      boolean, use robust GARCH if True
    bTheta :       boolean, estimate shape parameter theta if True
    vThetaFixed :  vector, fixed theta if bTheta is False
    sDistr :       string, distribution ('normal' or 't')

    Returns
    -------
    dNLL :         double, negative average log-likelihood
    """
    
    iT = len(vY)
    dPhi0 = vParams[0]
    vPhi = vParams[1:iP + 1]

    if bGamma:
        dOmega, dAlpha, dGamma, dBeta = vParams[iP+1:iP+5]
    else:
        dOmega, dAlpha, dBeta = vParams[iP+1:iP+4]
        dGamma=0
        
    if bTheta:
        if sDistr == 't':
            if bTheta:
                vTheta = vParams[-1]
            else:
                vTheta = vThetaFixed

    vEps = np.zeros_like(vY)
    vEps[:iP] = vY[:iP]
    for t in range(iP, len(vY)):
        vEps[t] = vY[t] - dPhi0 - np.dot(vPhi, vY[t - iP:t][::-1])

    vSig2 = np.zeros_like(vEps)
    vSig2[0] = vY.var() 
    
    # TGARCH
    if bGamma and not bRobust:
        for t in range(1, iT):
            vSig2[t] = dOmega + dAlpha * (vEps[t - 1] ** 2) + dGamma * (vEps[t - 1] < 0) * (vEps[t - 1] ** 2) + dBeta * vSig2[t - 1]
        
    # Robust GARCH, i.e. GAS-t
    elif not bGamma and bRobust and sDistr=='t':
            vSig2[0] = (dOmega + dAlpha * ((vTheta + 1) * (vTheta / (vTheta - 2))) / (vTheta - 2 + (vTheta / (vTheta - 2)))) / (1 - dBeta)
            for t in range(1, iT):
                vSig2[t] = dOmega + dAlpha * ((vTheta +1) *vEps[t - 1]**2)/((vTheta-2) + (vEps[t - 1]**2/vSig2[t - 1])) + dBeta * vSig2[t - 1]
    else:
        # GARCH
        for t in range(1, iT):
            vSig2[t] = dOmega + dAlpha * vEps[t - 1]**2 + dBeta * vSig2[t - 1]
            if vSig2[t] < 0:
                print(vSig2[t - 1], vEps[t - 1], dOmega, dAlpha, dBeta)
    
    if sDistr == 'normal':
        dLL =  -0.5 * (np.log(2 * np.pi) + np.log(vSig2) + vEps**2 / vSig2).mean()
    elif sDistr == 't':
        dLL =np.mean( loggamma((vTheta+1)/2) - loggamma(vTheta/2) - 1/2* np.log((vTheta-2)*np.pi) - 1/2 * np.log(vSig2) - ((vTheta+1)/2) * np.log(1+(vEps**2)/((vTheta-2)*vSig2)) )

    return -dLL

###########################################################
def ScaledTanh(vX, dA, dB):
    """
    Purpose
    ----------
    Compute bounded hyperbolic tangent transformation

    Parameters
    ----------
    vX :      vector, input values
    dA :      double, lower bound
    dB :      double, upper bound

    Returns
    -------
    vOut :    vector, values within (dA, dB)
    """
    
    return ((dB - dA)/2) * np.tanh(vX) + (dA + dB)/2

###########################################################
def ScaledTanhInv(vX, dA, dB):
    """
    Purpose
    ----------
    Compute inverse of the bounded hyperbolic tangent transformation

    Parameters
    ----------
    vX :      vector, transformed values
    dA :      double, lower bound
    dB :      double, upper bound

    Returns
    -------
    vOut :    vector, original unbounded values
    """
    
    return np.arctanh((2*(vX - (dA + dB)/2)) / (dB - dA))

###########################################################
def ARTGARCHLink(vParams, vY, iP, bGamma=False, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Function of Transformed (tilde) parameters, freely floating to untransformed

    Parameters
    ----------
    vParams :      array, untransformed model parameters
    vY :           array, time series data
    iP :           integer, AR order
    bGamma :       boolean, include gamma term if True
    bTheta :       boolean, use Student-t distribution if True
    sDistr :       string, innovation distribution

    Returns
    -------
    vParamsOut     vector, untransformed parameters
    """

    vParamsCopy = np.copy(vParams)
    vParamsTrOut = np.zeros(len(vParamsCopy))
    # dPhi0
    vParamsTrOut[0] = vParamsCopy[0] 
    # vPhi                                       
    vParamsTrOut[1:iP+1] = np.arctanh(vParamsCopy[1:iP+1])
    # dOmega                                  
    vParamsTrOut[iP+1:iP+2] = np.log(vParamsCopy[iP+1:iP+2])  
    # dAlpha                             
    vParamsTrOut[iP+2:iP+3] = np.log(vParamsCopy[iP+2:iP+3])                               
    # dGamma
    if bGamma: vParamsTrOut[iP+2+1:iP+3+1] = np.log(vParamsCopy[iP+2+1:iP+3+1])    
    # dBeta [really want logit here]
    vParamsTrOut[int(iP+2+bGamma+1):int(iP+3+bGamma+1)] = logit(vParamsCopy[int(iP+2+bGamma+1):int(iP+3+bGamma+1)])   # dBeta
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsTrOut[-1] = logit((vParamsCopy[-1]-2.1)/dNuMax)
    
    return vParamsTrOut
    
###########################################################
def ARTGARCHLinkInverse(vParamsTr, vY, iP, bGamma=False, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Function of Transformed (tilde) parameters, freely floating to untransformed

    Parameters
    ----------
    vParamsTr :    vector, transformed model parameters
    vY :           vector, time series data
    iP :           integer, AR order
    bGamma :       boolean, include gamma term if True
    bTheta :       boolean, use Student-t distribution if True
    sDistr :       string, innovation distribution

    Returns
    -------
    vParamsOut :   vector, original untransformed parameters
    """
    
    vParamsTrCopy = np.copy(vParamsTr)
    vParamsOut = np.zeros(len(vParamsTrCopy))
    # dPhi0
    vParamsOut[0] = vParamsTrCopy[0]
    # vPhi                                       
    vParamsOut[1:iP+1] = np.tanh(vParamsTrCopy[1:iP+1])
    # dOmega                                  
    vParamsOut[iP+1:iP+2] = np.exp(vParamsTrCopy[iP+1:iP+2])  
    # dAlpha                             
    vParamsOut[iP+2:iP+3] = np.exp(vParamsTrCopy[iP+2:iP+3])                               
    # dGamma
    if bGamma: vParamsOut[iP+2+1:iP+3+1] = np.exp(vParamsTrCopy[iP+2+1:iP+3+1]) 
    # dBeta
    vParamsOut[int(iP+2+bGamma+1):int(iP+3+bGamma+1)] = expit(vParamsTrCopy[int(iP+2+bGamma+1):int(iP+3+bGamma+1)])   # dBeta
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsOut[-1] = dNuMax*expit(vParamsTrCopy[-1]) +2.1
    
    return vParamsOut

###########################################################
def ARTGARCHAvgNLLTr(vParamsTr, vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal'):
    """
    Purpose
    ----------
    Wrapper for calculating the average negative log-likelihood for the TGARCH model

    Inputs
    ----------
    vParams :         vector, all parameters for AR(p)-TGARCH model
    vY :              vector, time series data
    iP :              integer, order of the AR(p) model
    bGamma :          boolean, True if TGARCH model includes gamma term, False otherwise
    bRobust :         boolean, True for robust GARCH (GAS-t), False otherwise
    bTheta :          boolean, True if the distribution has shape parameter theta
    vTheta :          vector, shape parameter theta for the distribution
    sDistr :          string, distribution for the innovations (normal or t)

    Returns
    ----------
    dNLL :            double, negative average log-likelihood
    """

    vParamsTrCopy = np.copy(vParamsTr)
    vParamsCopy = ARTGARCHLinkInverse(vParamsTrCopy, vY, iP, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
    return ARTGARCHAvgNLL(vParamsCopy, vY, iP, bGamma, bRobust, bTheta, vThetaFixed, sDistr)

###########################################################
def ARTGARCHEstim(vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, bImpStat=True):
    """
    Purpose
    ----------
    Compute average negative log-likelihood using transformed parameters

    Parameters
    ----------
    vY :           array, time series data
    iP :           integer, AR order
    bGamma :       boolean, include gamma term if True
    bRobust :      boolean, use robust GARCH if True
    bTheta :       boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :  double, fixed shape parameter if bTheta is False
    sDistr :       string, innovation distribution ('normal' or 't')
    bPackage :     boolean, use ARCH package for initial values if True
    bImpStat :     boolean or integer, apply stationarity constraint if True or 2

    Returns
    -------
    dPhi0 :          double, AR model intercept
    vPhi :           array, AR(p) coefficients
    vTGARCHParams :  array, TGARCH parameters
    vTheta :         vector, shape parameter for t-distribution (if applicable)
    """

    # Constraint for stationarity
    if bGamma:
         def StationarityConstraintGARCHTr(vParamsTr):
             vParamsTrCopy = np.copy(vParamsTr)
             vParamsCopy = TGARCHLinkInverse(vParamsTrCopy, vY, bGamma=True, bTheta=False, sDistr='normal')
             return 1-1e-3 - (vParamsCopy[iP+1+1] + vParamsCopy[iP+1+2]/2 + vParamsCopy[iP+1+3])
         
         def StationarityConstraintGARCH(vParams):
             return 1-1e-3 - (vParams[iP+1+1] + vParams[iP+1+2]/2 + vParams[iP+1+3])
    else:
         def StationarityConstraintGARCHTr(vParamsTr):
             vParamsTrCopy = np.copy(vParamsTr)
             vParamsCopy = TGARCHLinkInverse(vParamsTrCopy, vY, bGamma=False, bTheta=False, sDistr='normal')
             return 1-1e-3 - (vParamsCopy[iP+1+1] + vParamsCopy[iP+1+2])
         
         def StationarityConstraintGARCH(vParams):
             return 1-1e-3 - (vParams[iP+1+1] + vParams[iP+1+2])

    # Active constraint
    lCons = [{'type': 'ineq', 'fun': StationarityConstraintGARCHTr}]
   
    # Use ARCH model Sheppard for starting values
    model = arch_model(vY, mean='AR', lags=iP, vol='Garch', p=1, o=bGamma, q=1, power=2.0, dist=sDistr)
    result = model.fit(disp='off')
    
    if bPackage and not bRobust:
        vResult = result.params.values
    else:    
        if StationarityConstraintGARCH(result.params.values) > 0 and 0==1:
            vInitialParams = result.params.values
            if not bTheta and sDistr == 't':
                vInitialParams = vInitialParams[:-1]
        else:
            dInitialPhi0 = vY.mean()
            if iP>0: vInitialARCoefs = acf(vY, nlags=iP, fft=False)[1:]              
            if bGamma:
                vInitialTGARCHParams = [vY.var()/20, 0.05, 0.05, 0.85]
            else:
                vInitialTGARCHParams = [vY.var()/20, 0.05, 0.85]
            
            if bTheta and sDistr == 't':
                if iP>0: 
                    vInitialParams = np.concatenate([[dInitialPhi0], vInitialARCoefs, vInitialTGARCHParams, [4]])
                else:
                    vInitialParams = np.concatenate([[dInitialPhi0], vInitialTGARCHParams, [4]])
            else:
                if iP>0:
                    vInitialParams = np.concatenate([[dInitialPhi0], vInitialARCoefs, vInitialTGARCHParams])
                else:
                    vInitialParams = np.concatenate([[dInitialPhi0], vInitialTGARCHParams])
                    
        vInitialParamsTr = ARTGARCHLink(vInitialParams, vY, iP, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
        oResult = minimize(lambda x: ARTGARCHAvgNLLTr(x, vY, iP, bGamma, bRobust, bTheta, vThetaFixed, sDistr), vInitialParamsTr, method='L-BFGS-B') #L-BFGS-B
        vResultTr = np.copy(oResult.x)
        vResult = ARTGARCHLinkInverse(vResultTr, vY, iP, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
        
        # If stationarity condition is not satisfied, then use restricted optimisation [typically never used]
        if StationarityConstraintGARCH(vResult) <0 and bImpStat:
      
            oResult = minimize(lambda x: ARTGARCHAvgNLLTr(x, vY, iP, bGamma, bRobust, bTheta, vThetaFixed, sDistr), vInitialParamsTr, constraints=lCons, method='SLSQP') #L-BFGS-B
            vResultTr = np.copy(oResult.x)
            vResult = ARTGARCHLinkInverse(vResultTr, vY, iP, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
        
        # Restricted optimisation: Depreciated    
        if bImpStat == 2:
            
            # Bounds for parameters
            dBoundsPhi0 = (-5*np.abs(vY.mean()), 5*np.abs(vY.mean()))
            vBoundsvPhi = [(-1, 1)] * iP
            dBoundsOmega = (1e-6, 5*vY.var())
            vBoundsTGARCHParams = [(0, 1-1e-3)] * (2 + bGamma) 
            lBounds = [dBoundsPhi0] + vBoundsvPhi + [dBoundsOmega] +  vBoundsTGARCHParams
            if bTheta:
                dNuMax= 1e2
                if sDistr == 't':
                    lBounds = lBounds + [(2.1, dNuMax)] 
    
            oResult = minimize(lambda x: ARTGARCHAvgNLLTr(x, vY, iP, bGamma, bRobust, bTheta, vThetaFixed, sDistr),
                               vInitialParamsTr, bounds = lBounds, constraints=[{'type': 'ineq', 'fun': StationarityConstraintGARCH}], method='SLSQP') #L-BFGS-B
            vResult = np.copy(oResult.x)
            
        if not oResult.success:
            print("Warning: Optimization did not converge")
            print(oResult.message)
        
        if StationarityConstraintGARCH(vResult) <0 and bImpStat:
            if bTheta and sDistr=='t': 
                vResult[iP+1:-1] = vInitialParams[iP+1:-1]
            else:
                vResult[iP+1:] = vInitialParams[iP+1:]
    
    dPhi0 = vResult[0]
    vPhi = vResult[1:iP + 1]

    if bGamma:
        vTGARCHParams = vResult[iP+1:iP+5]
    else:
       vTGARCHParams = vResult[iP+1:iP+4]

    if bTheta:
        if sDistr == 't':
            vTheta = np.array([vResult[-1]])
    else:
        vTheta = None

    return dPhi0, vPhi, vTGARCHParams, vTheta

###########################################################
def ARTGARCHEstimRestr(vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False):
    """
    Purpose
    ----------
    Compute parameter estimates for the AR(p)-TGARCH model using restricted maximum likelihood estimation

    Parameters
    ----------
    vY :           vector, time series data
    iP :           integer, AR order
    bGamma :       boolean, include gamma term if True
    bRobust :      boolean, use robust GARCH (GAS-t) if True
    bTheta :       boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :  double, fixed shape parameter if bTheta is False
    sDistr :       string, innovation distribution ('normal' or 't')
    bPackage :     boolean, use ARCH package for initial values if True

    Returns
    -------
    dPhi0 :          double, AR model intercept
    vPhi :           vector, AR(p) coefficients
    vTGARCHParams :  vector, TGARCH parameters
    vTheta :         vector, shape parameter for t-distribution (if applicable)
    """
    
    # Constraint for stationarity
    if bGamma:
        def StationarityConstraintGARCH(vParams):
            return 1-1e-3 - (vParams[iP + 2] + vParams[iP + 3]/2 + vParams[iP + 4])
    else:
        def StationarityConstraintGARCH(vParams):
            return 1-1e-3 - (vParams[iP + 2] + vParams[iP + 3])

    def StationarityConstraintAR(vParams):
        vPhiCoef = np.r_[1, -vParams[1:iP + 1]]  # Form the AR polynomial
        vRootsAR = roots(vPhiCoef)  # Compute the roots of the AR polynomial
        dMinRootModulus = np.min(np.abs(vRootsAR))  # Compute the minimum root modulus
        return dMinRootModulus - 1  # The constraint is satisfied if min_root_modulus > 1
    
    # Active constraint
    lCons = [{'type': 'ineq', 'fun': StationarityConstraintGARCH}]
        
    # Bounds for parameters
    dBoundsPhi0 = (-5*np.abs(vY.mean()), 5*np.abs(vY.mean()))
    vBoundsvPhi = [(-1, 1)] * iP
    dBoundsOmega = (1e-6, 5*vY.var())
    vBoundsTGARCHParams = [(0, 1-1e-3)] * (2 + bGamma) 
    lBounds = [dBoundsPhi0] + vBoundsvPhi + [dBoundsOmega] +  vBoundsTGARCHParams
    if bTheta:
        dNuMax = 1e2
        if sDistr == 't':
            lBounds = lBounds + [(2.1, dNuMax)] 
            
    def CheckBounds(vParams, vBounds):
        vInBounds = [lower < param < upper for param, (lower, upper) in zip(vParams, vBounds)]
        return all(vInBounds)
    
    model = arch_model(vY, mean='AR', lags=iP, vol='Garch', p=1, o=bGamma, q=1, power=2.0, dist=sDistr)
    result = model.fit(disp='off')
    
    if bPackage and not bRobust:
        vResult = result.params.values
    else:    
        #print(CheckBounds(result.params.values, vBounds))
        if StationarityConstraintGARCH(result.params.values) >0 and CheckBounds(result.params.values, lBounds):
            vInitialParams = result.params.values
            if not bTheta and sDistr == 't':
                vInitialParams = vInitialParams[:-1]
        else:
            dInitialPhi0 = vY.mean()
            if iP>0: vInitialARCoefs = acf(vY, nlags=iP, fft=False)[1:]
            
            if bGamma:
                vInitialTGARCHParams = [vY.var()/20, 0.05, 0.05, 0.80]
            else:
                vInitialTGARCHParams = [vY.var()/20, 0.05, 0.80]
            
            if bTheta and sDistr == 't':
                vInitialParams = np.concatenate([[dInitialPhi0], vInitialARCoefs, vInitialTGARCHParams, [4]])
            else:
                 vInitialParams = np.concatenate([[dInitialPhi0], vInitialARCoefs, vInitialTGARCHParams])
    
        oResult = minimize(lambda x: ARTGARCHAvgNLL(x, vY, iP, bGamma, bRobust, bTheta, vThetaFixed, sDistr), vInitialParams, bounds=lBounds, method='L-BFGS-B') #L-BFGS-B
        vResult = oResult.x
        
        if StationarityConstraintGARCH(vResult) <0:
            oResult = minimize(lambda x: ARTGARCHAvgNLL(x, vY, iP, bGamma, bRobust, bTheta, vThetaFixed, sDistr), vInitialParams, bounds=lBounds, constraints=lCons, method='SLSQP') #L-BFGS-B
            vResult = oResult.x
            
        if not oResult.success:
            print("Warning: Optimization did not converge")
            print(oResult.message)
        
        if StationarityConstraintGARCH(vResult) <0:
            print('No stationarity solution', vResult[iP+1:])
            vResult = vInitialParams
            
    if StationarityConstraintGARCH(vResult) <0:
        print('Stationarity error, package:',bPackage)        
    
    dPhi0 = vResult[0]
    vPhi = vResult[1:iP + 1]

    if bGamma:
        vTGARCHParams = vResult[iP+1:iP+5]
    else:
       vTGARCHParams = vResult[iP+1:iP+4]

    if bTheta:
        if sDistr == 't':
            vTheta = np.array([vResult[-1]])
    else:
        vTheta = None

    return dPhi0, vPhi, vTGARCHParams, vTheta

###########################################################
def TGARCHAvgNLL(vParams, vEps, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal'):
    """
    Purpose
    ----------
    Compute average negative log-likelihood for the TGARCH or robust GARCH model

    Parameters
    ----------
    vParams :       vector, model parameters including variance dynamics (and shape if applicable)
    vEps :          vector, residuals from the mean equation
    bGamma :        boolean, include gamma term if True
    bRobust :       boolean, use robust GARCH (GAS-t) if True
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :   double, fixed shape parameter if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')

    Returns
    -------
    dNLL :          double, negative average log-likelihood
    """
    
    iT = len(vEps)
    
    if bGamma:
        dOmega, dAlpha, dGamma, dBeta = vParams[0:4]
    else:
        dOmega, dAlpha, dBeta = vParams[0:3]
        dGamma=0
        
    if bTheta:
        if sDistr == 't':
            if bTheta:
                vTheta = vParams[-1]
            else:
                vTheta = vThetaFixed

    vSig2 = np.zeros_like(vEps)
    vSig2[0] = vEps.var() 
    
    # TGARCH
    if bGamma and not bRobust:
        for t in range(1, iT):
            vSig2[t] = dOmega + dAlpha * (vEps[t - 1] ** 2) + dGamma * (vEps[t - 1] < 0) * (vEps[t - 1] ** 2) + dBeta * vSig2[t - 1]
        
    # Robust GARCH, i.e. GAS-t
    elif not bGamma and bRobust and sDistr=='t':
            vSig2[0] = (dOmega + dAlpha * ((vTheta + 1) * (vTheta / (vTheta - 2))) / (vTheta - 2 + (vTheta / (vTheta - 2)))) / (1 - dBeta)
            for t in range(1, iT):
                vSig2[t] = dOmega + dAlpha * ((vTheta +1) *vEps[t - 1]**2)/((vTheta-2) + (vEps[t - 1]**2/vSig2[t - 1])) + dBeta * vSig2[t - 1]
    else:
        # GARCH
        for t in range(1, iT):
            vSig2[t] = dOmega + dAlpha * vEps[t - 1]**2 + dBeta * vSig2[t - 1]

    if sDistr == 'normal':
        dLL =  -0.5 * (np.log(2 * np.pi) + np.log(vSig2) + vEps**2 / vSig2).mean()
    elif sDistr == 't':
        dLL =np.mean( loggamma((vTheta+1)/2) - loggamma(vTheta/2) - 1/2* np.log((vTheta-2)*np.pi) - 1/2 * np.log(vSig2) - ((vTheta+1)/2) * np.log(1+(vEps**2)/((vTheta-2)*vSig2)) )

    return -dLL

###########################################################
def TGARCHLink(vParams, vY, bGamma=False, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Compute transformed parameter vector for the TGARCH model to facilitate unconstrained optimization

    Parameters
    ----------
    vParams :       vector, original model parameters
    vY :            vector, time series data
    bGamma :        boolean, include gamma term if True
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    sDistr :        string, innovation distribution ('normal' or 't')

    Returns
    -------
    vParamsTrOut :  vector, transformed model parameters
    """
    
    vParamsCopy = np.copy(vParams)
    vParamsTrOut = np.zeros(len(vParamsCopy))
    
    # dOmega                                  
    vParamsTrOut[0] = np.log(vParamsCopy[0])  
    # dAlpha                             
    vParamsTrOut[1] = logit(vParamsCopy[1])                               
    # dGamma
    if bGamma: vParamsTrOut[2] = np.log(vParamsCopy[2])    
    # dBeta
    vParamsTrOut[int(2+bGamma)] = logit(vParamsCopy[int(2+bGamma)])   # dBeta
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsTrOut[-1] = logit((vParamsCopy[-1]-2.1)/dNuMax)
    
    return vParamsTrOut
    
###########################################################
def TGARCHLinkInverse(vParamsTr, vY, bGamma=False, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Compute untransformed TGARCH parameters from transformed vector for use in likelihood evaluation

    Parameters
    ----------
    vParamsTr :     vector, transformed model parameters
    vY :            vector, time series data
    bGamma :        boolean, include gamma term if True
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    sDistr :        string, innovation distribution ('normal' or 't')

    Returns
    -------
    vParamsOut :    vector, original untransformed model parameters
    """

    vParamsTrCopy = np.copy(vParamsTr)
    vParamsOut = np.zeros(len(vParamsTrCopy))
    
    # dOmega                                  
    vParamsOut[0] = np.exp(vParamsTrCopy[0])  
    # dAlpha                             
    vParamsOut[1] = expit(vParamsTrCopy[1])                               
    # dGamma
    if bGamma: vParamsOut[2] = np.exp(vParamsTrCopy[2]) 
    # dBeta
    vParamsOut[int(2+bGamma)] = expit(vParamsTrCopy[int(2+bGamma)])   # dBeta
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsOut[-1] = dNuMax*expit(vParamsTrCopy[-1]) +2.1
    
    return vParamsOut

###########################################################
def TGARCHAvgNLLTr(vParamsTr, vY, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal'):
    """
    Purpose
    ----------
    Compute average negative log-likelihood for the AR(p)-TGARCH model
    
    Parameters
    ----------
    vParams :       vector, model parameters 
    vY :            vector, time series data
    iP :            integer, AR order
    bGamma :        boolean, include gamma term if True
    bRobust :       boolean, use robust GARCH if True
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vTheta :        vector, fixed shape parameters if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')
    
    Returns
    -------
    dNLL :          double, negative average log-likelihood
    """

    vParamsTrCopy = np.copy(vParamsTr)
    vParamsCopy = TGARCHLinkInverse(vParamsTrCopy, vY, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
    return TGARCHAvgNLL(vParamsCopy, vY, bGamma, bRobust, bTheta, vThetaFixed, sDistr)

###########################################################
def TGARCHEstim(vEps, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, bImpStat=True):
    """
    Purpose
    ----------
    Estimate AR(p)-TGARCH model parameters using restricted optimisation
    
    - No use of link functions
    - ARCH-based starting values unless outside bounds
    
    Parameters
    ----------
    vY :            vector, time series data
    iP :            integer, AR order
    bGamma :        boolean, include gamma term if True
    bRobust :       boolean, use robust GARCH if True
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vTheta :        double, fixed shape parameters if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')
    bPackage :      boolean, use ARCH package for starting values if True
    
    Returns
    -------
    vTGARCHParams : vector, TGARCH model parameters
    vTheta :        vector, shape parameters 
    """

    # Constraint for stationarity
    if bGamma:
         def StationarityConstraintGARCHTr(vParamsTr):
             vParamsTrCopy = np.copy(vParamsTr)
             vParamsCopy = TGARCHLinkInverse(vParamsTrCopy, vEps, bGamma=True, bTheta=False, sDistr='normal')
             return 1-1e-3 - (vParamsCopy[1] + vParamsCopy[2]/2 + vParamsCopy[3])
         
         def StationarityConstraintGARCH(vParams):
             return 1-1e-3 - (vParams[1] + vParams[2]/2 + vParams[3])
    else:
         def StationarityConstraintGARCHTr(vParamsTr):
             vParamsTrCopy = np.copy(vParamsTr)
             vParamsCopy = TGARCHLinkInverse(vParamsTrCopy, vEps, bGamma=False, bTheta=False, sDistr='normal')
             return 1-1e-3 - (vParamsCopy[1] + vParamsCopy[2])
         
         def StationarityConstraintGARCH(vParams):
             return 1-1e-3 - (vParams[1] + vParams[2])
 
    # Active constraint
    lCons = [{'type': 'ineq', 'fun': StationarityConstraintGARCH}]
   
    # Use ARCH model Sheppard for starting values
    model = arch_model(vEps, mean='Zero', vol='Garch', p=1, o=bGamma, q=1, power=2.0, dist=sDistr)
    result = model.fit(disp='off')
    
    if bPackage and not bRobust:
        vResult = result.params.values
    else:    
        if StationarityConstraintGARCH(result.params.values)  > 0 and 0==1:
            
            vInitialParams = result.params.values
            if not bTheta and sDistr == 't':
                vInitialParams = vInitialParams[:-1]
        else:
            if bGamma:
                vInitialTGARCHParams = [vEps.var()/20, 0.05, 0.05, 0.85]
            else:
                vInitialTGARCHParams = [vEps.var()/20, 0.05, 0.85]
            
            if bTheta and sDistr == 't':
                vInitialParams = np.concatenate([vInitialTGARCHParams, [4]])
            else:
                 vInitialParams = np.concatenate([vInitialTGARCHParams])
    
        vInitialParamsTr = TGARCHLink(vInitialParams, vEps, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
        oResult = minimize(lambda x: TGARCHAvgNLLTr(x, vEps, bGamma, bRobust, bTheta, vThetaFixed, sDistr), vInitialParamsTr, method='L-BFGS-B') #L-BFGS-B
        vResultTr = np.copy(oResult.x)
 
        vResult = TGARCHLinkInverse(vResultTr, vEps, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
        
        # If stationarity condition is not satisfied, then use restricted optimisation
        if StationarityConstraintGARCH(vResult) <0 and bImpStat:
            
            oResult = minimize(lambda x: TGARCHAvgNLLTr(x, vEps, bGamma, bRobust, bTheta, vThetaFixed, sDistr), vInitialParamsTr, constraints=lCons, method='SLSQP') #L-BFGS-B
            vResultTr = np.copy(oResult.x)
            vResult = TGARCHLinkInverse(vResultTr, vEps, bGamma=bGamma, bTheta=bTheta, sDistr=sDistr)
            
        if not oResult.success:
            print("Warning: Optimization did not converge")
            print(oResult.message)
        
        if StationarityConstraintGARCH(vResult) <0 and bImpStat:
            if bTheta and sDistr=='t': 
                vResult[:-1] = vInitialParams[:-1]
            else:
                vResult = vInitialParams
            
    if StationarityConstraintGARCH(vResult) <0 and bImpStat:
        print('Stationarity error, package:',bPackage)        
    
    if bGamma:
        vTGARCHParams = vResult[:4]
    else:
       vTGARCHParams = vResult[:3]

    if bTheta:
        if sDistr == 't':
            vTheta = np.array([vResult[-1]])
    else:
        vTheta = None

    return vTGARCHParams, vTheta

###########################################################
def TGARCHEstimStdQML(vEps, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, bImpStat=True):
    """
    Purpose
    ----------
    Estimate TGARCH model parameters using standardized QML
    
    Parameters
    ----------
    vEps :          vector, residuals from mean equation
    bGamma :        boolean, include gamma term if True
    bRobust :       boolean, use robust GARCH if True
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :   vector, fixed shape parameters if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')
    bPackage :      boolean, use ARCH package for starting values if True
    bImpStat :      boolean, impose stationarity condition if True
    
    Returns
    -------
    vTGARCHParams : vector, TGARCH model parameters
    vTheta :        double, estimated shape parameter (if applicable)
    """

    vTGARCHParams_est, vTheta_est  = TGARCHEstim(vEps, bGamma=bGamma, bRobust=bRobust, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False)
    dOmega = vTGARCHParams_est[0]
    dAlpha = vTGARCHParams_est[1]
    if bGamma: dGamma =  vTGARCHParams_est[2]
    dBeta = vTGARCHParams_est[int(2+bGamma)]
    iT = len(vEps)
    
    vSig2Star = np.zeros_like(vEps)
    vSig2Star[0] = vEps.var() 
    
    # TGARCH
    if bGamma and not bRobust:
        for t in range(1, iT):
            vSig2Star[t] = dOmega + dAlpha * (vEps[t - 1] ** 2) + dGamma * (vEps[t - 1] < 0) * (vEps[t - 1] ** 2) + dBeta * vSig2Star[t - 1]
        
    # Robust GARCH, i.e. GAS-t
    elif not bGamma and bRobust and sDistr=='t':
            print('Error: Full QML not supported for GAS-t')
    else:
        # GARCH
        for t in range(1, iT):
            vSig2Star[t] = dOmega + dAlpha * vEps[t - 1]**2 + dBeta * vSig2Star[t - 1]
            
    # Residuals
    vResiduals = vEps/np.sqrt(vSig2Star)
    
    dNuStartTr = np.log(5 - 2.1)
    res= minimize(AvgNLnLStandTrStd, dNuStartTr, args=(vResiduals), method="BFGS", options={'disp': False})
    dNuStarTr = np.copy(res.x)       
    dNuStar = np.exp(dNuStarTr) + 2.1
    
    return vTGARCHParams_est, np.array(dNuStar)

###########################################################
def TGARCHSim(iT, iP, dPhi0, vPhi, dOmega, dAlpha, dGamma, dBeta, oDist):
    """
    Purpose
    ----------
    Generate simulated time series from an AR(p)-TGARCH(1,1) model
    
    Parameters
    ----------
    iT :        integer, length of the simulated time series
    iP :        integer, order of the AR(p) model
    dPhi0 :     double, constant term in the AR equation
    vPhi :      vector, AR(p) coefficients
    dOmega :    double, GARCH intercept parameter
    dAlpha :    double, ARCH parameter
    dGamma :    double, asymmetric term coefficient (TGARCH)
    dBeta :     double, GARCH parameter
    oDist :     object, distribution for the innovations 
    
    Returns
    -------
    vY :        vector, simulated time series
    vEps :      vector, AR residuals
    vSig2 :     vector, conditional variances
    """

    bGamma = dGamma == 0
    np.random.seed(1234)
    vY = np.zeros(iT + iP)
    vEpsStand =  oDist.rvs(size=iT+iP)
    vSig2 = np.zeros(iT+iP)
    vEps = np.zeros(iT+iP)

    # Simulate TGARCH(1,1,1) process
    vSig2[0] = dOmega / (1 - dAlpha - 1/2*dGamma * bGamma - dBeta) #np.mean(vEps ** 2)
    vEps[0] = np.sqrt(vSig2[0]) * vEpsStand[0]
    for t in range(1, iT+iP): 
        vSig2[t] = dOmega + dAlpha * vEps[t - 1]** 2 + dGamma * bGamma * (vEps[t - 1] < 0) * (vEps[t - 1] ** 2) + dBeta * vSig2[t - 1]
        vEps[t] = np.sqrt(vSig2[t]) * vEpsStand[t]
    
    vY[:iP] = dPhi0/(1-vPhi.sum()) + vEps[:iP]
    
    for t in range(iP, iT+iP):
        vY[t] = dPhi0 + (np.dot(vPhi, vY[t - iP:t][::-1]) if iP >0 else 0) + vEps[t]
    
    return vY[iP:], vEps[iP:], vSig2[iP:]

###########################################################
def ARTGARCHForecast(vY, dPhi0, vPhi, vTGARCHParams, vTheta=None, iHorizon=10, bGamma=False, bResiduals=False):
    """
    Purpose
    ----------
    Compute out-of-sample forecasts from an AR(p)-TGARCH model
    
    Parameters
    ----------
    vY :              vector, observed time series data
    dPhi0 :           double, constant term in the AR equation
    vPhi :            vector, AR(p) coefficients
    vTGARCHParams :   vector, TGARCH parameters (omega, alpha, [gamma,] beta)
    vTheta :          vector, shape parameters for the innovation distribution (optional)
    iHorizon :        integer, forecast horizon
    bGamma :          boolean, include gamma term if True
    bResiduals :      boolean, return standardized residuals if True
    
    Returns
    -------
    mForecast :       array, contains forecasted means, variances, and shape parameters
    vResiduals :      vector, standardized residuals (only if bResiduals is True)
    """
    
    iT = len(vY)
    iP = len(vPhi)
    if bGamma:
        dOmega, dAlpha, dGamma, dBeta = vTGARCHParams
    else:
        dOmega, dAlpha, dBeta = vTGARCHParams
        dGamma = 0
    mForecast = np.zeros((iT + iHorizon, 2 + (len(vTheta) if vTheta else 0)))
    mForecast[:iT, 0] = vY
    mForecast[:iT, 1] = np.nan

    # Forecast mean and variance
    for h in range(1, iHorizon + 1):
        mForecast[iT - 1 + h, 0] = dPhi0 + np.dot(vPhi, mForecast[iT - 1 + h - iP:iT - 1 + h, 0][::-1])

    # Filter volatility 
    vEps = np.zeros_like(vY)
    vEps[:iP] = vY[:iP]
    for t in range(iP, len(vY)):
        vEps[t] = vY[t] - dPhi0 - np.dot(vPhi, vY[t - iP:t][::-1])

    vSig2 = np.zeros_like(vEps)
    vSig2[0] = vY.var() #dOmega / (1 - dAlpha - bGamma * dGamma/2 - dBeta)
    
   
    # For symmetric distributions around zero: 
    for t in range(1, iT):
       vSig2[t] = dOmega + dAlpha * (vEps[t - 1] ** 2) + dGamma * (vEps[t - 1] < 0) * (vEps[t - 1] ** 2) + dBeta * vSig2[t - 1]
    dSigmaT = vSig2[iT-1] 
    dEpsT = vEps[iT-1]
    vResiduals = vEps / np.sqrt(vSig2)
    
    for h in range(1, iHorizon + 1):  
        dSumTerm = dOmega * (1-(dAlpha + bGamma*dGamma/2 + dBeta)**h)/(1-(dAlpha + bGamma*dGamma/2 + dBeta))
        mForecast[iT - 1 + h, 1]=  dSumTerm + (dAlpha + bGamma*dGamma/2 +  dBeta)**(h-1) * (dAlpha * dEpsT**2 + dBeta * dSigmaT + dGamma * bGamma * (dEpsT < 0) * (dEpsT ** 2))
   
    # Store static distribution parameters
    if vTheta:
        for i, dTheta in enumerate(vTheta):
            mForecast[:, 2 + i] = dTheta

    if bResiduals:
         return mForecast[iT:], vResiduals
    else:
         return mForecast[iT:]

###########################################################
def TGARCHForecast(vEps, vTGARCHParams, vTheta=None, iHorizon=10, bGamma=False, bResiduals=False):
    """
    Purpose
    ----------
    Compute out-of-sample variance forecasts from a TGARCH or robust GARCH model
    
    Parameters
    ----------
    vEps :            vector, residuals from the mean equation
    vTGARCHParams :   vector, TGARCH parameters (omega, alpha, [gamma,] beta)
    vTheta :          vector, shape parameters for the innovation distribution (optional)
    iHorizon :        integer, forecast horizon
    bGamma :          boolean, include gamma term if True
    bResiduals :      boolean, return standardized residuals if True
    
    Returns
    -------
    mForecast :       array, contains forecasted conditional variances and shape parameters
    vResiduals :      vector, standardized residuals (only if bResiduals is True)
    """

    iT = len(vEps)

    if bGamma:
        dOmega, dAlpha, dGamma, dBeta = vTGARCHParams
    else:
        dOmega, dAlpha, dBeta = vTGARCHParams
        dGamma = 0
    mForecast = np.zeros((iT + iHorizon, 1 + (len(vTheta) if vTheta else 0)))
    mForecast[:iT, 0] = np.nan

    vSig2 = np.zeros_like(vEps)
    vSig2[0] = vEps.var() 
    
    # For symmetric distributions around zero: 
    for t in range(1, iT):
       vSig2[t] = dOmega + dAlpha * (vEps[t - 1] ** 2) + dGamma * (vEps[t - 1] < 0) * (vEps[t - 1] ** 2) + dBeta * vSig2[t - 1]
    dSigmaT = vSig2[iT-1] 
    dEpsT = vEps[iT-1]
    vResiduals = vEps / np.sqrt(vSig2)
    
    for h in range(1, iHorizon + 1):  
        dSumTerm = dOmega * (1-(dAlpha + bGamma*dGamma/2 + dBeta)**h)/(1-(dAlpha + bGamma*dGamma/2 + dBeta))
        mForecast[iT - 1 + h, 0]=  dSumTerm + (dAlpha + bGamma*dGamma/2 +  dBeta)**(h-1) * (dAlpha * dEpsT**2 + dBeta * dSigmaT + dGamma * bGamma * (dEpsT < 0) * (dEpsT ** 2))
   
    # Store static distribution parameters
    if vTheta:
        for i, dTheta in enumerate(vTheta):
            mForecast[:, 1 + i] = dTheta
            
    if bResiduals:
        return mForecast[iT:], vResiduals
    else:
        return mForecast[iT:]

###########################################################
def TGARCHEstimAndForecast(vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, vH=np.array([1,5]), bQML=True, bQMLFull=True, bImpStat=True, bResiduals=False):
    """
    Purpose
    ----------
    Estimate and forecast the AR(p)-TGARCH model 
    
    Parameters
    ----------
    vY :             vector, time series data
    iP :             integer, AR order
    bGamma :         boolean, include gamma term if True
    bRobust :        boolean, use robust GARCH (GAS-t) if True
    bTheta :         boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :    double, fixed shape parameter if bTheta is False
    sDistr :         string, innovation distribution ('normal' or 't')
    bPackage :       boolean, use ARCH package estimation if True
    vH :             vector, horizons at which to evaluate forecast
    bQML :           boolean, use QML-type two-step estimator if True
    bQMLFull :       boolean, use full QML estimation including Student-t shape
    bImpStat :       boolean, enforce stationarity via restricted optimisation if True
    bResiduals :     boolean, return standardized residuals if True
    
    Returns
    -------
    mForecast :      array, forecasted means, conditional variances, and distribution parameters
    vResiduals :     vector, standardized residuals (only if bResiduals is True)
    """


    iHorizon = int(vH.max())
    if bPackage and not bRobust:
        model = arch_model(vY, mean='AR', lags=iP, vol='Garch', p=1, o=bGamma, q=1, power=2.0, dist=sDistr)
        result = model.fit(disp='off')
        vResult = result.params.values
        dPhi0_est = vResult[0]
        vPhi_est = vResult[1:iP + 1]

        if bGamma:
            vTGARCHParams_est = vResult[iP+1:iP+5]
        else:
           vTGARCHParams_est = vResult[iP+1:iP+4]

        if bTheta:
            if sDistr == 't':
                vTheta_est = np.array([vResult[-1]])
                mForecastAll = np.zeros((iHorizon, 3))
                mForecastAll[:,2] = vTheta_est
        else:
            vTheta_est = None
            mForecastAll = np.zeros((iHorizon, 2))
        forecasts = result.forecast(horizon=iHorizon, reindex=False)
        # Get the AR(p) forecasts
        mForecastAll[:,0]  = forecasts.mean.iloc[-1].values
        mForecastAll[:,1] = forecasts.variance.iloc[-3:]
            
    else:
        if bQML:
            
            vPhihat, vYRes, vMuHat = AROLSEstim(vY, iP, iHorizon)
            if bQMLFull and sDistr == 't':
                vTGARCHParams_est, vTheta_est = TGARCHEstimStdQML(vYRes, bGamma=bGamma, bRobust=bRobust, bTheta=bTheta, vThetaFixed=None, sDistr='t', bPackage=False)
               
            else:
                vTGARCHParams_est, vTheta_est  = TGARCHEstim(vYRes, bGamma=bGamma, bRobust=bRobust, bTheta=bTheta, vThetaFixed=vThetaFixed, sDistr=sDistr, bPackage=False, bImpStat=bImpStat)
               
            mForecastAll = np.zeros((iHorizon,2+bTheta))
            mForecastAll[:,0] = vMuHat
            
            if bResiduals:
                mForecastAll[:,1:], vResiduals = TGARCHForecast(vYRes, vTGARCHParams_est, vTheta=vTheta_est, iHorizon=iHorizon, bGamma=bGamma, bResiduals=True)
            else:
                mForecastAll[:,1:] = TGARCHForecast(vYRes, vTGARCHParams_est, vTheta=vTheta_est, iHorizon=iHorizon, bGamma=bGamma, bResiduals=False)
        else:
            
            dPhi0_est, vPhi_est, vTGARCHParams_est, vTheta_est  = ARTGARCHEstim(vY, iP, bGamma=bGamma, bRobust=bRobust, bTheta=bTheta, vThetaFixed=vThetaFixed, sDistr=sDistr, bPackage=False, bImpStat=bImpStat)
            if bResiduals:
                mForecastAll, vResiduals = ARTGARCHForecast(vY, dPhi0_est, vPhi_est, vTGARCHParams_est, vTheta=vTheta_est, iHorizon=iHorizon, bGamma=bGamma, bResiduals=True)
            else:
                mForecastAll = ARTGARCHForecast(vY, dPhi0_est, vPhi_est, vTGARCHParams_est, vTheta=vTheta_est, iHorizon=iHorizon, bGamma=bGamma, bResiduals=False)
   
    mForecast = np.zeros((len(vH), mForecastAll.shape[1]))
    iCount = 0
    for h in vH:
        mForecast[iCount,:] = mForecastAll[h-1,:]
        iCount += 1
        
    if bResiduals:
        return mForecast, vResiduals
    else:    
        return mForecast

###########################################################
###########################################################
###########################################################
################### REALIZED VOLATILITY ###################
###########################################################
###########################################################
###########################################################

###########################################################
def RGARCHNoCUpdateEq(vParams, vY, vRV):
    """
    Purpose
    ----------
    Compute conditional variances from an RGARCH-type model with realized variance
    
    Parameters
    ----------
    vParams :     vector, model parameters [dOmega, dAlpha, dBeta, dXi, dPhi, dTau, dKappa, dSig2u]
    vY :          vector, time series data
    vRV :         vector, realized variances
    
    Returns
    -------
    vSig2 :       vector, conditional variances implied by the model
    """

    iT= len(vY) 
    dOmega, dAlpha, dBeta, dXi, dPhi, dTau, dKappa, dSig2u = vParams[0:8] 
    
    vSig2= np.zeros(iT+1)
    vSig2[0]= vY.var() 
    
    for t in range(1,iT+1):
        vSig2[t]= dOmega + dAlpha * vRV[t-1]  +  dBeta * vSig2[t-1] 

    return vSig2

###########################################################
def RGARCHNoC11Pred(vRGARCHParams, vY, vH, vRV, bResiduals=False):
    """
    Purpose
    ----------
    Compute h-step ahead volatility forecasts for the RGARCH(1,1) model
    
    Parameters
    ----------
    vRGARCHParams :  vector, model parameters [dOmega, dAlpha, dBeta, dXi, dPhi, dTau, dKappa, dSig2u]
    vY :             vector, time series data
    vH :             vector, forecast horizons
    vRV :            vector, realized variances
    bResiduals :     boolean, return standardized residuals if True
    
    Returns
    -------
    vSig2Pred :      vector, h-step ahead volatility forecasts
    vResiduals :     vector, standardized residuals (if bResiduals is True)
    """

    iT = len(vY)
    vSig2 = RGARCHNoCUpdateEq(vRGARCHParams, vY, vRV)
    vResiduals = vY/np.sqrt(vSig2[:-1])

    dOmega, dAlpha, dBeta, dXi, dPhi, dTau, dKappa, dSig2u = vRGARCHParams[0:8] 
    
    dConst = dOmega + dAlpha * dXi
    dPers = dAlpha * dPhi  + dBeta

    vSig2Pred = dConst * (1-dPers**(vH-1))/(1-dPers) + dPers**(vH-1) * vSig2[iT]  
    
    if bResiduals:
        return vSig2Pred, vResiduals
    else:
        return vSig2Pred

###########################################################
def RGARCHLink(vParams, vY, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Compute untransformed parameters for the AR(p)-TGARCH model from transformed (tilde) parameters
    
    Parameters
    ----------
    vParamsTr :   vector, transformed model parameters
    iP :          integer, AR order
    bGamma :      boolean, include gamma term if True
    bRobust :     boolean, use robust GARCH (GAS-t) if True
    bTheta :      boolean, estimate shape parameter vector vTheta if True
    vTheta :      double, fixed shape parameter if bTheta is False
    sDistr :      string, innovation distribution ('normal' or 't')
    
    Returns
    -------
    vParamsOut :  vector, untransformed model parameters
    """

    vParamsCopy = np.copy(vParams)
    vParamsTrOut = np.zeros(len(vParamsCopy))
    
    # dOmega                                  
    vParamsTrOut[0] = np.log(vParamsCopy[0])  
    # dAlpha                             
    vParamsTrOut[1] = np.log(vParamsCopy[1])                               
    # dBeta
    vParamsTrOut[2] = logit(vParamsCopy[2])
    #dXi
    vParamsTrOut[3] = np.log(vParamsCopy[3])
    #dPhi
    vParamsTrOut[4] = np.log(vParamsCopy[4])
    #dTau
    vParamsTrOut[5] = vParamsCopy[5]
    #dKappa
    vParamsTrOut[6] = np.log(vParamsCopy[6])
    #dSig2u
    vParamsTrOut[7] = np.log(vParamsCopy[7])
    
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsTrOut[-1] = logit((vParamsCopy[-1]-2.1)/dNuMax)
    
    return vParamsTrOut
    
###########################################################
def RGARCHLinkInverse(vParamsTr, vY, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Compute untransformed RGARCH model parameters from transformed (tilde) parameters
    
    Parameters
    ----------
    vParamsTr :   vector, transformed model parameters
    vY :          vector, time series data
    bTheta :      boolean, estimate shape parameter vector vTheta if True
    sDistr :      string, innovation distribution ('normal' or 't')
    
    Returns
    -------
    vParamsOut :  vector, untransformed model parameters including variance and measurement equation terms
    """

    vParamsTrCopy = np.copy(vParamsTr)
    vParamsOut = np.zeros(len(vParamsTrCopy))
    
    # dOmega                                  
    vParamsOut[0] = np.exp(vParamsTrCopy[0])
    # dAlpha                             
    vParamsOut[1] = np.exp(vParamsTrCopy[1])                               
    # dBeta
    vParamsOut[2] = expit(vParamsTrCopy[2])   
    #dXi
    vParamsOut[3] = np.exp(vParamsTrCopy[3])
    #dPhi
    vParamsOut[4] = np.exp(vParamsTrCopy[4])
    #dTau
    vParamsOut[5] = vParamsTrCopy[5]
    #dKappa
    vParamsOut[6] = np.exp(vParamsTrCopy[6])
    #dSig2u
    vParamsOut[7] = np.exp(vParamsTrCopy[7])
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsOut[-1] = dNuMax*expit(vParamsTrCopy[-1]) +2.1
    
    return vParamsOut

################################################################
##################### AR-RGARCH-N (Full QML) ###################
################################################################

###########################################################
def AvgNLnLRGARCHNoCNorm(vParams, vY, vRV):
    """
    Purpose
    ----------
    Compute negative average log-likelihood for the RGARCH model with normal innovations
    
    Parameters
    ----------
    vParams :     vector, model parameters 
    vY :          vector, time series data
    vRV :         vector, realized volatility observations
    
    Returns
    -------
    dNLL :        double, negative average log-likelihood of the joint model
    """

    dXi, dPhi, dTau, dKappa, dSig2u = vParams[3:] 
    
    # Volatility    
    vSig2 = RGARCHNoCUpdateEq(vParams, vY, vRV)[:-1]
    
    # Likelihood GARCH
    vLLGARCH= -0.5*(np.log(2*np.pi) + np.log(vSig2) + ((vY)**2)/vSig2)
    dALLGARCH= np.mean(vLLGARCH, axis= 0)
    
    # Realized volatility
    vRVCondMean= dXi + dPhi * vSig2 + dTau * (vY)/np.sqrt(vSig2)  + dKappa * ((vY)**2/vSig2-1)
  
    # Likelihood RV
    vLLRV= -0.5*(np.log(2*np.pi) + np.log(dSig2u) + ((vRV-vRVCondMean)**2)/dSig2u)
    dALLRV= np.mean(vLLRV, axis= 0)
        
    return -(dALLGARCH+dALLRV)

###########################################################
def AvgNLnLRGARCHNoCTrNorm(vParamsTr, vY, vRV):
    """
    Purpose
    ----------
    Compute negative average log-likelihood for the RGARCH model using transformed parameters
    
    Parameters
    ----------
    vParamsTr :    vector, transformed model parameters
    vY :           vector, time series data
    vRV :          vector, realized volatility observations
    
    Returns
    -------
    dNLL :         double, negative average log-likelihood of the joint model
    """
    
    vParamsTrCopy = np.copy(vParamsTr)
    vParamsCopy = RGARCHLinkInverse(vParamsTrCopy, vY, bTheta=False, sDistr='normal')
    return AvgNLnLRGARCHNoCNorm(vParamsCopy, vY, vRV)

###########################################################
def RGARCHNoCNormEstim(vRGARCHParamsStart, vY, vRV, bTheta=False, sDistr='normal', bImpStat=True):
    """
    Purpose
    ----------
    Estimate RGARCH(1,1) model parameters using normal innovations and transformed parameter space
    
    Parameters
    ----------
    vRGARCHParamsStart :    vector, initial parameter values for optimization
    vY :                    vector, time series data
    vRV :                   vector, realized volatility observations
    bTheta :                boolean, estimate shape parameter if True (not used for normal case)
    sDistr :                string, innovation distribution ('normal' or 't')
    bImpStat :              boolean, impose stationarity constraint if True
    
    Returns
    -------
    vResult :               vector, estimated untransformed RGARCH(1,1) parameters
    """

    # Stationarity constraint
    def StationarityConstraintRGARCHTr(vParamsTr):
        vParamsTrCopy = np.copy(vParamsTr)
        vParamsCopy = RGARCHLinkInverse(vParamsTrCopy, vY, bTheta=False, sDistr='normal')
        return 1-1e-3 - (vParamsCopy[1] * vParamsCopy[4] + vParamsCopy[2])
    
    def StationarityConstraintRGARCH(vParams):
        return 1-1e-3 - (vParams[1] * vParams[4] + vParams[2])

    vGARCHParamsStartTr= RGARCHLink(vRGARCHParamsStart, vY, bTheta=False, sDistr='normal')

    # Use inqueality constraint to insure stationarity when desired
    lCons = [{'type': 'ineq', 'fun': StationarityConstraintRGARCHTr}]

    # Start with unconstrained optimisation
    oResult= minimize(AvgNLnLRGARCHNoCTrNorm, vGARCHParamsStartTr, args=(vY, vRV), method="BFGS", options={'disp': False})
    vResultTr = np.copy(oResult.x)
    vResult = RGARCHLinkInverse(vResultTr, vY, bTheta=False, sDistr='normal')
    
    # If stationarity condition is not satisfied, then use restricted optimisation
    if StationarityConstraintRGARCH(vResult) <0 and bImpStat:
        oResult = minimize(lambda x: AvgNLnLRGARCHNoCTrNorm(x, vY, vRV), vGARCHParamsStartTr, constraints=lCons, method='SLSQP') #L-BFGS-B
        vResultTr = np.copy(oResult.x)
        vResult = RGARCHLinkInverse(vResultTr, vY, bTheta=False, sDistr='normal')
        
    if not oResult.success:
        print("Warning: Optimization did not converge")
        print(oResult.message)
    
    if StationarityConstraintRGARCH(vResultTr) <0 and bImpStat:
        vResult[:-1] = vRGARCHParamsStart[:-1]
        
    return vResult

################################################################
##################### AR-RGARCH-t (Full QML) ###################
################################################################

###########################################################
def AvgNLnLStandStd(dNu, vY):
    """
    Purpose
    ----------
    Compute negative average log-likelihood for standardized Student-t distribution
    
    Parameters
    ----------
    dNu :     double, degrees of freedom parameter
    vY :      vector, standardized residuals
    
    Returns
    -------
    dNLL :    double, negative average log-likelihood
    """
   
    # Likelihood
    vLL = loggamma((dNu+1)/2) - loggamma(dNu/2) - 1/2* np.log((dNu-2)*np.pi) - ((dNu+1)/2) * np.log(1+(vY**2)/((dNu-2)))
    dALL= np.mean(vLL, axis= 0)

    return -dALL

###########################################################
def AvgNLnLStandTrStd(dNuTr, vY):
    """
    Purpose
    ----------
    Wrapper negative average log-likelihood for standardized Student-t distribution 
    
    Parameters
    ----------
    dNuTr :   double, transformed degrees of freedom parameter
    vY :      vector, standardized residuals
    
    Returns
    -------
    dNLL :    double, negative average log-likelihood
    """
        
    dNuCopy= np.copy(dNuTr)
    return AvgNLnLStandStd(np.exp(dNuCopy) + 2.1, vY)

###########################################################
def RGARCHNoCStdEstimQML(vRGARCHParamsStart, vY, vRV):
    """
    Purpose
    ----------
    Estimate RGARCH-t model using QML
    
    Parameters
    ----------
    vGARCHParamsStart :   vector, initial parameter values
    vY :                  vector, time series data
    
    Returns
    -------
    vRGARCHParamsStar :   vector, estimated RGARCH-t model parameters
    """
    
    vRGARCHParamsStartCopy = np.copy(vRGARCHParamsStart)
    vRGARCHNormParamsStar = RGARCHNoCNormEstim(vRGARCHParamsStartCopy[:-1], vY, vRV)
    
    # Estimated volatility series
    vSig2Star = RGARCHNoCUpdateEq(vRGARCHNormParamsStar, vY, vRV)[:-1]
    
    # Residuals
    vResiduals = (vY)/np.sqrt(vSig2Star)
    
    dNuStartTr = np.log(vRGARCHParamsStart[-1] - 2.1)
    res= minimize(AvgNLnLStandTrStd, dNuStartTr, args=(vResiduals), method="BFGS", options={'disp': False})
    dNuStarTr = np.copy(res.x)       
    dNuStar = np.exp(dNuStarTr) + 2.1
    
    vRGARCHParamsStar = np.hstack((vRGARCHNormParamsStar, dNuStar))
    
    return vRGARCHParamsStar

################################################################
######################## AR-RGARCH-t (QML) #####################
################################################################

###########################################################
def AvgNLnLRGARCHStd(vParams, vY, vRV):
    """
    Purpose
    ----------
    Compute negative average log-likelihood for the RGARCH-t model with Student-t innovations
    
    Parameters
    ----------
    vParams :   vector, model parameters including GARCH and RV equations, and shape parameter dNu
    vY :        vector, time series data
    vRV :       vector, realized volatility observations
    
    Returns
    -------
    dNLL :      double, negative average log-likelihood
    """
    
    dXi, dPhi, dTau, dKappa, dSig2u, dNu = vParams[3:] 
        
    # Volatility    
    vSig2 = RGARCHNoCUpdateEq(vParams, vY, vRV)[:-1]

    # Likelihood GARCH
    vLLGARCH= loggamma((dNu+1)/2) - loggamma(dNu/2) - 1/2* np.log((dNu-2)*np.pi) - 1/2 * np.log(vSig2) - ((dNu+1)/2) * np.log(1+((vY)**2)/((dNu-2)*vSig2))

    dALLGARCH= np.mean(vLLGARCH, axis= 0)
    
    # Realized volatility
    vRVCondMean= dXi + dPhi * vSig2 + dTau * (vY)/np.sqrt(vSig2)  + dKappa * ((vY)**2/vSig2-1)
        
    # Likelihood RV
    vLLRV= -0.5*(np.log(2*np.pi) + np.log(dSig2u) + ((vRV-vRVCondMean)**2)/dSig2u)
    dALLRV= np.mean(vLLRV, axis= 0)
        
    return -(dALLGARCH+dALLRV)

###########################################################
def AvgNLnLRGARCHTrStd(vParamsTr, vY, vRV):
    """
    Purpose
    ----------
    Wrapper negative average log-likelihood for the RGARCH-t model using transformed parameters
    
    Parameters
    ----------
    vParamsTr :    vector, transformed model parameters
    vY :           vector, time series data
    vRV :          vector, realized volatility observations
    
    Returns
    -------
    dNLL :         double, negative average log-likelihood
    """

    vParamsTrCopy = np.copy(vParamsTr)
    
    vParamsCopy =RGARCHLinkInverse(vParamsTrCopy, vY, bTheta=True, sDistr='t')
    return AvgNLnLRGARCHStd(vParamsCopy, vY, vRV)

###########################################################
def RGARCHStdEstim(vRGARCHParamsStart, vY, vRV, bImpStat=True):
    """
    Purpose
    ----------
    Estimate parameters of the RGARCH-t model using QML
    
    Parameters
    ----------
    vRGARCHParamsStart :   vector, initial values for the model parameters
    vY :                   vector, time series data
    vRV :                  vector, realized volatility observations
    bImpStat :             boolean, enforce stationarity constraint if True
    
    Returns
    -------
    vResult :              vector, estimated untransformed model parameters
    """

    # Stationarity constraint
    def StationarityConstraintRGARCHTr(vParamsTr):
         vParamsTrCopy = np.copy(vParamsTr)
         vParamsCopy = RGARCHLinkInverse(vParamsTrCopy, vY, bTheta=True, sDistr='t')
         return 1-1e-3 - (vParamsCopy[1] * vParamsCopy[4] + vParamsCopy[2])
     
    def StationarityConstraintRGARCH(vParams):
         return 1-1e-3 - (vParams[1] * vParams[4] + vParams[2])

    vGARCHParamsStartTr= RGARCHLink(vRGARCHParamsStart, vY, bTheta=True, sDistr='t')
    # Use inqueality constraint to ensure stationarity when desired
    lCons = [{'type': 'ineq', 'fun': StationarityConstraintRGARCHTr}]

    # Start with unconstrained optimisation
    oResult= minimize(AvgNLnLRGARCHTrStd, vGARCHParamsStartTr, args=(vY, vRV), method="BFGS", options={'disp': False})
    vResultTr = np.copy(oResult.x)
    vResult = RGARCHLinkInverse(vResultTr, vY, bTheta=True, sDistr='t')
    
    # If stationarity condition is not satisfied, then use restricted optimisation
    if StationarityConstraintRGARCH(vResult) <0 and bImpStat:
        oResult = minimize(lambda x: AvgNLnLRGARCHTrStd(x, vY, vRV), vGARCHParamsStartTr, constraints=lCons, method='SLSQP') #L-BFGS-B
        vResultTr = np.copy(oResult.x)
        vResult = RGARCHLinkInverse(vResultTr, vY, bTheta=True, sDistr='t')
        
    if not oResult.success:
        print("Warning: Optimization did not converge")
        print(oResult.message)
    
    if StationarityConstraintRGARCH(vResultTr) <0 and bImpStat:
        vResult[:-1] = vRGARCHParamsStart[:-1]

    return vResult

###########################################################
###########################################################
###################### RGARCH (ML) ########################
###########################################################
###########################################################

###########################################################
def ARRGARCHLink(vParams, vY, iP, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Compute transformed parameters for the AR(p)-RGARCH model
    
    Parameters
    ----------
    vParams :       vector, untransformed model parameters
    vY :            vector, time series data
    iP :            integer, AR order
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    sDistr :        string, innovation distribution ('normal' or 't')
    
    Returns
    -------
    vParamsTrOut :  vector, transformed model parameters
    """

    vParamsCopy = np.copy(vParams)
    vParamsTrOut = np.zeros(len(vParamsCopy))
    # dPhi0
    vParamsTrOut[0] = vParamsCopy[0] 
    # vPhi                                       
    vParamsTrOut[1:iP+1] = np.arctanh(vParamsCopy[1:iP+1])
    # dOmega                                  
    vParamsTrOut[iP+1:iP+2] = np.log(vParamsCopy[iP+1:iP+2])  
    # dAlpha                             
    vParamsTrOut[iP+2:iP+3] = np.log(vParamsCopy[iP+2:iP+3])                               
    # dBeta
    vParamsTrOut[iP+3:iP+4] = logit(vParamsCopy[iP+3:iP+4])   #helpful for optimisation
    #dXi
    vParamsTrOut[iP+4] = np.log(vParamsCopy[iP+4])
    #dPhi
    vParamsTrOut[iP+5] = np.log(vParamsCopy[iP+5])
    #dTau
    vParamsTrOut[iP+6] = vParamsCopy[iP+6]
    #dKappa
    vParamsTrOut[iP+7] = np.log(vParamsCopy[iP+7])
    #dSig2u
    vParamsTrOut[iP+8] = np.log(vParamsCopy[iP+8])
    
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsTrOut[-1] =  logit((vParamsCopy[-1]-2.1)/dNuMax) # robustification
    
    return vParamsTrOut
    
###########################################################
def ARRGARCHLinkInverse(vParamsTr, vY, iP, bTheta=False, sDistr='normal'):
    """
    Purpose
    ----------
    Compute untransformed model parameters for the AR(p)-RGARCH model from transformed (freely floating) parameters
    
    Parameters
    ----------
    vParamsTr :     vector, transformed model parameters
    vY :            vector, time series data
    iP :            integer, AR order
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    sDistr :        string, innovation distribution ('normal' or 't')
    
    Returns
    -------
    vParamsOut :    vector, untransformed model parameters
    """

    vParamsTrCopy = np.copy(vParamsTr)
    vParamsOut = np.zeros(len(vParamsTrCopy))
    # dPhi0
    vParamsOut[0] = vParamsTrCopy[0]# ScaledTanhInv(vParamsTrCopy[0], -10*np.abs(vY.mean()),10*np.abs(vY.mean()))
    # vPhi                                       
    vParamsOut[1:iP+1] = np.tanh(vParamsTrCopy[1:iP+1])
    # dOmega                                  
    vParamsOut[iP+1:iP+2] = np.exp(vParamsTrCopy[iP+1:iP+2])  
    # dAlpha                             
    vParamsOut[iP+2:iP+3] = np.exp(vParamsTrCopy[iP+2:iP+3])                               
    # dBeta
    vParamsOut[iP+3:iP+4] = expit(vParamsTrCopy[iP+3:iP+4])   # dBeta
    #dXi
    vParamsOut[iP+4] = np.exp(vParamsTrCopy[iP+4])
    #dPhi
    vParamsOut[iP+5] = np.exp(vParamsTrCopy[iP+5])
    #dTau
    vParamsOut[iP+6] = vParamsTrCopy[iP+6]
    #dKappa
    vParamsOut[iP+7] = np.exp(vParamsTrCopy[iP+7])
    #dSig2u
    vParamsOut[iP+8] = np.exp(vParamsTrCopy[iP+8])
    
    # vTheta
    dNuMax = 1e2 # upperbound degrees of freedom parameter [to avoid numerical issues]
    if bTheta and sDistr == 't':
        vParamsOut[-1] = dNuMax*expit(vParamsTrCopy[-1]) +2.1
    
    return vParamsOut

###########################################################
def ARRGARCHAvgNLL(vParams, vY, vRV, iP, bTheta=False, vThetaFixed=None, sDistr='normal'):
    """
    Purpose
    ----------
    Compute average negative log-likelihood for the AR(p)-RGARCH model with observed realized volatility
    
    Parameters
    ----------
    vParams :       vector, model parameters including AR, GARCH, and measurement equations
    vY :            vector, time series data
    vRV :           vector, realized volatility measures
    iP :            integer, AR order
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :   double, fixed shape parameter if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')
    
    Returns
    -------
    dNLL :          double, negative average log-likelihood
    """

    dPhi0 = vParams[0]
    vPhi = vParams[1:iP + 1]

    dOmega, dAlpha, dBeta = vParams[iP+1:iP+4]
    
    vEps = np.zeros_like(vY)
    vEps[:iP] = vY[:iP]
    for t in range(iP, len(vY)):
        vEps[t] = vY[t] - dPhi0 - np.dot(vPhi, vY[t - iP:t][::-1])

    vSig2 = np.zeros_like(vEps)
    vSig2[0] = vY.var() 

    dXi, dPhi, dTau, dKappa, dSig2u = vParams[iP+4:iP+9] 
    
    if sDistr == 't' and bTheta:
        dNu = vParams[-1]
    # Volatility    
    vSig2 = RGARCHNoCUpdateEq(vParams[iP+1:], vEps, vRV)[:-1]
    
    # Likelihood GARCH
    if sDistr == 'normal':
        vLLGARCH= -0.5*(np.log(2*np.pi) + np.log(vSig2) + ((vEps)**2)/vSig2)
    elif sDistr == 't' and bTheta:
        vLLGARCH= loggamma((dNu+1)/2) - loggamma(dNu/2) - 1/2* np.log((dNu-2)*np.pi) - 1/2 * np.log(vSig2) - ((dNu+1)/2) * np.log(1+((vEps)**2)/((dNu-2)*vSig2))

    dALLGARCH= np.mean(vLLGARCH, axis= 0)
    
    # Realized volatility
    vRVCondMean= dXi + dPhi * vSig2 + dTau * (vY)/np.sqrt(vSig2)  + dKappa * ((vY)**2/vSig2-1)
  
    # Likelihood RV
    vLLRV= -0.5*(np.log(2*np.pi) + np.log(dSig2u) + ((vRV-vRVCondMean)**2)/dSig2u)
    dALLRV= np.mean(vLLRV, axis= 0)
        
    return -(dALLGARCH+dALLRV)

###########################################################
def ARRGARCHAvgNLLTr(vParamsTr, vY, vRV, iP, bTheta=False, vThetaFixed=None, sDistr='normal'):
    """
    Purpose
    ----------
    Compute average negative log-likelihood for the AR(p)-RGARCH model using transformed parameters
    
    Parameters
    ----------
    vParamsTr :     vector, transformed model parameters
    vY :            vector, time series data
    vRV :           vector, realized volatility measures
    iP :            integer, AR order
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :   double, fixed shape parameter if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')
    
    Returns
    -------
    dNLL :          double, negative average log-likelihood
    """

    vParamsTrCopy = np.copy(vParamsTr) 
    vParamsCopy = ARRGARCHLinkInverse(vParamsTrCopy, vY, iP, bTheta=bTheta, sDistr=sDistr)
    return ARRGARCHAvgNLL(vParamsCopy, vY, vRV, iP, bTheta, vThetaFixed, sDistr)

###########################################################
def ARRGARCHEstim(vY, vRV, iP, bTheta=False, vThetaFixed=None, sDistr='normal', bImpStat=True):
    """
    Purpose
    ----------
    Estimate parameters of the AR(p)-RGARCH model using transformed parameters and restricted optimization
    
    Parameters
    ----------
    vY :            vector, time series data
    vRV :           vector, realized volatility measures
    iP :            integer, AR order
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :   vector, fixed shape parameters if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')
    bImpStat :      boolean, enforce stationarity through inequality constraint if True
    
    Returns
    -------
    dPhi0 :         double, constant term of the AR(p) model
    vPhi :          vector, AR coefficients
    vRGARCHParams : vector, RGARCH model parameters excluding AR terms
    vTheta :        vector, estimated shape parameters 
    """

    # Starting values
    dOmegaStart= vY.var()/20
    vAlphaStart= np.array([0.1]) 
    vBetaStart= np.array([0.85])
    dNuStart = 6
    vGARCHNormParamsStart= np.hstack((dOmegaStart, vAlphaStart, vBetaStart))
    dXiStart = 0.1
    dPhiStart = 0.2
    dTauStart = 0.1
    dKappaStart = 0.3
    dSig2uStart = 1
    vRGARCHParamsStartRV = np.hstack((dXiStart, dPhiStart, dTauStart, dKappaStart, dSig2uStart))
    vRGARCHNormParamsStart = np.hstack((vGARCHNormParamsStart, vRGARCHParamsStartRV))
    vRGARCHStdParamsStart = np.hstack((vGARCHNormParamsStart, vRGARCHParamsStartRV, dNuStart))
    dInitialPhi0 = vY.mean()
    if iP>0: vInitialARCoefs = acf(vY, nlags=iP, fft=False)[1:]
     
    if bTheta and sDistr == 't':
        if iP>0:
            vInitialParams = np.concatenate([[dInitialPhi0], vInitialARCoefs, vRGARCHStdParamsStart])
        else:
            vInitialParams = np.concatenate([[dInitialPhi0], vRGARCHStdParamsStart])
    else:
        if iP>0:
            vInitialParams = np.concatenate([[dInitialPhi0], vInitialARCoefs, vRGARCHNormParamsStart])
        else:
            vInitialParams = np.concatenate([[dInitialPhi0], vRGARCHNormParamsStart])

          
    # Stationarity constraint
    def StationarityConstraintARRGARCHTr(vParamsTr):
         #alpha * phi + beta
         vParamsTrCopy = np.copy(vParamsTr)
         vParamsCopy = ARRGARCHLinkInverse(vParamsTrCopy, vY, iP, bTheta=False, sDistr='normal')
         return 1-1e-3 - (vParamsCopy[iP+2] * vParamsCopy[4+iP+1] + vParamsCopy[2+iP+1])
     
    def StationarityConstraintARRGARCH(vParams):
         return 1-1e-3 - (vParams[iP+2] * vParams[4+iP+1] + vParams[2+iP+1])
    
    # Active constraint
    lCons = [{'type': 'ineq', 'fun': StationarityConstraintARRGARCHTr}]
    
    # Start with unconstrained optimisation
    vInitialParamsTr = ARRGARCHLink(vInitialParams, vY, iP, bTheta=bTheta, sDistr=sDistr)
    ARRGARCHAvgNLLTr(vInitialParamsTr, vY, vRV, iP, bTheta=False, vThetaFixed=None, sDistr='normal')
    oResult = minimize(lambda x: ARRGARCHAvgNLLTr(x, vY, vRV, iP, bTheta=bTheta, vThetaFixed=vThetaFixed, sDistr=sDistr), vInitialParamsTr, method='L-BFGS-B') #L-BFGS-B
    vResultTr = np.copy(oResult.x)
    vResult = ARRGARCHLinkInverse(vResultTr, vY, iP, bTheta=bTheta, sDistr=sDistr)
        
    # If stationarity condition is not satisfied, then use restricted optimisation
    if StationarityConstraintARRGARCH(vResult) <0 and bImpStat:
        oResult = minimize(lambda x: ARRGARCHAvgNLLTr(x, vY, vRV, iP, bTheta=bTheta, vThetaFixed=vThetaFixed, sDistr=sDistr), vInitialParamsTr, constraints=lCons, method='SLSQP', options={'ftol': 1e-9, 'disp': False}) #L-BFGS-B
        vResultTr = np.copy(oResult.x)
        vResult = ARRGARCHLinkInverse(vResultTr, vY, iP, bTheta=bTheta, sDistr=sDistr)
        
    if not oResult.success:
        print("Warning: Optimization did not converge")
        print(oResult.message)
        
    dPhi0 = vResult[0]
    vPhi = vResult[1:iP + 1]
    
    if bTheta and sDistr == 't':
        vRGARCHParams = vResult[iP+1:-1]
        vTheta = np.array([vResult[-1]])
    else:
        vRGARCHParams = vResult[iP+1:]
        vTheta = None

    return dPhi0, vPhi, vRGARCHParams, vTheta

###########################################################
def ARRGARCHForecast(vY, dPhi0, vPhi, vRGARCHParams, vRV, bTheta=False, vTheta=None, iHorizon=10, bResiduals=False):
    """
    Purpose
    ----------
    Compute multi-step forecasts for the AR(p)-RGARCH model, including conditional means, variances, and distribution parameters
    
    Parameters
    ----------
    vY :            vector, time series data
    dPhi0 :         double, constant term in the AR(p) model
    vPhi :          vector, AR coefficients
    vRGARCHParams : vector, RGARCH model parameters 
    vRV :           vector, realized volatility measures
    bTheta :        boolean, include shape parameter vector vTheta in output if True
    vTheta :        vector, static shape parameters values 
    iHorizon :      integer, forecast horizon
    bResiduals :    boolean, return standardized residuals if True
    
    Returns
    -------
    mForecast :     array, contains forecasted means, variances, and distribution parameters
    vResiduals :    vector, standardized residuals (only if bResiduals is True)
    """

    iT = len(vY)
    iP = len(vPhi)

    dOmega, dAlpha, dBeta = vRGARCHParams[:3]
    mForecastFull = np.ones((iT + iHorizon, 3))
    mForecastFull[:iT, 0] = vY
    mForecastFull[:iT, 1] = np.nan
    mForecast = np.ones((iHorizon, 3)) * np.nan
    # Forecast mean and variance
    for h in range(1, iHorizon + 1):
        mForecastFull[iT - 1 + h, 0] = dPhi0 + np.dot(vPhi, mForecastFull[iT - 1 + h - iP:iT - 1 + h, 0][::-1])

    # Filter volatility 
    vEps = np.zeros_like(vY)
    vEps[:iP] = vY[:iP]
    for t in range(iP, len(vY)):
        vEps[t] = vY[t] - dPhi0 - np.dot(vPhi, vY[t - iP:t][::-1])

    mForecast[:,0] = mForecastFull[iT:,0]
    
    if bResiduals:
        mForecast[:,1], vResiduals = RGARCHNoC11Pred(vRGARCHParams, vEps, np.arange(1,iHorizon+1), vRV, bResiduals=True)
    else:
        mForecast[:,1] = RGARCHNoC11Pred(vRGARCHParams, vEps, np.arange(1,iHorizon+1), vRV)
    
    # Store static distribution parameters
    if bTheta:
        for i in range(len(vTheta)):
            mForecast[:, 2 + i] = vTheta[i]

    if bResiduals:
        return mForecast, vResiduals
    else:
        return mForecast

###########################################################
def RGARCHEstimAndForecast(vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, vH=np.array([1,5]), bQML=True, bQMLFull=True, vRV=None, bImpStat=True, bResiduals=False):
    """
    Purpose
    ----------
    Compute h-step ahead forecasts for the AR(p)-RGARCH model using QML or full likelihood estimation
    
    Parameters
    ----------
    vY :            vector, time series data
    iP :            integer, AR order
    bGamma :        boolean, placeholder
    bRobust :       boolean, placeholder
    bTheta :        boolean, estimate shape parameter vector vTheta if True
    vThetaFixed :   double or None, fixed shape parameter if bTheta is False
    sDistr :        string, innovation distribution ('normal' or 't')
    bPackage :      boolean, placeholder
    vH :            vector, forecast horizons (e.g., [1, 5])
    bQML :          boolean, use QML estimation if True
    bQMLFull :      boolean, use full QML (including Student-t) if True
    vRV :           vector, realized volatility measures
    bImpStat :      boolean, enforce stationarity constraint if True
    bResiduals :    boolean, return standardized residuals if True
    
    Returns
    -------
    mForecast :     array, contains forecasted means, variances, and (if applicable) shape parameters
    vResiduals :    vector, standardized residuals (only if bResiduals is True)
    """
    
    iHorizon = int(vH.max())
  
    # Starting values
    dOmegaStart= vY.var()/20
    vAlphaStart= np.array([0.1])
    vBetaStart= np.array([0.85])
    dNuStart = 6
    vGARCHNormParamsStart= np.hstack((dOmegaStart, vAlphaStart, vBetaStart))
    dXiStart = 0.1
    dPhiStart = 0.2
    dTauStart = 0.1
    dKappaStart = 0.3
    dSig2uStart = 1
    vRGARCHParamsStartRV = np.hstack((dXiStart, dPhiStart, dTauStart, dKappaStart, dSig2uStart))
    vRGARCHNormParamsStart = np.hstack((vGARCHNormParamsStart, vRGARCHParamsStartRV))
    vRGARCHStdParamsStart = np.hstack((vGARCHNormParamsStart, vRGARCHParamsStartRV, dNuStart))
    
    ## Full QML and QML [three and two stage estimation for Student-t]
    if bQML:
        vPhihat, vYRes, vMuHat = AROLSEstim(vY, iP, iHorizon)
        mForecastAll = np.ones((iHorizon,3)) * np.nan
        mForecastAll[:,0] = vMuHat
        vRVRes = vRV[iP:] # vRV with same length as vYRes
        
        # Estimation
        if sDistr == 'normal':
            vParamsHat = RGARCHNoCNormEstim(vRGARCHNormParamsStart, vYRes, vRVRes, bImpStat=bImpStat)
        elif 't' and bTheta:
            if bQMLFull:
                vParamsHat = RGARCHNoCStdEstimQML(vRGARCHStdParamsStart, vYRes, vRVRes)
            else:
                vParamsHat = RGARCHStdEstim(vRGARCHStdParamsStart, vYRes, vRVRes, bImpStat=bImpStat)
            
        # Forecast of mu is muhat for all h [constant mean model]
        mForecastAll[:,0] = vMuHat
        
        # Forecast of volatility 
        if sDistr == 'normal':
            mForecastAll[:,1], vResiduals = RGARCHNoC11Pred(vParamsHat, vYRes, np.arange(1,iHorizon+1), vRV, bResiduals=True) 
        elif sDistr == 't' and bTheta:
            mForecastAll[:,1], vResiduals = RGARCHNoC11Pred(vParamsHat[:-1], vYRes, np.arange(1,iHorizon+1), vRV, bResiduals=True) 
        
        # Forecast of nu is nuhat if Student-t is selected distribution
        if sDistr == 't': mForecastAll[:,2] = vParamsHat[-1] * np.ones(iHorizon) 
        
    else:
        dPhi0_est, vPhi_est, vRGARCHParams_est, vTheta_est  =  ARRGARCHEstim(vY, vRV, iP, bTheta=bTheta, vThetaFixed=vThetaFixed, sDistr=sDistr, bImpStat=bImpStat)
        mForecastAll, vResiduals = ARRGARCHForecast(vY, dPhi0_est, vPhi_est, vRGARCHParams_est, vRV, bTheta=bTheta, vTheta=vTheta_est, iHorizon=iHorizon, bResiduals=True)
    
    mForecast = np.ones((len(vH), mForecastAll.shape[1])) * np.nan
    iCount = 0
    for h in vH:
        mForecast[iCount,:] = mForecastAll[h-1,:]
        iCount += 1

    if bResiduals:
        return mForecast, vResiduals
    else:
        return mForecast