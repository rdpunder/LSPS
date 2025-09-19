#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: File containing multivariate scoring rule functions 
"""

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration
import types # check whether object is function
from scipy.integrate import nquad

######################################################################
####################### FUNDAMENTAL INGREDIENTS ######################
######################################################################

###########################################################  
def AlphaNormsF(dictDistr, dictW, vAlpha):
    """
    Purpose
    ----------
    Calculate alpha norm density f
    Note: Regular scoring rules can be calculated from weighted scoring rules, by setting the weights equal to one.
    We program regular scoring rules also standalone to validate the weighted rules 

    Parameters
    ----------
    dictDistr :     dictionary, selected distribution
                        randF :  distribution F, stats object
                        sRandF : name distribution
    dictW :         dictionary, weight function: 
                        fW : selected weighted function
                        vR : vector, threshold grid
                        vParamsW : parameter vector of weight function                 
    vAlpha :        vector, alpha grid for which norms should be calculated
    
    Returns
    ----------
    dictionary [key: alpha] with vector of length vR of alpha norms per alpha 
    """    
    
    vR = np.array(dictW['vR'])
    if type(dictDistr['randDistr']) == dict:
        return {str(dAlpha): np.array([(integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x))**dAlpha, -np.inf, np.inf)[0])**(1/dAlpha)]) * np.ones(np.array(vR).size) for dAlpha in vAlpha[vAlpha > 0]} 
    else:
        return {str(dAlpha): np.array([(integrate.quad(lambda x:  (dictDistr['randDistr'].pdf(x))**dAlpha, -np.inf, np.inf)[0])**(1/dAlpha)]) * np.ones(np.array(vR).size) for dAlpha in vAlpha[vAlpha > 0]} 

###########################################################  
def AlphaNormsFw(dictDistr, dictW, vAlpha, vIntR=False):
    """
    Purpose
    ----------
    Calculate alpha norm weighted kernel f_w

    Parameters
    ----------
    dictDistr :     dictionary, selected distribution
                        randF :  distribution F, stats object
                        sRandF : name distribution
    dictW :         dictionary, weight function: 
                        fW : selected weight function
                        vR : vector, threshold grid
                        vParamsW : parameter vector of weight function           
    vAlpha :        vector, alpha grid for which norms should be calculated
    vIntR :         vector, optional, when used: skip all other threshold values 
    
    Returns
    ----------
    dictionary [key: alpha] with vector of length vR of alpha norms per alpha 
    """    
    
    vR = np.array(dictW['vR']) # matrix in bivariate case
    dictAlphaNormsFw = {}
    if type(dictDistr['randDistr']) == dict:
        for dAlpha in vAlpha[vAlpha > 0]:
            vFwNorm = np.zeros(vR.shape)
            if dictW['fW'].__name__ == 'fWIndicatorL':
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, -np.inf, vR[vIntR[i]],epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)   
            elif dictW['fW'].__name__ == 'fWIndicatorR':
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, vR[vIntR[i]], np.inf, epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)   
            elif dictW['fW'].__name__ == 'fWIndicatorC':
                for i in range(vIntR.size): # has been checked for large intervals relative to distribution 
                    vFwNorm[vIntR[i]] = (integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, dictW['vParamsW']-vR[vIntR[i]], dictW['vParamsW']+vR[vIntR[i]],epsabs=1e-15, epsrel=1e-15, limit=100)[0])**(1/dAlpha) 
            elif dictW['fW'].__name__ == 'fWIndicatorLBivProd':
                vFwNorm = np.zeros(vR.shape[1])
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.dblquad(lambda x1, x2:  (dictDistr['randDistr']['pdf'](x1, x2) * dictW['fW'](x1, x2, dictW['vParamsW'], vR[:,vIntR[i]]))**dAlpha, -np.inf,  vR[0,vIntR[i]], -np.inf, vR[1,vIntR[i]], epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)
            elif dictW['fW'].__name__ == 'fWIndicatorLBivSum':
                vFwNorm = np.zeros(vR.shape[1])
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.dblquad(lambda x1, x2:  (dictDistr['randDistr']['pdf'](x1, x2) * dictW['fW'](x1, x2, dictW['vParamsW'], vR[:,vIntR[i]]))**dAlpha, -np.inf,  vR[0,vIntR[i]], lambda x2: -np.inf, lambda x2: vR[0,vIntR[i]]-x2, epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)
            elif dictW['fW'].__name__ == 'fWLogisticLBivProd':
                vFwNorm = np.zeros(vR.shape[1])
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.dblquad(lambda x1, x2:  (dictDistr['randDistr']['pdf'](x1, x2) * dictW['fW'](x1, x2, dictW['vParamsW'], vR[:,vIntR[i]]))**dAlpha, -np.inf,  np.inf, -np.inf, np.inf, epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)        
            else:
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, -np.inf, np.inf,epsabs=1e-15, epsrel=1e-15, limit=100)[0])**(1/dAlpha) 
            dictAlphaNormsFw[str(dAlpha)] = vFwNorm
    else: 
        for dAlpha in vAlpha[vAlpha > 0]:
            vFwNorm = np.zeros(vR.shape)
            if dictW['fW'].__name__ == 'fWIndicatorL':
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr'].pdf(x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, -np.inf, vR[vIntR[i]],epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)
            elif dictW['fW'].__name__ == 'fWIndicatorR':
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr'].pdf(x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, vR[vIntR[i]], np.inf, epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)
            elif dictW['fW'].__name__ == 'fWIndicatorC':
                for i in range(vIntR.size): # has been checked for large intervals relative to distribution 
                    vFwNorm[vIntR[i]] = (integrate.quad(lambda x:  (dictDistr['randDistr'].pdf(x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, dictW['vParamsW']-vR[vIntR[i]], dictW['vParamsW']+vR[vIntR[i]],epsabs=1e-15, epsrel=1e-15, limit=100)[0])**(1/dAlpha) 
                dictAlphaNormsFw[str(dAlpha)] = vFwNorm
            elif dictW['fW'].__name__ == 'fWIndicatorLBivProd':
                vFwNorm = np.zeros(vR.shape[1])
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.dblquad(lambda x1, x2:  (dictDistr['randDistr'].pdf([x1, x2]) * dictW['fW'](np.array([x1, x2]), dictW['vParamsW'], vR[:,vIntR[i]]))**dAlpha, -np.inf,  vR[0,vIntR[i]], -np.inf, vR[1,vIntR[i]], epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)    
            elif dictW['fW'].__name__ == 'fWLogisticLBivProd': 
                vFwNorm = np.zeros(vR.shape[1])
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.dblquad(lambda x1, x2:  (dictDistr['randDistr'].pdf([x1, x2]) * dictW['fW'](np.array([x1, x2]), dictW['vParamsW'], vR[:,vIntR[i]]))**dAlpha, -np.inf,  np.inf, -np.inf, np.inf, epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)            
            elif dictW['fW'].__name__ == 'fWIndicatorLBivSum':
                vFwNorm = np.zeros(vR.shape[1])
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.dblquad(lambda x1, x2:  (dictDistr['randDistr'].pdf([x1, x2]) * dictW['fW'](np.array([x1, x2]), dictW['vParamsW'], vR[:,vIntR[i]]))**dAlpha, -np.inf,  np.inf, lambda x2: -np.inf, lambda x2: vR[0,vIntR[i]]-x2, epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)            
            else:
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr'].pdf(x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, -np.inf, np.inf,epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha) 
            dictAlphaNormsFw[str(dAlpha)] = vFwNorm
            
    return dictAlphaNormsFw

###########################################################  
def DistrBar(dictDistr, dictW, vIntR=False):
    """
    Purpose
    ----------
    Calculate `right-tail' probability

    Parameters
    ----------
    dictDistr :     dictionary, selected distribution
                        randF :  distribution F, stats object
                        sRandF : name distribution
    dictW :         dictionary, weight function: 
                        fW : selected weighted function
                        vR : vector, threshold grid
                        vParamsW : parameter vector of weight function          
    vIntR :         vector, optional, when used: skip all other threshold values 
    
    Returns
    ----------
    vector,         length vR with \bar F_w for each threshold r
    """    
    
    vR = np.array(dictW['vR']) # matrix if bivariate
    
    # The integral simplifies if the weight function is an indicator function
    # introduce booleans for these cases
    iLeftIndicator = int(dictW['fW'].__name__ == 'fWIndicatorL')
    iCentreIndicator = int(dictW['fW'].__name__ == 'fWIndicatorC')
    iRightIndicator = int(dictW['fW'].__name__ == 'fWIndicatorR')
    
    vFwBar = np.zeros(vR.shape)
    if dictW['fW'].__name__ == 'fWIndicatorL' and type(dictDistr['randDistr']) != dict:
        for i in range(vIntR.size):
            vFwBar[vIntR[i]] = 1- dictDistr['randDistr'].cdf(vR[vIntR[i]])
    elif dictW['fW'].__name__ == 'fWIndicatorR' and type(dictDistr['randDistr']) != dict:
        for i in range(vIntR.size):
            vFwBar[vIntR[i]] = dictDistr['randDistr'].cdf(vR[vIntR[i]])    
    elif dictW['fW'].__name__ == 'fWIndicatorC':
        dC = dictW['vParamsW']
        if type(dictDistr['randDistr']) == dict:
            for i in range(vIntR.size):
                vFwBar[vIntR[i]] = dictDistr['randDistr']['cdf'](np.array([dC-vR[vIntR[i]]]))   +  1- dictDistr['randDistr']['cdf'](np.array([dC+vR[vIntR[i]]]))    
        else:    
            for i in range(vIntR.size):
                vFwBar[vIntR[i]] = dictDistr['randDistr'].cdf(dC-vR[vIntR[i]])   +  1- dictDistr['randDistr'].cdf(dC+vR[vIntR[i]])    
    elif dictW['fW'].__name__ == 'fWIndicatorTails':
        dC = dictW['vParamsW']
        if type(dictDistr['randDistr']) == dict:
            for i in range(vIntR.size):
                vFwBar[vIntR[i]] = float(dictDistr['randDistr']['cdf'](np.array([dC+vR[vIntR[i]]]))-dictDistr['randDistr']['cdf'](np.array([dC-vR[vIntR[i]]])))
        else:
            for i in range(vIntR.size):
                vFwBar[vIntR[i]] = float(dictDistr['randDistr'].cdf(np.array([dC+vR[vIntR[i]]]))-dictDistr['randDistr'].cdf(np.array([dC-vR[vIntR[i]]])))
    elif dictW['fW'].__name__ == 'fWIndicatorC':
        if type(dictDistr['randDistr']) == dict:
            for i in range(vIntR.size):
                vFwBar[vIntR[i]] = 1-(float(dictDistr['randDistr']['cdf'](np.array([dC+vR[vIntR[i]]]))-dictDistr['randDistr']['cdf'](np.array([dC-vR[vIntR[i]]]))))
        else:
            for i in range(vIntR.size):
                vFwBar[vIntR[i]] = 1-(float(dictDistr['randDistr'].cdf(np.array([dC+vR[vIntR[i]]]))-dictDistr['randDistr'].cdf(np.array([dC-vR[vIntR[i]]]))))
    elif  (dictW['fW'].__name__ == 'fWIndicatorLBivProd') and type(dictDistr['randDistr']) != dict:
        vFwBar = np.zeros(vR.shape[1])
        for i in range(vIntR.size):
            vFwBar[vIntR[i]] = 1- integrate.dblquad(lambda x1, x2:  dictDistr['randDistr'].pdf([x1, x2]) * dictW['fW'](np.array([x1, x2]), dictW['vParamsW'], vR[:,vIntR[i]]), -np.inf,  vR[0,vIntR[i]], -np.inf, vR[1,vIntR[i]], epsabs=1e-15, epsrel=1e-15)[0]          
    elif dictW['fW'].__name__ == 'fWLogisticLBivProd' and type(dictDistr['randDistr']) != dict:
        vFwBar = np.zeros(vR.shape[1])
        for i in range(vIntR.size):
            vFwBar[vIntR[i]] = 1- integrate.dblquad(lambda x1, x2:  dictDistr['randDistr'].pdf([x1, x2]) * dictW['fW'](np.array([x1, x2]), dictW['vParamsW'], vR[:,vIntR[i]]), -np.inf,  np.inf, -np.inf, np.inf, epsabs=1e-15, epsrel=1e-15)[0]    
    elif dictW['fW'].__name__ == 'fWIndicatorLBivSum' and type(dictDistr['randDistr']) != dict:
         vFwBar = np.zeros(vR.shape[1])
         for i in range(vIntR.size):
             vFwBar[vIntR[i]] = 1- integrate.dblquad(lambda x1, x2:  dictDistr['randDistr'].pdf([x1, x2]) * dictW['fW'](np.array([x1, x2]), dictW['vParamsW'], vR[:,vIntR[i]]),  -np.inf,  np.inf, lambda x2: -np.inf, lambda x2: vR[0,vIntR[i]]-x2, epsabs=1e-15, epsrel=1e-15)[0]     
    return vFwBar

######################################################################
###################### LOGARITHMIC SCORING RULE ######################
######################################################################

###########################################################  
def LogS(dictDistr, mY, iRidx=0, dictW=False, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Regular Logarithmic scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated, redundant for LogS                    
    dictW :         dictionary,  weight function: 
                        fW : selected weighted function
                        vR : vector, threshold grid
                        vParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    iT x iRep matrix with calculated scores  
    """
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    mF[mF==0] = 1e-100 # avoid numerical zeros
    return np.log(mF)

###########################################################  
def LogSSharp(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional Logarithmic scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        vParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Return
    ----------
    matrix, shape mY, with calculated scores
    """
    # threshold grid
    vR = np.array(dictW['vR'])
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    mF[mF==0] = 1e-100 # avoid numerical zeros
    dFwBarForLog = np.min((1-10e-10, dFwBar))
   
    return mW * (np.log(mF) - np.log(1-dFwBarForLog)) 

###########################################################  
def LogSFlat(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Censored Logarithmic scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated                  
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores 
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
        
    dFwBar = np.max((dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx],1e-100)) # \bar F_w for single r
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
        
    mF[mF==0] = 1e-100 # avoid numerical zeros
   
    return mW * np.log(mF) + (1-mW) * np.log(dFwBar)

###########################################################  
def LogSSharpslog(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional Logarithmic scoring rule + slog correction HK

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated            
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        vParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Return
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    mF[mF==0] = 1e-100 # avoid numerical zeros
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    mSwSharp = mW * (np.log(mF) - np.log(1-dFwBarForLog)) 
    mSlogCorr = mW * np.log(1-dFwBarForLog) + (1-mW) * np.log(dFwBarForLog)

    return mSwSharp + mSlogCorr

###########################################################  
def LogSSharpsbar(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional Logarithmic scoring rule + sbar correction HK

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        vParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Return
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    mF[mF==0] = 1e-100 # avoid numerical zeros
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    mSwSharp = mW * (np.log(mF) - np.log(1-dFwBarForLog)) 
    mSbarCorr = mW * (np.log(1-dFwBarForLog) + 1) - (1-dFwBarForLog)

    return mSwSharp + mSbarCorr
    
######################################################################
#################### PSEUDOSPHERICAL SCORING RULE ####################
######################################################################

###########################################################  
def PsSphS(dictDistr, mY, iRidx, dictW=False, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Regular PseudoSpherical scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated (placeholder)           
    dictW :         dictionary,  weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    dFNorm = dictPreCalc['AlphaNormsF'][dictDistr['sDistr']][str(vParamsS)][iRidx] # alpha norm f
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
        
    return (mF/dFNorm)**(vParamsS-1)

###########################################################  
def PsSphSSharp(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional PseudoSpherical scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated               
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx]
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = np.array(mW * (mW * mF/dFwNorm)**(vParamsS-1))
    mSwSharp[mW==0] = 0 #potential numerical nan mapped to 0 [enforcing 0*nan=0]
    
    return mSwSharp 

###########################################################  
def PsSphSFlat(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Censored PseudoSpherical scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated               
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Return
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx] # alpha norm f_w for single r
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
        
    return (mW * (mW * mF)**(vParamsS-1) + (1-mW) * dFwBar**(vParamsS-1)) / (dFwNorm**vParamsS + dFwBar**vParamsS )**((vParamsS-1)/vParamsS)

###########################################################  
def PsSphSSharpslog(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional PseudoSpherical scoring rule + slog correction HK

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated                  
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx]
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = np.array(mW * (mW * mF/dFwNorm)**(vParamsS-1))   
    mSwSharp[mW==0] = 0 
    
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    mSlogCorr = mW * np.log(1-dFwBarForLog) + (1-mW) * np.log(dFwBarForLog)
    
    return mSwSharp + mSlogCorr

###########################################################  
def PsSphSSharpsbar(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional PseudoSpherical scoring rule + sbar correction HK

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated                
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx]
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = np.array(mW * (mW * mF/dFwNorm)**(vParamsS-1)) 
    mSwSharp[mW==0] = 0 
    
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    mSbarCorr = mW * (np.log(1-dFwBarForLog) + 1) - (1-dFwBarForLog)

    return mSwSharp + mSbarCorr

######################################################################
######################### POWER SCORING RULE #########################
######################################################################

###########################################################  
def PowS(dictDistr, mY, iRidx, dictW=False, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Regular PseudoSpherical scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated (placeholder)              
    dictW :         dictionary,  weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Return
    ----------
    matrix, shape mY, with calculated scores
    """
    
    dFNorm = dictPreCalc['AlphaNormsF'][dictDistr['sDistr']][str(vParamsS)][iRidx] # alpha norm f
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
        
    return vParamsS * mF**(vParamsS-1) - (vParamsS-1) * dFNorm**vParamsS 

###########################################################  
def PowSSharp(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional Power scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx] # alpha norm f_w for single r
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    dFwBarForDiv = np.min((1-10e-10, dFwBar))
   
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
        
    return mW * (vParamsS * (mF/(1-dFwBarForDiv))**(vParamsS-1) - (vParamsS-1) * (dFwNorm/(1-dFwBarForDiv))**vParamsS)

###########################################################  
def PowSFlat(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Censored Power scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx] # alpha norm f_w for single r
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
        
    return mW * vParamsS * mF**(vParamsS-1) + (1-mW) * vParamsS * dFwBar**(vParamsS-1) - (vParamsS-1) * (dFwNorm**vParamsS + dFwBar**vParamsS)

###########################################################  
def PowSSharpslog(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional Power scoring rule + slog correction HK

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx] # alpha norm f_w for single r
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    dFwBarForDiv = np.min((1-10e-10, dFwBar))
   
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = mW * (vParamsS * (mF/(1-dFwBarForDiv))**(vParamsS-1) - (vParamsS-1) * (dFwNorm/(1-dFwBarForDiv))**vParamsS)

    mSlogCorr = mW * np.log(1-dFwBarForDiv) + (1-mW) * np.log(dFwBarForDiv)
    
    return mSwSharp + mSlogCorr

###########################################################  
def PowSSharpsbar(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional Power scoring rule + sbar correction HK

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    """
    
    vR = np.array(dictW['vR']) # threshold grid
    if vR.ndim == 1:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    elif vR.ndim == 2:
        mW = dictW['fW'](mY, dictW['vParamsW'], vR[:,iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx] # alpha norm f_w for single r
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    dFwBarForDiv = np.min((1-10e-10, dFwBar))
   
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = mW * (vParamsS * (mF/(1-dFwBarForDiv))**(vParamsS-1) - (vParamsS-1) * (dFwNorm/(1-dFwBarForDiv))**vParamsS)

    mSbarCorr = mW * (np.log(1-dFwBarForDiv) + 1) - (1-dFwBarForDiv)
    
    return mSwSharp + mSbarCorr

######################################################################
############ CONTINUOUSLY RANKED PROBABILITY SCORING RULE ############
######################################################################

###########################################################  
def CRPSSharpBivariate(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional CRPS Holzmann and Klar, using expectation presentation of Energy Score

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: dictionary with numerical integration settings
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    
    Note: Requires adaptation for general weight functions
    """
   
    mR = np.array(dictW['vR']) # threshold grid
    vR = mR[:,iRidx].reshape(2) # selected threshold value
    mZ = dictPreCalc['NumIntSettings']['mZ']
    dMemoryGB = dictPreCalc['NumIntSettings']['dMemoryGB']
    mZ1 = mZ[0,:,:]
    mZ2 = mZ[1,:,:]
    
    def E1(mZ1, mZ2):

        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vW2 = dictW['fW'](mZ2, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        vFz2 = dictDistr['randDistr'].pdf(mZ2)

        # Estimate maximum batch size for memory constraints
        dMemoryLimit = dMemoryGB * (1024**3)  # Convert GiB to bytes
        dAvailableMemoryPerDiff = dMemoryLimit // 8  # Floats are 8 bytes
        dMaxElementsPerDiff = dAvailableMemoryPerDiff // (2 * len(mZ2))  # Account for pairwise computation
        iBatchSize = int(np.sqrt(dMaxElementsPerDiff))  # Derive batch size from elements
        
        dSum = 0
        for i in range(0, len(mZ1), iBatchSize):
            for j in range(0, len(mZ2), iBatchSize):
                mBatch1 = mZ1[i:i+iBatchSize]
                mBatch2 = mZ2[j:j+iBatchSize]
                    
                mPairwiseDiffs = mBatch1[:, None, :] - mBatch2[None, :, :]  # Shape: (iBatch1, iBatch2, 2)
                mPairwiseNorms = np.linalg.norm(mPairwiseDiffs, axis=2)  # Shape: (iBatch1, iBatch2)
                mWeightProduct = np.outer(vW1[i:i+iBatchSize], vW2[j:j+iBatchSize])
                mPdfProduct = np.outer(vFz1[i:i+iBatchSize], vFz2[j:j+iBatchSize])
                    
                dSum += np.sum(mWeightProduct * mPairwiseNorms * mPdfProduct)

        return dSum * (1/(1-dFwBar))**2 * (1/len(vFz1))**2
    
    def E2(mZ1):
        vY = np.array([mY[0], mY[1]])
        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        return  np.sum(vW1 * np.linalg.norm(mZ1 - vY) * (1/(1-dFwBar)) * vFz1 * (1/len(vFz1)))

    dFwBar = np.min((1-1e-10, dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx])) # \bar F_w for single r
    dWy = dictW['fW'](mY, dictW['vParamsW'], vR)
    
    dSSharp = dWy * (1/2 * E1(mZ1, mZ2) - E2(mZ1))
    
    return dSSharp

###########################################################  
def CRPSSharpslogBivariate(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional CRPS Holzmann and Klar, using expectation presentation of Energy Score

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: dictionary with numerical integration settings
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    
    Note: Requires adaptation for general weight functions
    """
   
    mR = np.array(dictW['vR']) # threshold grid
    vR = mR[:,iRidx].reshape(2) # selected threshold value
    mZ = dictPreCalc['NumIntSettings']['mZ']
    dMemoryGB = dictPreCalc['NumIntSettings']['dMemoryGB']
    mZ1 = mZ[0,:,:]
    mZ2 = mZ[1,:,:]
    
    def E1(mZ1, mZ2):

        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vW2 = dictW['fW'](mZ2, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        vFz2 = dictDistr['randDistr'].pdf(mZ2)

        # Estimate maximum batch size for memory constraints
        dMemoryLimit = dMemoryGB * (1024**3)  # Convert GiB to bytes
        dAvailableMemoryPerDiff = dMemoryLimit // 8  # Floats are 8 bytes
        dMaxElementsPerDiff = dAvailableMemoryPerDiff // (2 * len(mZ2))  # Account for pairwise computation
        iBatchSize = int(np.sqrt(dMaxElementsPerDiff))  # Derive batch size from elements
        
        dSum = 0
        for i in range(0, len(mZ1), iBatchSize):
            for j in range(0, len(mZ2), iBatchSize):
                mBatch1 = mZ1[i:i+iBatchSize]
                mBatch2 = mZ2[j:j+iBatchSize]
                    
                mPairwiseDiffs = mBatch1[:, None, :] - mBatch2[None, :, :]  # Shape: (iBatch1, iBatch2, 2)
                mPairwiseNorms = np.linalg.norm(mPairwiseDiffs, axis=2)  # Shape: (iBatch1, iBatch2)
                mWeightProduct = np.outer(vW1[i:i+iBatchSize], vW2[j:j+iBatchSize])
                mPdfProduct = np.outer(vFz1[i:i+iBatchSize], vFz2[j:j+iBatchSize])
                    
                dSum += np.sum(mWeightProduct * mPairwiseNorms * mPdfProduct)

        return dSum * (1/(1-dFwBar))**2 * (1/len(vFz1))**2
    
    def E2(mZ1):
        vY = np.array([mY[0], mY[1]])
        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        return  np.sum(vW1 * np.linalg.norm(mZ1 - vY) * (1/(1-dFwBar)) * vFz1 * (1/len(vFz1)))
    
    dFwBar = np.min((1-1e-10, dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx])) # \bar F_w for single r
    dWy = dictW['fW'](mY, dictW['vParamsW'], vR)
    
    dSSharp =  dWy * (1/2 * E1(mZ1, mZ2) - E2(mZ1))
    
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    dSlogCorr = dWy * np.log(1-dFwBarForLog) + (1-dWy) * np.log(dFwBarForLog)
    
    return dSSharp + dSlogCorr

###########################################################  
def CRPSSharpsbarBivariate(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional CRPS Holzmann and Klar, using expectation presentation of Energy Score

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: dictionary with numerical integration settings
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    
    Note: Requires adaptation for general weight functions
    """
   
    mR = np.array(dictW['vR']) # threshold grid
    vR = mR[:,iRidx].reshape(2) # selected threshold value
    mZ = dictPreCalc['NumIntSettings']['mZ']
    dMemoryGB = dictPreCalc['NumIntSettings']['dMemoryGB']
    mZ1 = mZ[0,:,:]
    mZ2 = mZ[1,:,:]
    
    def E1(mZ1, mZ2):

        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vW2 = dictW['fW'](mZ2, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        vFz2 = dictDistr['randDistr'].pdf(mZ2)

        # Estimate maximum batch size for memory constraints
        dMemoryLimit = dMemoryGB * (1024**3)  # Convert GiB to bytes
        dAvailableMemoryPerDiff = dMemoryLimit // 8  # Floats are 8 bytes
        dMaxElementsPerDiff = dAvailableMemoryPerDiff // (2 * len(mZ2))  # Account for pairwise computation
        iBatchSize = int(np.sqrt(dMaxElementsPerDiff))  # Derive batch size from elements
        
        dSum = 0
        for i in range(0, len(mZ1), iBatchSize):
            for j in range(0, len(mZ2), iBatchSize):
                mBatch1 = mZ1[i:i+iBatchSize]
                mBatch2 = mZ2[j:j+iBatchSize]
                    
                mPairwiseDiffs = mBatch1[:, None, :] - mBatch2[None, :, :]  # Shape: (iBatch1, iBatch2, 2)
                mPairwiseNorms = np.linalg.norm(mPairwiseDiffs, axis=2)  # Shape: (iBatch1, iBatch2)
                mWeightProduct = np.outer(vW1[i:i+iBatchSize], vW2[j:j+iBatchSize])
                mPdfProduct = np.outer(vFz1[i:i+iBatchSize], vFz2[j:j+iBatchSize])
                    
                dSum += np.sum(mWeightProduct * mPairwiseNorms * mPdfProduct)

        return dSum * (1/(1-dFwBar))**2 * (1/len(vFz1))**2
    
    def E2(mZ1):
        vY = np.array([mY[0], mY[1]])
        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        return  np.sum(vW1 * np.linalg.norm(mZ1 - vY) * (1/(1-dFwBar)) * vFz1 * (1/len(vFz1)))
    
    dFwBar = np.min((1-1e-10, dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx])) # \bar F_w for single r
    dWy = dictW['fW'](mY, dictW['vParamsW'], vR)
    
    dSSharp = dWy * (1/2 * E1(mZ1, mZ2) - E2(mZ1))
    
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    dSlogCorr = dWy * (np.log(1-dFwBarForLog) + 1) - (1-dFwBarForLog)
    
    return dSSharp + dSlogCorr

###########################################################  
def CRPSFlatBivariateL(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Censored CRPS - Single point (L)

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated   
    dictW :         dictionary, about weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: dictionary with numerical integration settings
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    
    Note: CRPS with multiple pivotal points, generalised censored measure has two point masses at a1 and a2,
    namely gamma Fbar at a1 and (1-gamma) Fbar at a2, with gamma >=0.
    
    """
    
    mR = np.array(dictW['vR']) # threshold grid
    vR = mR[:,iRidx].reshape(2) # selected threshold value
    mZ = dictPreCalc['NumIntSettings']['mZ']
    dMemoryGB = dictPreCalc['NumIntSettings']['dMemoryGB']
    mZ1 = mZ[0,:,:]
    mZ2 = mZ[1,:,:]
    
    dFwBar = np.min((1-1e-10, dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx])) # \bar F_w for single r
    dWy = dictW['fW'](mY, dictW['vParamsW'], vR)
    
    # E_{F^flat}|Y-Y'| 
    def E1(mZ1, mZ2):
        # Initialise weights and densities
        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vW2 = dictW['fW'](mZ2, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        vFz2 = dictDistr['randDistr'].pdf(mZ2)
        
        # Estimate maximum batch size for memory constraints
        dMemoryLimit = dMemoryGB * (1024**3)  # Convert GiB to bytes
        dAvailableMemoryPerDiff = dMemoryLimit // 8  # Floats are 8 bytes
        dMaxElementsPerDiff = dAvailableMemoryPerDiff // (2 * len(mZ2))  # Account for pairwise computation
        iBatchSize = int(np.sqrt(dMaxElementsPerDiff))  # Derive batch size from elements
        
        dSum = 0
        for i in range(0, len(mZ1), iBatchSize):
            for j in range(0, len(mZ2), iBatchSize):
                mBatch1 = mZ1[i:i+iBatchSize]
                mBatch2 = mZ2[j:j+iBatchSize]
                    
                mPairwiseDiffs = mBatch1[:, None, :] - mBatch2[None, :, :]  # Shape: (iBatch1, iBatch2, 2)
                mPairwiseNorms = np.linalg.norm(mPairwiseDiffs, axis=2)  # Shape: (iBatch1, iBatch2)
                mWeightProduct = np.outer(vW1[i:i+iBatchSize], vW2[j:j+iBatchSize])
                mPdfProduct = np.outer(vFz1[i:i+iBatchSize], vFz2[j:j+iBatchSize])
                    
                dSum += np.sum(mWeightProduct * mPairwiseNorms * mPdfProduct)
        
        dE1Part1 = dSum * (1 / len(vFz1))**2  # E|Y-Y'| - continuous part
        dE1Part2 = np.sum(vW1 * np.linalg.norm(mZ1 - vR, axis=1) * dFwBar * vFz1 * (1 / len(vFz1))) 
    
        return dE1Part1 + 2 * dE1Part2
    
    # w(y)E_{F^flat}||Y-y|| + (1-w(y)) E_{F^flat}||Y-r||
    def E2(mZ1):
        vY = np.array([mY[0], mY[1]])
        vW1 = dictW['fW'](mZ1, dictW['vParamsW'], vR)
        vFz1 = dictDistr['randDistr'].pdf(mZ1)
        
        dE2Part1 = np.sum(vW1 * np.linalg.norm(mZ1 - vY) * vFz1 * (1/len(vFz1))) + dFwBar *  np.linalg.norm(vR - vY)
        dE2Part2 = np.sum(vW1 * np.linalg.norm(mZ1 - vR) * vFz1 * (1/len(vFz1)))
        
        return  dWy * dE2Part1 + (1-dWy) * dE2Part2

    dSFlat = 1/2 * E1(mZ1, mZ2) - E2(mZ1)
      
    return dSFlat