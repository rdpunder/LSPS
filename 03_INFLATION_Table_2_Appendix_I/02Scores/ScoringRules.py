#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: File containing scoring rule functions 
"""

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration

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
    
    vR = np.array(dictW['vR'])
    dictAlphaNormsFw = {}
    
    if type(dictDistr['randDistr']) == dict:
        for dAlpha in vAlpha[vAlpha > 0]:
            vFwNorm = np.zeros(vR.shape)
            if dictW['fW'].__name__ == 'fWIndicatorL':
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, -np.inf, vR[vIntR[i]],epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)
                dictAlphaNormsFw[str(dAlpha)] = vFwNorm    
            elif dictW['fW'].__name__ == 'fWIndicatorR':
                for i in range(vIntR.size):
                    vFwNorm[vIntR[i]]= (integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, vR[vIntR[i]], np.inf, epsabs=1e-15, epsrel=1e-15)[0])**(1/dAlpha)
                dictAlphaNormsFw[str(dAlpha)] = vFwNorm     
            elif dictW['fW'].__name__ == 'fWIndicatorC':
                for i in range(vIntR.size): # has been checked for large intervals relative to distribution 
                    vFwNorm[vIntR[i]] = (integrate.quad(lambda x:  (dictDistr['randDistr']['pdf'](x) * dictW['fW'](x, dictW['vParamsW'], vR[vIntR[i]]))**dAlpha, dictW['vParamsW']-vR[vIntR[i]], dictW['vParamsW']+vR[vIntR[i]],epsabs=1e-15, epsrel=1e-15, limit=100)[0])**(1/dAlpha) 
                dictAlphaNormsFw[str(dAlpha)] = vFwNorm
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
    Calculate Fbar

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
    vector, length vR with \bar F_w for each threshold r
    """    
    
    vR = np.array(dictW['vR'])
    
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
    iRidx :         integer, index vR for which score should be evaluated (placeholder)          
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
    
    vR = np.array(dictW['vR']) # threshold grid
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    iRidx :         integer, index vR for which score should be evaluated
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx]
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = mW * (mW * mF/dFwNorm)**(vParamsS-1)    
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx]
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = mW * (mW * mF/dFwNorm)**(vParamsS-1)    
    mSwSharp[mW==0] = 0 #potential numerical nan mapped to 0 [enforcing 0*nan=0]
    
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    dFwNorm = dictPreCalc['AlphaNormsFw'][dictDistr['sDistr']][str(vParamsS)][iRidx]
    
    # Density f(y) - via stats object or user-defined function
    if type(dictDistr['randDistr']) == dict:
        mF = np.array(dictDistr['randDistr']['pdf'](mY)) 
    else:
        mF = np.array(dictDistr['randDistr'].pdf(mY)) 
    
    mSwSharp = mW * (mW * mF/dFwNorm)**(vParamsS-1)    
    mSwSharp[mW==0] = 0 #potential numerical nan mapped to 0 [enforcing 0*nan=0]
    
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
    Regular Power scoring rule

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
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
def CRPS(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Regular CRPS scoring rule

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
    
    dictCRPSNumInt = vParamsS
    dDeltaZ = dictCRPSNumInt['dDeltaZ']
    dStart = dictCRPSNumInt['dMinInf'] 
    dEnd = dictCRPSNumInt['dPlusInf'] 
    vZ = np.arange(dStart, dEnd, dDeltaZ) 
    
    if type(dictDistr['randDistr']) == dict:
        vFz = np.array(dictDistr['randDistr']['cdf'](vZ)) # F(z)
    else:
        vFz = np.array(dictDistr['randDistr'].cdf(vZ)) # F(z)
    
    # Note: further vectorization is slower 
    mLoss = np.zeros(mY.shape)
    for i in range(vZ.size):
        mLoss += (vFz[i] - (mY <= vZ[i]))**2 * dDeltaZ
    
    return -mLoss

###########################################################  
def twCRPS(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Threshold weighted CRPS Gneiting

    Parameters
    ----------
    dictDistr :     dictionary, distribution
                        randDistr :  distribution F, stats object
                        sRandDistr : name distribution
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    iRidx :         integer, index vR for which score should be evaluated
    dictW :         dictionary, about weight function: 
                        fW : selected weight function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: dictionary with numerical integration settings
    dictPreCalc :   dictionary, precalculated values
    
    Returns
    ----------
    matrix, shape mY, with calculated scores
    
    Note: Requires adaptation for general weight functions
    """

    vR = np.array(dictW['vR']) # threshold grid
    dR = vR[iRidx] # selected threshold value
    
    # The integral simplifies if the weight function is an indicator function
    # introduce booleans for these cases
    iLeftIndicator = int(dictW['fW'].__name__ == 'fWIndicatorL')
    iCentreIndicator = int(dictW['fW'].__name__ == 'fWIndicatorC')
    iRightIndicator = int(dictW['fW'].__name__ == 'fWIndicatorR')
    iLorCIndicator = int(np.max([iLeftIndicator, iCentreIndicator]))
    iCorRIndicator = int(np.max([iCentreIndicator, iRightIndicator]))
    dC = dictW['vParamsW']
    
    dictCRPSNumInt = vParamsS
    dDeltaZ = dictCRPSNumInt['dDeltaZ']
    dStart = dictCRPSNumInt['dMinInf'] * (1-iCorRIndicator) + dR * iRightIndicator + (dC-dR) * iCentreIndicator 
    dEnd = dictCRPSNumInt['dPlusInf'] * (1-iLorCIndicator) + (dC + dR) * iLorCIndicator 
    
    vZ = np.arange(dStart, dEnd, dDeltaZ) 
    vWz = dictW['fW'](vZ, dictW['vParamsW'], vR[iRidx]) # w(z) for single r
    if type(dictDistr['randDistr']) == dict:
        vFz = np.array(dictDistr['randDistr']['cdf'](vZ)) # F(z)
    else:
        vFz = np.array(dictDistr['randDistr'].cdf(vZ)) # F(z)
    
    # Note: further vectorization makes the function slower 
    mLoss = np.zeros(mY.shape)
    for i in range(vZ.size):
        mLoss += vWz[i] * (vFz[i] - (mY <= vZ[i]))**2 * dDeltaZ
    
    return -mLoss

###########################################################  
def CRPSSharp(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional CRPS Holzmann and Klar + slog correction HK

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
   
    vR = np.array(dictW['vR']) # threshold grid
    dR = vR[iRidx] # selected threshold value
    
    # The integral simplifies if the weight function is an indicator function
    # introduce booleans for these cases
    iLeftIndicator = int(dictW['fW'].__name__ == 'fWIndicatorL')
    iCentreIndicator = int(dictW['fW'].__name__ == 'fWIndicatorC')
    iRightIndicator = int(dictW['fW'].__name__ == 'fWIndicatorR')
    iLorCIndicator = int(np.max([iLeftIndicator, iCentreIndicator]))
    iCorRIndicator = int(np.max([iCentreIndicator, iRightIndicator]))
    iTailsIndicator = int(dictW['fW'].__name__ == 'fWIndicatorTails')
    dC = dictW['vParamsW']
    
    dictCRPSNumInt = vParamsS
    dDeltaZ = dictCRPSNumInt['dDeltaZ']
    dStart = dictCRPSNumInt['dMinInf'] * (1-iCorRIndicator) + dR * iRightIndicator + (dC - dR) * iCentreIndicator 
    dEnd = dictCRPSNumInt['dPlusInf'] * (1-iLorCIndicator) + (dC + dR) * iLorCIndicator 
    
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    dFwBar = np.min((1-1e-10, dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx])) # \bar F_w for single r
    
    vZ = np.arange(dStart, dEnd, dDeltaZ) 
    if np.max([iLeftIndicator, iCentreIndicator, iRightIndicator]) == 1:
        if type(dictDistr['randDistr']) == dict:
            vFzSharp = (np.array(dictDistr['randDistr']['cdf'](vZ))-iCentreIndicator*np.array(dictDistr['randDistr']['cdf'](dC-dR))-iRightIndicator*np.array(dictDistr['randDistr']['cdf'](-dR)))/(np.max((1e-10,1-dFwBar))) 
        else:
            vFzSharp = (np.array(dictDistr['randDistr'].cdf(vZ))-iCentreIndicator*np.array(dictDistr['randDistr'].cdf(dC-dR))-iRightIndicator*np.array(dictDistr['randDistr'].cdf(-dR)))/(np.max((1e-10,1-dFwBar))) 
    elif iTailsIndicator == 1 and type(dictDistr['randDistr']) == dict:
        vZ = np.arange(dictCRPSNumInt['dMinInf'], dictCRPSNumInt['dPlusInf'], dDeltaZ) 
        vFzSharp1 = np.array(dictDistr['randDistr']['cdf'](vZ[vZ<dC-dR]))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp2 = np.ones(len(vZ[(vZ>dC-dR) & (vZ<dC+dR)])) * np.array(dictDistr['randDistr']['cdf'](np.array([dC-dR])))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp3 = (np.array(dictDistr['randDistr']['cdf'](vZ[vZ>dC+dR]))-dFwBar)/(np.max((1e-10,1-dFwBar))) 
        vFzSharp = np.hstack((vFzSharp1,vFzSharp2,vFzSharp3))
    elif iTailsIndicator == 1 and type(dictDistr['randDistr']) != dict:
        vZ = np.arange(dictCRPSNumInt['dMinInf'], dictCRPSNumInt['dPlusInf'], dDeltaZ) 
        vFzSharp1 = np.array(dictDistr['randDistr'].cdf(vZ[vZ<dC-dR]))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp2 = np.ones(len(vZ[(vZ>dC-dR) & (vZ<dC+dR)])) * np.array(dictDistr['randDistr'].cdf(np.array([dC-dR])))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp3 = (np.array(dictDistr['randDistr'].cdf(vZ[vZ>dC+dR]))-dFwBar)/(np.max((1e-10,1-dFwBar))) 
        vFzSharp = np.hstack((vFzSharp1,vFzSharp2,vFzSharp3))    
    
    mLossInt = np.zeros(mY.shape)
    for i in range(vZ.size):
        mLossInt +=  (vFzSharp[i] - (mY <= vZ[i]))**2 * dDeltaZ
    
    mLoss = mW * mLossInt 
    
    return -mLoss

###########################################################  
def CRPSSharpsbar(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional CRPS Holzmann and Klar + sbar correction HK

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
    
   
    vR = np.array(dictW['vR']) # threshold grid
    dR = vR[iRidx] # selected threshold value
    
    # The integral simplifies if the weight function is an indicator function
    # introduce booleans for these cases
    iLeftIndicator = int(dictW['fW'].__name__ == 'fWIndicatorL')
    iCentreIndicator = int(dictW['fW'].__name__ == 'fWIndicatorC')
    iRightIndicator = int(dictW['fW'].__name__ == 'fWIndicatorR')
    iLorCIndicator = int(np.max([iLeftIndicator, iCentreIndicator]))
    iCorRIndicator = int(np.max([iCentreIndicator, iRightIndicator]))
    iTailsIndicator = int(dictW['fW'].__name__ == 'fWIndicatorTails')
    dC = dictW['vParamsW']
    
    dictCRPSNumInt = vParamsS
    dDeltaZ = dictCRPSNumInt['dDeltaZ']
    dStart = dictCRPSNumInt['dMinInf'] * (1-iCorRIndicator) + dR * iRightIndicator + (dC - dR) * iCentreIndicator 
    dEnd = dictCRPSNumInt['dPlusInf'] * (1-iLorCIndicator) + (dC + dR) * iLorCIndicator 
    
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    dFwBar = np.min((1-1e-10, dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx])) # \bar F_w for single r
    
    vZ = np.arange(dStart, dEnd, dDeltaZ) 
    if np.max([iLeftIndicator, iCentreIndicator, iRightIndicator]) == 1:
        if type(dictDistr['randDistr']) == dict:
            vFzSharp = (np.array(dictDistr['randDistr']['cdf'](vZ))-iCentreIndicator*np.array(dictDistr['randDistr']['cdf'](dC-dR))-iRightIndicator*np.array(dictDistr['randDistr']['cdf'](-dR)))/(np.max((1e-10,1-dFwBar))) 
        else:
            vFzSharp = (np.array(dictDistr['randDistr'].cdf(vZ))-iCentreIndicator*np.array(dictDistr['randDistr'].cdf(dC-dR))-iRightIndicator*np.array(dictDistr['randDistr'].cdf(-dR)))/(np.max((1e-10,1-dFwBar))) 
    elif iTailsIndicator == 1 and type(dictDistr['randDistr']) == dict:
        vZ = np.arange(dictCRPSNumInt['dMinInf'], dictCRPSNumInt['dPlusInf'], dDeltaZ) 
        vFzSharp1 = np.array(dictDistr['randDistr']['cdf'](vZ[vZ<dC-dR]))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp2 = np.ones(len(vZ[(vZ>dC-dR) & (vZ<dC+dR)])) * np.array(dictDistr['randDistr']['cdf'](np.array([dC-dR])))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp3 = (np.array(dictDistr['randDistr']['cdf'](vZ[vZ>dC+dR]))-dFwBar)/(np.max((1e-10,1-dFwBar))) 
        vFzSharp = np.hstack((vFzSharp1,vFzSharp2,vFzSharp3))
    elif iTailsIndicator == 1 and type(dictDistr['randDistr']) != dict:
        vZ = np.arange(dictCRPSNumInt['dMinInf'], dictCRPSNumInt['dPlusInf'], dDeltaZ) 
        vFzSharp1 = np.array(dictDistr['randDistr'].cdf(vZ[vZ<dC-dR]))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp2 = np.ones(len(vZ[(vZ>dC-dR) & (vZ<dC+dR)])) * np.array(dictDistr['randDistr'].cdf(np.array([dC-dR])))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp3 = (np.array(dictDistr['randDistr'].cdf(vZ[vZ>dC+dR]))-dFwBar)/(np.max((1e-10,1-dFwBar))) 
        vFzSharp = np.hstack((vFzSharp1,vFzSharp2,vFzSharp3))    
    
    mLossInt = np.zeros(mY.shape)
    for i in range(vZ.size):
        mLossInt +=  (vFzSharp[i] - (mY <= vZ[i]))**2 * dDeltaZ
    
    mLoss = mW * mLossInt 
    
    mSwSharp = -mLoss
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    mSbarCorr = mW * (np.log(1-dFwBarForLog) + 1) - (1-dFwBarForLog)
    
    return mSwSharp + mSbarCorr

###########################################################  
def CRPSSharpslog(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Conditional CRPS Holzmann and Klar + Logarithmic correction

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
    
   
    vR = np.array(dictW['vR']) # threshold grid
    dR = vR[iRidx] # selected threshold value
    
    # The integral simplifies if the weight function is an indicator function
    # introduce booleans for these cases
    iLeftIndicator = int(dictW['fW'].__name__ == 'fWIndicatorL')
    iCentreIndicator = int(dictW['fW'].__name__ == 'fWIndicatorC')
    iRightIndicator = int(dictW['fW'].__name__ == 'fWIndicatorR')
    iLorCIndicator = int(np.max([iLeftIndicator, iCentreIndicator]))
    iCorRIndicator = int(np.max([iCentreIndicator, iRightIndicator]))
    iTailsIndicator = int(dictW['fW'].__name__ == 'fWIndicatorTails')
    dC = dictW['vParamsW']
    
    dictCRPSNumInt = vParamsS
    dDeltaZ = dictCRPSNumInt['dDeltaZ']
    dStart = dictCRPSNumInt['dMinInf'] * (1-iCorRIndicator) + dR * iRightIndicator + (dC - dR) * iCentreIndicator 
    dEnd = dictCRPSNumInt['dPlusInf'] * (1-iLorCIndicator) + (dC + dR) * iLorCIndicator 
    
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    dFwBar = np.min((1-1e-10, dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx])) # \bar F_w for single r
    
    vZ = np.arange(dStart, dEnd, dDeltaZ) 
    if np.max([iLeftIndicator, iCentreIndicator, iRightIndicator]) == 1:
        if type(dictDistr['randDistr']) == dict:
            vFzSharp = (np.array(dictDistr['randDistr']['cdf'](vZ))-iCentreIndicator*np.array(dictDistr['randDistr']['cdf'](dC-dR))-iRightIndicator*np.array(dictDistr['randDistr']['cdf'](-dR)))/(np.max((1e-10,1-dFwBar))) 
        else:
            vFzSharp = (np.array(dictDistr['randDistr'].cdf(vZ))-iCentreIndicator*np.array(dictDistr['randDistr'].cdf(dC-dR))-iRightIndicator*np.array(dictDistr['randDistr'].cdf(-dR)))/(np.max((1e-10,1-dFwBar))) 
    elif iTailsIndicator == 1 and type(dictDistr['randDistr']) == dict:
        vZ = np.arange(dictCRPSNumInt['dMinInf'], dictCRPSNumInt['dPlusInf'], dDeltaZ) 
        vFzSharp1 = np.array(dictDistr['randDistr']['cdf'](vZ[vZ<dC-dR]))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp2 = np.ones(len(vZ[(vZ>dC-dR) & (vZ<dC+dR)])) * np.array(dictDistr['randDistr']['cdf'](np.array([dC-dR])))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp3 = (np.array(dictDistr['randDistr']['cdf'](vZ[vZ>dC+dR]))-dFwBar)/(np.max((1e-10,1-dFwBar))) 
        vFzSharp = np.hstack((vFzSharp1,vFzSharp2,vFzSharp3))
    elif iTailsIndicator == 1 and type(dictDistr['randDistr']) != dict:
        vZ = np.arange(dictCRPSNumInt['dMinInf'], dictCRPSNumInt['dPlusInf'], dDeltaZ) 
        vFzSharp1 = np.array(dictDistr['randDistr'].cdf(vZ[vZ<dC-dR]))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp2 = np.ones(len(vZ[(vZ>dC-dR) & (vZ<dC+dR)])) * np.array(dictDistr['randDistr'].cdf(np.array([dC-dR])))/(np.max((1e-10,1-dFwBar))) 
        vFzSharp3 = (np.array(dictDistr['randDistr'].cdf(vZ[vZ>dC+dR]))-dFwBar)/(np.max((1e-10,1-dFwBar))) 
        vFzSharp = np.hstack((vFzSharp1,vFzSharp2,vFzSharp3))    
    
    mLossInt = np.zeros(mY.shape)
    for i in range(vZ.size):
        mLossInt +=  (vFzSharp[i] - (mY <= vZ[i]))**2 * dDeltaZ
    
    mLoss = mW * mLossInt 
    
    mSwSharp = -mLoss
    dFwBarForLog = np.min((1-10e-10, dFwBar))
    mSlogCorr = mW * np.log(1-dFwBarForLog) + (1-mW) * np.log(dFwBarForLog)
    
    return mSwSharp + mSlogCorr

###########################################################  
def CRPSGenFlatGamma(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Censored CRPS - Multiple Points

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
    
    vR = np.array(dictW['vR']) # threshold grid
    dR = vR[iRidx] # selected threshold value
    
    # The integral simplifies if the weight function is an indicator function
    # introduce booleans for these cases
    iCentreIndicator = int(dictW['fW'].__name__ == 'fWIndicatorC')
    iTailsTwoPivIndicator = 0 # manually choose whether mass should be distributed between to points
    iTailsIndicator = int(dictW['fW'].__name__ == 'fWIndicatorTails')
    # Note: For left- and right tail indicator: we use twCRPS - simple numerical integration
    
    # Pivotal points
    dC = dictW['vParamsW']
    dA1 = dC - dR
    dA2 = dC + dR
    dictCRPSSettings = vParamsS
    dDeltaZ = dictCRPSSettings['dDeltaZ']
    dStart = dA1
    dEnd = dA2

    
    # Estimate gamma, based on data
    dGamma = dictPreCalc['Gamma']
    
    vZ1 = np.arange(dStart, dEnd, dDeltaZ) 
    
    if iCentreIndicator:
        vZ1 = np.arange(dA1, dA2, dDeltaZ) 
    elif np.max([iTailsIndicator, iTailsTwoPivIndicator]):
        vZ1 = np.hstack((np.arange(dictCRPSSettings['dMinInf'], dA1), np.arange(dA2, dictCRPSSettings['dPlusInf'])))
    vZ2 = np.copy(vZ1)
    
    # Tails, one pivot: put mass at c
    if iTailsIndicator: 
        dGamma = 1 # outside A is considered to be on target
        dA1 = dC
        dA2 = dA1
   
    if type(dictDistr['randDistr']) == dict:
        vfz1 = np.array(dictDistr['randDistr']['pdf'](vZ1)).reshape(vZ1.shape) # f(z)
    else:
        vfz1 = np.array(dictDistr['randDistr'].pdf(vZ1)) # f(z)
        
    vfz2 = np.copy(vfz1) # f(z)
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    
    ## E|Y-Y'|
    # Continuous part
    iMatrixSizeMax = dictCRPSSettings['iMatrixSizeMax']
    dE1=0
    iUpperLim = int(np.floor(vZ1.size/iMatrixSizeMax))
    for i in range(iUpperLim):
        dE1 += np.sum(np.abs(np.subtract.outer(vZ1[i*iMatrixSizeMax:(i+1)*iMatrixSizeMax], vZ2)) * np.outer(vfz1[i*iMatrixSizeMax:(i+1)*iMatrixSizeMax], vfz2)) * dDeltaZ**2
    dE1 += np.sum(np.abs(np.subtract.outer(vZ1[iUpperLim*iMatrixSizeMax:], vZ2)) * np.outer(vfz1[iUpperLim*iMatrixSizeMax:], vfz2)) * dDeltaZ**2

    # second part, at least one a_1 or a_2:
    dE1 += 2*dFwBar * (np.sum((dGamma * np.abs(vZ1 - dA1) + (1-dGamma) * np.abs(vZ1 - dA2)) * vfz1) * dDeltaZ + dGamma*(1-dGamma) * dFwBar * np.abs(dA1 - dA2))
    
    ## E|Y-y|
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    
    def CRPSofFlat(mX):
        mEout = 0
        for i in range(vZ1.size):
            # y is not censored - continuous part of z
            mEout += np.abs(mX - vZ1[i]) * vfz1[i] * dDeltaZ 
        mEout += dFwBar * (dGamma * np.abs(mX-dA1) + (1-dGamma) * np.abs(mX-dA2))
        return  1/2*dE1 - mEout 
        
    mY1 = np.copy(mY)
    mOut = np.zeros(mY.shape)
    mOut[mW > 0] =  CRPSofFlat(mY1[mW>0])
    mOut[mW == 0] = dGamma * CRPSofFlat(dA1) + (1-dGamma) * CRPSofFlat(dA2)     
    return mOut

###########################################################  
def CRPSGenFlatGammaFixed(dictDistr, mY, iRidx, dictW, vParamsS=False, dictPreCalc=False):
    """
    Purpose
    ----------
    Censored CRPS - Multiple Points

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
    
    vR = np.array(dictW['vR']) # threshold grid
    dR = vR[iRidx] # selected threshold value
    
    # The integral simplifies if the weight function is an indicator function
    # introduce booleans for these cases
    iCentreIndicator = int(dictW['fW'].__name__ == 'fWIndicatorC')
    iTailsTwoPivIndicator = 0 # manually choose whether mass should be distributed between to points
    iTailsIndicator = int(dictW['fW'].__name__ == 'fWIndicatorTails')
    # Note: For left- and right tail indicator: we use twCRPS - simple numerical integration
    
    # Pivotal points
    dC = dictW['vParamsW']
    dA1 = dC - dR
    dA2 = dC + dR
    dictCRPSSettings = vParamsS
    dDeltaZ = dictCRPSSettings['dDeltaZ']
    dStart = dA1
    dEnd = dA2
    
    vZ1 = np.arange(dStart, dEnd, dDeltaZ) 
    
    if iCentreIndicator:
        vZ1 = np.arange(dA1, dA2, dDeltaZ) 
    elif np.max([iTailsIndicator, iTailsTwoPivIndicator]):
        vZ1 = np.hstack((np.arange(dictCRPSSettings['dMinInf'], dA1), np.arange(dA2, dictCRPSSettings['dPlusInf'])))
    vZ2 = np.copy(vZ1)
    
    # Tails, one pivot: put mass at c
    if iTailsIndicator: 
        dGamma = 1 # outside A is considered to be on target
        dA1 = dC
        dA2 = dA1
    else:
        dGamma = dictPreCalc['Gamma']
   
    if type(dictDistr['randDistr']) == dict:
        vfz1 = np.array(dictDistr['randDistr']['pdf'](vZ1)).reshape(vZ1.shape) # f(z)
    else:
        vfz1 = np.array(dictDistr['randDistr'].pdf(vZ1)) # f(z)
        
    vfz2 = np.copy(vfz1) # np.array(dictDistr['randDistr'].pdf(vZ2)) # f(z)
    dFwBar = dictPreCalc['DistrBar'][dictDistr['sDistr']][iRidx] # \bar F_w for single r
    
    ## E|Y-Y'|
    # Continuous part
    iMatrixSizeMax = dictCRPSSettings['iMatrixSizeMax']
    dE1=0
    iUpperLim = int(np.floor(vZ1.size/iMatrixSizeMax))
    for i in range(iUpperLim):
        dE1 += np.sum(np.abs(np.subtract.outer(vZ1[i*iMatrixSizeMax:(i+1)*iMatrixSizeMax], vZ2)) * np.outer(vfz1[i*iMatrixSizeMax:(i+1)*iMatrixSizeMax], vfz2)) * dDeltaZ**2
    dE1 += np.sum(np.abs(np.subtract.outer(vZ1[iUpperLim*iMatrixSizeMax:], vZ2)) * np.outer(vfz1[iUpperLim*iMatrixSizeMax:], vfz2)) * dDeltaZ**2

    # second part, at least one a_1 or a_2:
    dE1 += 2*dFwBar * (np.sum((dGamma * np.abs(vZ1 - dA1) + (1-dGamma) * np.abs(vZ1 - dA2)) * vfz1) * dDeltaZ + dGamma*(1-dGamma) * dFwBar * np.abs(dA1 - dA2))
    
    ## E|Y-y|
    mW = dictW['fW'](mY, dictW['vParamsW'], vR[iRidx]) # w(y) for single r
    
    def CRPSofFlat(mX):
        mEout = 0
        for i in range(vZ1.size):
            # y is not censored - continuous part of z
            mEout += np.abs(mX - vZ1[i]) * vfz1[i] * dDeltaZ 
        mEout += dFwBar * (dGamma * np.abs(mX-dA1) + (1-dGamma) * np.abs(mX-dA2))
        return  1/2*dE1 - mEout 
        
    mY1 = np.copy(mY)
    mOut = np.zeros(mY.shape)
    mOut[mW > 0] =  CRPSofFlat(mY1[mW>0])
    mOut[mW == 0] = dGamma * CRPSofFlat(dA1) + (1-dGamma) * CRPSofFlat(dA2)     
    return mOut
