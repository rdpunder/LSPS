#### Imports

# Fundamentals
import numpy as np  
import pandas as pd
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration
from scipy.integrate import quad

# Functions
###########################################################  
def funcWIndicatorL(mY, dR, vParamsW=None):
    """
    Purpose
    ----------
    Compute indicator-based weight function for left-sided threshold

    Parameters
    ----------
    mY :           array, observations  
    dR :           float, threshold r  
    vParamsW :     float or array, placeholder  

    Returns
    ----------
    array, indicator weights equal to one for observations less than dR and zero otherwise
    """
    
    mWbool = mY < dR 
    return np.ones(mWbool.shape) * mWbool

###########################################################  
def funcWIndicatorR(mY, dR, vParamsW=None):
    """
    Purpose
    ----------
    Compute indicator-based weight function for right-sided threshold

    Parameters
    ----------
    mY :           array, observations  
    dR :           float, threshold r  
    vParamsW :     float or array, placeholder    

    Returns
    ----------
    array, indicator weights equal to one for observations greater than dR and zero otherwise
    """
    
    mWbool = mY > dR 
    return np.ones(mWbool.shape) * mWbool

###########################################################  
def funcWIndicatorCfixed(mY, dR, vParamsW=0):
    """
    Purpose
    ----------
    Compute indicator-based weight function for fixed interval centered at dR with width vParamsW

    Parameters
    ----------
    mY :           array, observations  
    dR :           float, threshold r  
    vParamsW :     float or array, width of the interval  

    Returns
    ----------
    array, indicator weights equal to one inside (dR − vParamsW⁄2, dR + vParamsW⁄2) and zero outside
    """

    mWbool = (mY > (dR - vParamsW/2)) * (mY < (dR + vParamsW/2))
    return np.ones(mWbool.shape) * mWbool

###########################################################
def funcWIndicatorC(mY, dR, vParamsW=0):
    """
    Purpose
    ----------
    Compute indicator-based weight function for centered interval (vParamsW ± dR)

    Parameters
    ----------
    mY :           array, observations  
    dR :           float, threshold r  
    vParamsW :     float or array, center of the interval  

    Returns
    ----------
    array, indicator weights equal to one inside (vParamsW − dR, vParamsW + dR) and zero outside
    """
    
    mWbool = (mY > vParamsW-dR) * (mY < vParamsW+dR)
    return np.ones(mWbool.shape) * mWbool

###########################################################  
def funcLogS(mY, randF, vParamsS=None):
    """
    Purpose
    ----------
    Compute unweighted logarithmic score 

    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution  
    vParamsS :     float or array, additional parameter(s) for scoring rule  

    Returns
    ----------
    array, logarithmic score values evaluated at each observation
    """
    
    # Density
    mF = np.array(randF.pdf(mY)) # f(y)
    mF[mF==0] = 1e-100 # avoid numerical zeros
    
    # Score
    mS = np.log(mF)
    
    return mS

###########################################################  
def funcLogSLocal(mY, randF, dR, vParamsS=None, vParamsW=0, sSide='L'):
    """
    Purpose
    ----------
    Compute localized logarithmic scores and return both sharp and flat score variants
    
    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution
    dR :           float, threshold r  
    vParamsS :     float or array, additional parameter(s) for scoring rule
    vParamsW :     float or array, additional parameter(s) for weight function  
    sSide :        string, side or configuration of the weight function ('L', 'R', 'C', 'Cfixed')  
    
    Returns
    ----------
    list, contains two arrays corresponding to:
        - conditional score
        - censored score
    """

    # Density
    mF = np.array(randF.pdf(mY)) # f(y)
    mF[mF==0] = 1e-100 # avoid numerical zeros
    
    # Weights
    if sSide == 'L':
        dFBar = 1 - randF.cdf(dR) # Fbar
        mW = funcWIndicatorL(mY, dR) # w(y)
    elif sSide == 'R':
        dFBar = randF.cdf(dR) # Fbar
        mW = funcWIndicatorR(mY, dR) # w(y)
    elif sSide == 'C':
        dFBar = randF.cdf(vParamsW-dR) + (1-randF.cdf(vParamsW+dR)) # Fbar
        mW = funcWIndicatorC(mY, dR, vParamsW) # w(y)
    elif sSide == 'Cfixed':
        dFBar = randF.cdf(dR-vParamsW/2) + (1-randF.cdf(dR+vParamsW/2)) # Fbar
        mW = funcWIndicatorCfixed(mY, dR, vParamsW) # w(y)
        
    # Localized scores
    mSFSharp = mW * (np.log(mF) - np.log(1-dFBar)) 
    mSFFlat = mW * np.log(mF) + (1-mW) * np.log(dFBar)
    
    return [mSFSharp, mSFFlat]

###########################################################  
def funcPsSphS(mY, randF, vParamsS=None):
    """
    Purpose
    ----------
    Compute unweighted pseudospherical score

    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution  
    vParamsS :     float or array, additional parameter(s) for scoring rule  

    Returns
    ----------
    array, pseudospherical score values evaluated at each observation
    """
    
    # Density
    mF = np.array(randF.pdf(mY)) # f(y)
    mF[mF==0] = 1e-100 # avoid numerical zeros
    
    # Norm
    dFNorm = (integrate.quad(lambda x:  (randF.pdf(x))**vParamsS, -np.inf, np.inf)[0])**(1/vParamsS) # ||f||_alpha
    
    # Score
    mS = (mF/dFNorm)**(vParamsS-1)

    return mS

###########################################################  
def funcPsSphSLocal(mY, randF, dR, vParamsS=2, vParamsW=0, sSide='L'):
    """
    Purpose
    ----------
    Compute localized pseudospherical scores and return both sharp and flat score variants
    
    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution
    dR :           float, threshold r
    vParamsS :     float or array, additional parameter(s) for scoring rule  
    vParamsW :     float or array, additional parameter(s) for weight function  
    sSide :        string, side or configuration of the weight function ('L', 'R', 'C', 'Cfixed')  
    
    Returns
    ----------
    list, contains two arrays corresponding to:
        - conditional score  
        - censored score  
    """
    
    dAlpha = vParamsS
    if sSide == 'L':
        mW = funcWIndicatorL(mY, dR) # w(y) for single r
        dFBar = 1 - randF.cdf(dR) # Fbar
        
    # Weights
    if sSide == 'L':
        dFBar = 1 - randF.cdf(dR) # Fbar
        mW = funcWIndicatorL(mY, dR) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, -np.inf, dR)[0])**(1/dAlpha) # ||fw||_alpha
    elif sSide == 'R':
        dFBar = randF.cdf(dR) # Fbar
        mW = funcWIndicatorR(mY, dR) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, dR, np.inf)[0])**(1/dAlpha) # ||fw||_alpha
    elif sSide == 'C':
        dFBar = randF.cdf(vParamsW-dR) + (1-randF.cdf(vParamsW+dR)) # Fbar
        mW = funcWIndicatorC(mY, dR, vParamsW) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, vParamsW-dR, vParamsW+dR)[0])**(1/dAlpha) # ||fw||_alpha    
    elif sSide == 'Cfixed':
        dFBar = randF.cdf(dR-vParamsW/2) + (1-randF.cdf(dR+vParamsW/2)) # Fbar
        mW = funcWIndicatorCfixed(mY, dR, vParamsW) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, dR-vParamsW/2, dR+vParamsW/2)[0])**(1/dAlpha) # ||fw||_alpha
            
    mF = np.array(randF.pdf(mY)) # f(y)

    # Localized scores
    mSFSharp = mW * (mW * mF/dFwNorm)**(dAlpha-1)
    mSFFlat = (mW * (mW * mF)**(dAlpha-1) + (1-mW) * dFBar**(dAlpha-1)) / (dFwNorm**dAlpha + dFBar**dAlpha )**((dAlpha-1)/dAlpha)
    
    return [mSFSharp, mSFFlat]

###########################################################  
def funcPowS(mY, randF, vParamsS=None):
    """
    Purpose
    ----------
    Compute unweighted power score 

    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution  
    vParamsS :     float or array, additional parameter(s) for scoring rule  

    Returns
    ----------
    array, power score values evaluated at each observation
    """
    
    # Density
    mF = np.array(randF.pdf(mY)) # f(y)
    mF[mF==0] = 1e-100 # avoid numerical zeros
    
    # Norm
    dFNorm = (integrate.quad(lambda x:  (randF.pdf(x))**vParamsS, -np.inf, np.inf)[0])**(1/vParamsS) # ||f||_alpha
    
    # Score
    mS = vParamsS * mF**(vParamsS-1) - (vParamsS-1) * dFNorm**vParamsS 

    return mS

###########################################################  
def funcPowSLocal(mY, randF, dR, vParamsS=2, vParamsW=0, sSide='L'):
    """
    Purpose
    ----------
    Compute localized power scores and return both sharp and flat score variants
    
    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution  
    dR :           float, threshold r  
    vParamsS :     float or array, additional parameter(s) for scoring rule  
    vParamsW :     float or array, additional parameter(s) for weight function  
    sSide :        string, side or configuration of the weight function ('L', 'R', 'C', 'Cfixed')  
    
    Returns
    ----------
    list, contains two arrays corresponding to:
        - conditional score  
        - censored score  
    """

    dAlpha = vParamsS
    
    # Density
    mF = np.array(randF.pdf(mY)) # f(y)
    
    # Weights
    if sSide == 'L':
        dFBar = 1 - randF.cdf(dR) # Fbar
        mW = funcWIndicatorL(mY, dR) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, -np.inf, dR)[0])**(1/dAlpha) # ||fw||_alpha
    elif sSide == 'R':
        dFBar = randF.cdf(dR) # Fbar
        mW = funcWIndicatorR(mY, dR) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, dR, np.inf)[0])**(1/dAlpha) # ||fw||_alpha
    elif sSide == 'C':
        dFBar = randF.cdf(vParamsW-dR) + (1-randF.cdf(vParamsW+dR)) # Fbar
        mW = funcWIndicatorC(mY, dR, vParamsW) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, vParamsW-dR, vParamsW+dR)[0])**(1/dAlpha) # ||fw||_alpha
    elif sSide == 'Cfixed':
        dFBar = randF.cdf(dR-vParamsW/2) + (1-randF.cdf(dR+vParamsW/2)) # Fbar
        mW = funcWIndicatorCfixed(mY, dR, vParamsW) # w(y)
        dFwNorm = (integrate.quad(lambda x:  (randF.pdf(x))**dAlpha, dR-vParamsW/2, dR+vParamsW/2)[0])**(1/dAlpha) # ||fw||_alpha
    
    # Localized scores
    mSFSharp = mW * (dAlpha * (mF/(1-dFBar))**(dAlpha-1) - (dAlpha-1) * (dFwNorm/(1-dFBar))**dAlpha)
    mSFFlat = mW * dAlpha * mF**(dAlpha-1) + (1-mW) * dAlpha * dFBar**(dAlpha-1) - (dAlpha-1) * (dFwNorm**dAlpha + dFBar**dAlpha)
    
    return [mSFSharp, mSFFlat]

###########################################################  
def funtwCRPS(mY, randF, dR, vParamsS=2, vParamsW=0, sSide='L'):
    """
    Purpose
    ----------
    Compute localized threshold-weighted CRPS for center-based weight functions 
    and return both sharp and flat score variants

    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution  
    dR :           float, threshold r  
    vParamsS :     float or array, additional parameter(s) for scoring rule  
    vParamsW :     float or array, additional parameter(s) for weight function  
    sSide :        string, side or configuration of the weight function ('L', 'R', 'C', 'Cfixed')  

    Returns
    ----------
    list, contains two arrays corresponding to:
        - conditional score  
        - censored score  
    """
    
    dDeltaZ = 0.001 # same as in MC
    dMinInf = -20
    dPlusInf = 20

    if sSide == 'C':
        dFBar = randF.cdf(vParamsW-dR) + (1-randF.cdf(vParamsW+dR)) # Fbar
        vZ = np.arange(vParamsW-dR, vParamsW+dR, dDeltaZ) 
        vWz = funcWIndicatorC(vZ, dR) # w(z) for single r
        vFz = randF.cdf(vZ) # F(z)
        vFzSharp = (np.array(randF.cdf(vZ))-np.array(randF.cdf(vParamsW-dR)))/(1-dFBar) # F(z)
        mW = funcWIndicatorC(mY, dR, vParamsW) # w(y)
    else:
        print('twCRPS and CRPSFlat coincide for left and right tail indicator')
    
   
    
    mLoss = np.zeros(mY.shape)
    for i in range(vZ.size):
        mLoss += vWz[i] * (vFz[i] - (mY <= vZ[i]))**2 * dDeltaZ
    
    mtwCRPS = -mLoss

    # The following is to maintain the same structure for all local rules
    
    mSFSharp = mtwCRPS
    mSFFlat = mtwCRPS
    
    return [mSFSharp, mSFFlat]

###########################################################  
def funCRPSLocal(mY, randF, dR, vParamsS=2, vParamsW=0, sSide='L'):
    """
    Purpose
    ----------
    Compute localized CRPS and return both sharp and flat score variants

    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution  
    dR :           float, threshold r  
    vParamsS :     float or array, additional parameter(s) for scoring rule  
    vParamsW :     float or array, additional parameter(s) for weight function  
    sSide :        string, side or configuration of the weight function ('L', 'R', 'C', 'Cfixed')  

    Returns
    ----------
    list, contains two arrays corresponding to:
        - conditional score  
        - censored score  
    """
    
    dDeltaZ = 0.001 # same as in MC
    dMinInf = -20
    dPlusInf = 20
 
    # Weights
    if sSide == 'L':
        dFBar = 1 - randF.cdf(dR) # Fbar
        vZ = np.arange(dMinInf, dR, dDeltaZ) 
        vWz = funcWIndicatorL(vZ, dR) # w(z) for single r
        vFz = randF.cdf(vZ) # F(z)
        vFzSharp = (np.array(randF.cdf(vZ)))/(1-dFBar) # F^\sharp(z)
        mW = funcWIndicatorL(mY, dR) # w(y)
    elif sSide == 'R':
        dFBar = randF.cdf(dR) # Fbar
        vZ = np.arange(dR, dPlusInf, dDeltaZ) 
        vWz = funcWIndicatorR(vZ, dR) # w(z) for single r
        vFz = randF.cdf(vZ) # F(z)
        vFzSharp = (np.array(randF.cdf(vZ))-np.array(randF.cdf(dR)))/(1-dFBar) # F^\sharp(z)
        mW = funcWIndicatorR(mY, dR) # w(y)
    elif sSide == 'C':
        dFBar = randF.cdf(vParamsW-dR) + (1-randF.cdf(vParamsW+dR)) # Fbar
        vZ = np.arange(vParamsW-dR, vParamsW+dR, dDeltaZ) 
        vWz = funcWIndicatorC(vZ, dR) # w(z) for single r
        vFz = randF.cdf(vZ) # F(z)
        vFzSharp = (np.array(randF.cdf(vZ))-np.array(randF.cdf(vParamsW-dR)))/(1-dFBar) # F(z)
        mW = funcWIndicatorC(mY, dR, vParamsW) # w(y)
        
        # Pivotal points
        dC = 0# dictW['vParamsW']
        dA1 = dC - dR
        dA2 = dC + dR
        dictCRPSSettings = vParamsS

        # Estimate gamma, based on data
        dGamma =1/2# dictPreCalc['Gamma']
        
        vZ1 = np.arange(dA1, dA2, dDeltaZ) 
        vZ2 = np.copy(vZ1)
        vfz1 = randF.pdf(vZ1).reshape(vZ1.shape) # f(z)
        vfz2 = np.copy(vfz1) # np.array(dictDistr['randDistr'].pdf(vZ2)) # f(z)
        
        ## E|Y-Y'|
        # Continuous part
        #dE1b = np.sum(np.abs(np.subtract.outer(vZ1, vZ2)) * np.outer(vfz1, vfz2)) * dDeltaZ**2 # to check, too large for some nodes
        iMatrixSizeMax = 5
        dE1=0
        iUpperLim = int(np.floor(vZ1.size/iMatrixSizeMax))
        for i in range(iUpperLim):
            dE1 += np.sum(np.abs(np.subtract.outer(vZ1[i*iMatrixSizeMax:(i+1)*iMatrixSizeMax], vZ2)) * np.outer(vfz1[i*iMatrixSizeMax:(i+1)*iMatrixSizeMax], vfz2)) * dDeltaZ**2
        dE1 += np.sum(np.abs(np.subtract.outer(vZ1[iUpperLim*iMatrixSizeMax:], vZ2)) * np.outer(vfz1[iUpperLim*iMatrixSizeMax:], vfz2)) * dDeltaZ**2
    
        # second part, at least one a_1 or a_2:
        dE1 += 2*dFBar * (np.sum((dGamma * np.abs(vZ1 - dA1) + (1-dGamma) * np.abs(vZ1 - dA2)) * vfz1) * dDeltaZ + dGamma*(1-dGamma) * dFBar * np.abs(dA1 - dA2))
        
        ## E|Y-y|
        #vZ1full = np.arange(dictCRPSSettings['dMinInf'], dictCRPSSettings['dPlusInf'], dDeltaZ)         
        def CRPSofFlat(mX):
            mEout = 0
            for i in range(vZ1.size):
                # y is not censored - continuous part of z
                mEout += np.abs(mX - vZ1[i]) * vfz1[i] * dDeltaZ 
            mEout += dFBar * (dGamma * np.abs(mX-dA1) + (1-dGamma) * np.abs(mX-dA2))
            return  1/2*dE1 - mEout 
            
        mY1 = np.copy(mY)
        mOut = np.zeros(mY.shape)
        mOut[mW > 0] =  CRPSofFlat(mY1[mW>0])
        mOut[mW == 0] = dGamma * CRPSofFlat(dA1) + (1-dGamma) * CRPSofFlat(dA2)     


    # Note: further vectorisation makes the function slower 
    mLossInt = np.zeros(mY.shape)
    for i in range(vZ.size):
        mLossInt +=  (vFzSharp[i] - (mY <= vZ[i]))**2 * dDeltaZ
    
    mLossSharp = mW * mLossInt 
    mSFSharp = -mLossSharp
    
    if sSide == 'C':
        mSFFlat = mOut
    else:
        mLossFlat = np.zeros(mY.shape)
        for i in range(vZ.size):
            mLossFlat += vWz[i] * (vFz[i] - (mY <= vZ[i]))**2 * dDeltaZ
        
        mSFFlat = -mLossFlat

    return [mSFSharp, mSFFlat]

###########################################################  
def funcSHKCorrection(mY, randF, dR, vParamsS=None, vParamsW=0, sSide='L'):
    """
    Purpose
    ----------
    Compute SHK correction terms for localized logarithmic scores 

    Parameters
    ----------
    mY :           array, observations at which the score is evaluated  
    randF :        object, selected distribution  
    dR :           float, threshold r  
    vParamsS :     float or array, additional parameter(s) for scoring rule  
    vParamsW :     float or array, additional parameter(s) for weight function  
    sSide :        string, side or configuration of the weight function ('L', 'R', 'C', 'Cfixed')  

    Returns
    ----------
    list, contains two arrays corresponding to:
        - logarithmic correction term  
        - first-order Taylor expansion of the correction  
    """
    
    # Density
    mF = np.array(randF.pdf(mY)) # f(y)
    mF[mF==0] = 1e-100 # avoid numerical zeros
    
    # Weights
    if sSide == 'L':
        dFBar = 1 - randF.cdf(dR) # Fbar
        mW = funcWIndicatorL(mY, dR) # w(y)
    elif sSide == 'R':
        dFBar = randF.cdf(dR) # Fbar
        mW = funcWIndicatorR(mY, dR) # w(y)
    elif sSide == 'C':
        dFBar = randF.cdf(vParamsW-dR) + (1-randF.cdf(vParamsW+dR)) # Fbar
        mW = funcWIndicatorC(mY, dR, vParamsW) # w(y)
    elif sSide == 'Cfixed':
        dFBar = randF.cdf(dR-vParamsW/2) + (1-randF.cdf(dR+vParamsW/2)) # Fbar
        mW = funcWIndicatorCfixed(mY, dR, vParamsW) # w(y)
    
    # Correction terms HK
    mFslog =  mW * np.log(1-dFBar) + (1-mW) * np.log(dFBar)
    mFsbar = mW * (np.log(1-dFBar) + 1) - (1-dFBar)
    
    return [mFslog, mFsbar]

