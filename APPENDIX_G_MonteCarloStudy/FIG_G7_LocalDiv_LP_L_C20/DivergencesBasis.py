#### Imports

# Fundamentals
import numpy as np  
import pandas as pd
from scipy import stats # pre-programmed random variables

# Dependencies
from ScoringRulesLocalDiv import *

###########################################################
def funcDistributionSelection(sCandidates, bSwitch=False):
    """
    Purpose
    ----------
    Select and construct two distributions F and G based on the provided candidate setting
    
    Parameters
    ----------
    sCandidates :    string, identifier for the candidate pair of distributions to be used
                     options include 'NormT5', 'LP10LP11', 'NormNorm'
    bSwitch :        boolean, if True, return distributions in reversed order: G, F instead of F, G                
    
    Returns
    ----------
    randF :          distribution object, distribution F for use in scoring or testing procedures
    randG :          distribution object, distribution G for use in scoring or testing procedures
    """

    if sCandidates == 'NormT5':
        randF = stats.norm(0,1)
        dNuG = 5
        randG = stats.t(dNuG,loc=0, scale=1/np.sqrt(dNuG/(dNuG-2)))
    elif sCandidates == 'LP10LP11':
        randF = stats.laplace(-1,1)
        randG = stats.laplace(1,1.1)
    elif sCandidates == 'NormNorm':
        randF = stats.norm(-0.2,1)
        randG = stats.norm(0.2,1)
    else:
        print('Distribution not implemented')
        return None
    
    if bSwitch:
        return randG, randF    
    else:
        return randF, randG        
    
###########################################################
def funcWeightFuncDict(lSides, iRTotal):
    """
    Purpose
    ----------
    Construct dictionary with weight function settings for each side in lSides
    
    Parameters
    ----------
    lSides :       list of strings, sides for which weight functions must be constructed
                   options include 'L', 'R', 'C', 'Cfixed'
    iRTotal :      integer, number of grid points for threshold vector vR
    
    Returns
    ----------
    dictWeightFuncDict :    dictionary, for each side: weight function, parameter value, and threshold vector
    """

    iRMax = 4 # for all sides
    dictWeightFuncDict = {}
    for side in lSides:
        iRMin = -4 + 4.1*(int(side == 'C') + int(side == 'Cfixed')) # minimum r
        dictWeightFuncDict[side] = {'vR': np.linspace(iRMin,iRMax, iRTotal), 'vParamsW': int(side == 'Cfixed')}
        if side == 'L':
            dictWeightFuncDict[side]['vParamsW'] = None
            dictWeightFuncDict[side]['funcFunction'] = funcWIndicatorL
        elif side == 'R':
            dictWeightFuncDict[side]['funcFunction'] = funcWIndicatorR
        elif side == 'C':
            dictWeightFuncDict[side]['funcFunction'] = funcWIndicatorC
        elif side == 'Cfixed':
            dictWeightFuncDict[side]['funcFunction'] = funcWIndicatorCfixed    
        
    return dictWeightFuncDict
    
###########################################################  
def funcMomentsScoreDiff(randF, randG, vInt, lFuncScoringRulesLocal, vYGrid, vParamsS=None, dictWeightFunction=None, dNumInf=10e3, dDeltaY=1e-3):
    """
    Purpose
    ----------
    Calculate matrix of moment differences in scoring rules between distributions F and G
    
    Parameters
    ----------
    randF :                  distribution object, distribution F
    randG :                  distribution object, distribution G
    vInt :                   vector of integers, indices of threshold values to evaluate
    lFuncScoringRulesLocal : list of functions, local scoring rules to be applied
    vYGrid :                 vector, evaluation grid over support of distributions
    vParamsS :               optional, vector, parameter(s) for scoring rules
    dictWeightFunction :     dictionary, weight function specifications for each side
    dNumInf :                float, upper bound for numerical approximation 
    dDeltaY :                float, grid step size for numerical integration 
    
    Returns
    ----------
    mMomentsProc :           4-dimensional array, moment differences for scoring rule variants across
                             sides, scoring rules, thresholds, and moment types
    """

    mMomentsProc = np.ones((len(dictWeightFunction),len(lFuncScoringRulesLocal),len(vInt),10))* np.nan # matrix to store moments
    vF = np.array(randF.pdf(vYGrid)) # f(y)   
    
    for w in range(len(dictWeightFunction)):
        sSide = list(dictWeightFunction.keys())[w]
        vRProc = dictWeightFunction[sSide]['vR'][vInt]
        vParamsW = dictWeightFunction[sSide]['vParamsW']
        for r in range(len(vInt)):
            dR = vRProc[r]
            lSCorrectionF = funcSHKCorrection(vYGrid, randF, dR, vParamsS, vParamsW, sSide)
            lSCorrectionG = funcSHKCorrection(vYGrid, randG, dR, vParamsS, vParamsW, sSide)
            vSslogDiff = lSCorrectionF[0] - lSCorrectionG[0]
            vSsbarDiff = lSCorrectionF[1] - lSCorrectionG[1]
            
            for s in range(len(lFuncScoringRulesLocal)):
                funcS = lFuncScoringRulesLocal[s]

                lSF = funcS(vYGrid, randF, dR, vParamsS, vParamsW, sSide)
                lSG = funcS(vYGrid, randG, dR, vParamsS, vParamsW, sSide)
                vSSharpDiff = lSF[0] - lSG[0]
                vSFlatDiff = lSF[1] - lSG[1]
                
                # E_F[D_S] 
                mMomentsProc[w,s,r,0] = np.sum(vSFlatDiff * vF) * dDeltaY 
                mMomentsProc[w,s,r,1] = np.sum(vSSharpDiff * vF) * dDeltaY 
                
                # E_F[D_S^2] 
                mMomentsProc[w,s,r,4] = np.sum(vSFlatDiff**2 * vF) * dDeltaY 
                mMomentsProc[w,s,r,5] = np.sum(vSSharpDiff**2 * vF) * dDeltaY 

                # E_F[D_S * D_s]       
                mMomentsProc[w,s,r,8] = np.sum(vSSharpDiff * vSslogDiff * vF) * dDeltaY
                mMomentsProc[w,s,r,9] = np.sum(vSSharpDiff * vSsbarDiff * vF) * dDeltaY
            
            # Correction terms [same for all S]    
            mMomentsProc[w,:,r,2] = np.sum(vSslogDiff * vF) * dDeltaY 
            mMomentsProc[w,:,r,3] = np.sum(vSsbarDiff * vF) * dDeltaY 
            mMomentsProc[w,:,r,6] = np.sum(vSslogDiff**2 * vF) * dDeltaY
            mMomentsProc[w,:,r,7] = np.sum(vSsbarDiff**2 * vF) * dDeltaY
                  
    return mMomentsProc

###########################################################    
def funcDataFrameMomentsSDiff(mMomentsCalcS, sScoringRule, dictWeightFunction, sSide, sCandidates, sVersion):
    """
    Purpose
    ----------
    Construct output dataframe with score divergences, variances, covariances and standardised divergences 
    for different score components, and export to Excel
    
    Parameters
    ----------
    mMomentsCalcS :         matrix, calculated moments and covariances for score components
    sScoringRule :          string, name of scoring rule used
    dictWeightFunction :    dictionary, contains weight function settings per side
    sSide :                 string, side of weight function ('L' or 'R')
    sCandidates :           string, name of candidate distributions
    sVersion :              string, version label for file name
    
    Returns
    ----------
    Saves dataframe to Excel file in OutputDataFrames directory
    """
    
    vR = dictWeightFunction[sSide]['vR']
    vParamsW = dictWeightFunction[sSide]['vParamsW']

    dfOut = pd.DataFrame(columns=['r'])
    dfColumns = ['r', 'E[DSFlat]', 'E[DSSharp]', 'E[Dslog]', 'E[Dsbar]', 'E[DSSharpslog]', 'E[DSSharpsbar]', 
                 'V[DSFlat]', 'V[DSSharp]', 'V[Dslog]', 'V[Dsbar]', 'V[DSSharpslog]', 'V[DSSharpsbar]', 'C[DSSharp, Dslog]', 'C[DSSharp, Dsbar]','XiS', 'XiS_E', 'XiS_V']
    
    dfOut['r'] = vR 
    dfOut['E[DSFlat]'] = mMomentsCalcS[:,0]
    dfOut['E[DSSharp]'] = mMomentsCalcS[:,1]
    dfOut['E[Dslog]'] = mMomentsCalcS[:,2]
    dfOut['E[Dsbar]'] = mMomentsCalcS[:,3]
    dfOut['E[DSSharpslog]'] = mMomentsCalcS[:,1] + mMomentsCalcS[:,2]
    dfOut['E[DSSharpsbar]'] = mMomentsCalcS[:,1] + mMomentsCalcS[:,3]
    
    dfOut['V[DSFlat]'] = mMomentsCalcS[:,4] - mMomentsCalcS[:,0]**2
    dfOut['V[DSSharp]'] = mMomentsCalcS[:,5] - mMomentsCalcS[:,1]**2
    dfOut['V[Dslog]'] = mMomentsCalcS[:,6] - mMomentsCalcS[:,2]**2
    dfOut['V[Dsbar]'] = mMomentsCalcS[:,7] - mMomentsCalcS[:,3]**2
    dfOut['C[DSSharp, Dslog]'] = mMomentsCalcS[:,8] - mMomentsCalcS[:,1]*mMomentsCalcS[:,2]
    dfOut['C[DSSharp, Dsbar]'] = mMomentsCalcS[:,9] - mMomentsCalcS[:,1]*mMomentsCalcS[:,3]
    dfOut['V[DSSharpslog]'] = dfOut['V[DSSharp]'] + dfOut['V[Dslog]'] + 2*dfOut['C[DSSharp, Dslog]']
    dfOut['V[DSSharpsbar]'] = dfOut['V[DSSharp]'] + dfOut['V[Dsbar]'] + 2*dfOut['C[DSSharp, Dsbar]']
    
    dfOut['StandDivSFlat'] = dfOut['E[DSFlat]'] / np.sqrt(dfOut['V[DSFlat]'])
    dfOut['StandDivSSharp'] = dfOut['E[DSSharp]'] / np.sqrt(dfOut['V[DSSharp]'])
    dfOut['StandDivSSharpslog'] = dfOut['E[DSSharpslog]'] / np.sqrt(dfOut['V[DSSharpslog]'])
    dfOut['StandDivSSharpsbar'] = dfOut['E[DSSharpsbar]'] / np.sqrt(dfOut['V[DSSharpsbar]'])
    
    dfOut['XiSslog_E'] = dfOut['E[DSSharp]'] / dfOut['E[DSSharpslog]'] 
    dfOut['XiSsbar_E'] = dfOut['E[DSSharp]'] / dfOut['E[DSSharpsbar]']
    
    dfOut['XiSslog_V'] = np.sqrt(dfOut['V[DSSharp]'] / dfOut['V[DSSharpslog]'])
    dfOut['XiSsbar_V'] = np.sqrt(dfOut['V[DSSharp]'] / dfOut['V[DSSharpsbar]'])
    
    dfOut['XiSslog'] = dfOut['StandDivSSharp'] / dfOut['StandDivSSharpslog'] 
    dfOut['XiSsbar'] = dfOut['StandDivSSharp'] / dfOut['StandDivSSharpsbar']   
    
    sFileName = 'OutputDataFrames/mMomentsSDiff_RTot'+str(len(vR))+'_'+sScoringRule+'_'+str(sSide)+str(vParamsW)+'_'+sCandidates+'_'+sVersion+'.xlsx'
    dfOut.to_excel(sFileName, index=False)

###########################################################
def MPITaskDistributor(iRank, iProc, iTotal, bOrder= True):
    """
    Purpose
    ----------
    Construct vector of indices [integers] for which process with rank iRank should do calculations

    Parameters
    ----------
    iRank :     integer, rank running process
    iProc :     integer, total number of processes in MPI program
    iTotal :    integer, total number of tasks
    bOrder :    boolean, use standard ordering of integers if True

    Returns
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