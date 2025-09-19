#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Helper functions power experiments
"""

###########################################################
### Imports

# Fundamentals
import numpy as np  

# Dependencies
from ScoringRulesMC  import *    

###########################################################
def MPITaskDistributor(iRank, iProc, iTotal, bOrder=True):
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
def ScoringRuleCollectionName(dictScores):
    """
    Purpose
    ----------
    Use short codes to refer to a selection of scoring rules in the file name, 
    to prevent for the error of having a too long file name.

    Parameters
    ----------
    dictScores :        dictionary, with scoring rules

    Returns
    ----------
    sSelectedScores :   abbreviated name
    """
    
    sSelectedScoresFull = ''
    for s in list(dictScores.keys()): sSelectedScoresFull += s
    
    if sSelectedScoresFull == 'LogSLogSSharpLogSFlatPsSphSPsSphSSharpPsSphSFlatPowSPowSSharpPowSFlatCRPStwCRPSwsCRPSCRPSSharpCRPSFlat':
       sSelectedScores = 'LogSPsSphSPowSCRPSall'
    elif sSelectedScoresFull == 'LogSLogSSharpLogSFlatPsSphSPsSphSSharpPsSphSFlatPowSPowSSharpPowSFlatCRPStwCRPSwsCRPSCRPSSharpCRPSFlatCRPSFlatConstCRPSFlatCentre':
       sSelectedScores = 'LogSPsSphSPowSCRPSallflats'   
    elif sSelectedScoresFull == 'LogSLogSSharpLogSFlatPsSphSPsSphSSharpPsSphSFlatPowSPowSSharpPowSFlatCRPStwCRPSwsCRPSCRPSSharpCRPSFlatMinCRPSFlatConstCRPSFlatRandom':
          sSelectedScores = 'LogSPsSphSPowSCRPSallflats'   
    else:
       sSelectedScores = sSelectedScoresFull
    
    return  sSelectedScores

###########################################################
def SaveAsHDF(mX, sNameFile, sNameObj):
    """
    Purpose
    ----------
    Save numpy array as HDF object (currently a placeholder)

    Parameters
    ----------
        mX :        array, object to be saved
        sNameFile :     string, name of file
        sNameObj:   string, name of object 
        
    Returns
    ----------
    Saves file
    """
    
    #hf = h5py.File(sNameFile + '.h5', 'w')
    #hf.create_dataset(sNameObj, data=mX)
    #hf.close()
    np.save(sNameFile +'.npy', mX)
    
###########################################################
def ReadHDF(sNameFile, sNameObj, bSave=True):
    """
    Purpose
    ----------
    Load numpy array as HDF object (currently a placeholder)

    Parameters
    ----------
    sNameFile :     string, name of file
    sNameObj:       string, name of object 

    Returns
    ----------
    Loaded object
    """

    #hf = h5py.File(sNameFile + '.h5', 'r')
    #mX = hf.get(sNameObj)
    #mX = np.array(mX)
    #hf.close()  
    mX = np.load(sNameFile +'.npy')
    return mX
    
###########################################################
def SimulateData(dictCand, dictW, vIntR, iC, iTupper, dISWeightLB, iRep, iSeed, lSettings, vTmax=False, bSimulateData=True, bSave=True, bH5=True, bSim=False):
    """
    Purpose
    ----------
    Simulate Data
    Apply sample size reduction for simulated values with weights below dISWeightLB
    Calculate Importance Sampling Weights

    Parameters
    ----------
    dictCand :      dictionary, contains dictionary for F and G distribution:
                        dictF: dictionary, distribution F candidate
                            randDistr:  distribution F, stats object
                            sDistr : name distribution
                        dictG: dictionary, distribution G candidate
                            randDistr:  distribution F, stats object
                            sDistr: name distribution
    dictW :         dictionary,  weight function: 
                            fW : selected weighted function
                            vR : vector, total threshold grid
                            vParamsW : vector, parameters of weight function  
                            vIntR :         vector, indices vR grid for which data is simulated
    iC :            integer, expected number of observations in {w>0}        
    iTupper :       integer or boolean, simulation without IS, otherwise, simulation with IS,
                    where iTupper is the upperbound of the sample size       
    dISWeightLB :   double, lowerbound, observation values below this threshold are included in IS        
    iRep :          integer, number of MC replications
    iSeed :         integer, random seed
    lSettings :     list, settings F and G candidate
    vTmax :         vector, optional, vector with maximum sample sizes per distribution
    bSimulateData : boolean, optional, True simulates new data, False loads existing results
    bSave :         boolean, optional, True saves simulated results, False returns them
    bH5 :           boolean, optional, True saves in HDF5 format, False in .npy
    bSim :          boolean, optional, flag to distinguish simulation output for naming

    Returns
    ----------
    List of length len(vIntR) with dictionaries containing for each true distribution [keys] 
    a matrix of size 2 x iT x iRep, where the simulated data is assigned to the first dimension and
    the corresponding importance sampling weights to the second dimension. Furthermore, iT = min(iTc, iTupper), where 
    iTc denotes the number of simulations required to obtain iC observations with positive weight
    """
    
    lCandKeys = list(dictCand.keys())
    vR = dictW['vR'] 
    vOneMinFbarF = 1-DistrBar(dictCand[lCandKeys[0]], dictW, vIntR)
    vOneMinFbarG = 1-DistrBar(dictCand[lCandKeys[1]], dictW, vIntR)
    if iTupper != False:
        mTmax = np.zeros((2,vIntR.size))
        for i in range(len(vIntR)):
            mTmax[:,i] =np.array([int(np.round(iC/(vOneMinFbarF[i]))), int(np.round(iC/(vOneMinFbarG[i])))])
        if(vTmax[0] != mTmax.max(axis=1)[0]): print('Error: vTmax mismatch.')
        
    #############################################
    ### SIMULATE DATA 
    if bSimulateData:
        
        ## With Importance Sampling [r horizontal axis]
        if iTupper != False:
            vPermIdx = np.random.permutation(np.arange(0,int(vTmax.max())))
            dictDataAndWeights = {}
            for d in range(len(lCandKeys)):
                mDataAndWeights = np.zeros((2, vIntR.size, iTupper, iRep)) * np.nan
                np.random.seed(int(iSeed+bSim*iSeed))
                
                for rep in range(iRep): # no vectorisation due to memory limitations due to large sample sizes
                    vData = np.array(dictCand[lCandKeys[d]]['randDistr'].rvs(int(vTmax[d])))
                    for i in range(len(vIntR)):
                        iTc = int(mTmax[d,i])
                        vDataR = vData[0:iTc]
                        if iTc > iTupper:
                            vIdxISWeightOne = dictW['fW'](vDataR, dictW['vParamsW'], vR[i]) > dISWeightLB
                            vDataISWeightOne = vDataR[vIdxISWeightOne]
                            iTotWeightOne = vDataISWeightOne.size
                            vDataISWeightOneC = vDataR[np.invert(vIdxISWeightOne)]
                            vPermIdxR = vPermIdx[vPermIdx<int(min(iTupper-iTotWeightOne,vDataISWeightOneC.size))]
                            vDataISWeightOneC_Selected = vDataISWeightOneC[vPermIdxR]
                            iTotOneC_Selected =  vDataISWeightOneC_Selected.size
                            iTot = iTotWeightOne + iTotOneC_Selected
                            mDataAndWeights[0,i,0:iTot,rep] = np.concatenate((vDataISWeightOne, vDataISWeightOneC_Selected))
                  
                            # Observations with weight one get IS weight iTotWeightOne/iTc * 1/iTotWeightOne
                            # Other: iTotOneC_Selected/iTc * 1/iTotOneC_Selected
                            mDataAndWeights[1,i,0:iTot,rep] = np.concatenate((np.repeat(1/iTc,iTotWeightOne),np.repeat(vDataISWeightOneC.size/iTc*(1/iTotOneC_Selected),iTotOneC_Selected)))              
        
                        else:
                            mDataAndWeights[0,i,0:iTc,rep] = vData[0:iTc]
                            mDataAndWeights[1,i,0:iTc,rep] = np.ones((1,iTc))/iTc
                # Save results
                if bSave:
                    if bSim:
                        sNameFile = 'mDataAndWeights/mDataAndWeightsSim_' +  lSettings[d] + '_fW_' + dictW['fW'].__name__
                    else:
                        sNameFile = 'mDataAndWeights/mDataAndWeights_' +  lSettings[d] + '_fW_' + dictW['fW'].__name__
                    if bH5:
                        for r in range(len(vIntR)):
                            SaveAsHDF(mDataAndWeights[:,r,:], sNameFile + '_iRidx_'+str(r), 'DataAndWeights' + str(d) + str(r))
                    else:
                        np.save(sNameFile + '.npy', mDataAndWeights)
                        
                    
                else:
                    dictDataAndWeights[lCandKeys[d]] = mDataAndWeights
            
            np.save('mDataAndWeights/mTmax_' +  lSettings[0] + '_fW_' + dictW['fW'].__name__ , mTmax) 
            if not bSave:
                if iTupper == False: mTmax = vTmax
                print('Simulation completed.')
                return dictDataAndWeights, mTmax
                    
        ## Without Importance Sampling [r or c horizontal axis]
        else:
                np.random.seed(iSeed)
                mDataAndWeightsF = dictCand[lCandKeys[0]]['randDistr'].rvs((int(vTmax[0]),iRep)) 
                np.random.seed(iSeed)
                mDataAndWeightsG = dictCand[lCandKeys[1]]['randDistr'].rvs((int(vTmax[1]),iRep)) 
                # Save results
                if bSave: 
                    for d in range(len(lCandKeys)):
                        np.save('mDataAndWeights/mDataAndWeights_' +  lSettings[d] + '_fW_' + dictW['fW'].__name__ +  '.npy' , mDataAndWeights)
                else:
                    dictDataAndWeights = {lCandKeys[0]: mDataAndWeightsF, lCandKeys[1]: mDataAndWeightsG}
                    if iTupper == False: mTmax = vTmax
                    print('Simulation completed.')
                    return dictDataAndWeights, mTmax
    
    #############################################
    ### LOAD DATA     
    else:
        mDataAndWeightsF = np.load('mDataAndWeights/mDataAndWeights_' +  lSettings[0] + '_fW_' + dictW['fW'].__name__ +  '.npy')
        mDataAndWeightsG = np.load('mDataAndWeights/mDataAndWeights_' +  lSettings[1] + '_fW_' + dictW['fW'].__name__ +  '.npy')
        dictDataAndWeights = {lCandKeys[0]: mDataAndWeightsF, lCandKeys[1]: mDataAndWeightsG}
        if iTupper == False: mTmax = vTmax
        
        return dictDataAndWeights, mTmax

###########################################################  
def DMCalc(dictCand, mY, fScore, iRidx, dictW, vParamsS, dictPreCalc):
    """
    Purpose
    ----------
    Calculate Diebold Mariano Statistics

    Parameters
    ----------
    dictCand :  dictionary, contains dictionary for F and G distribution:
                    dictF: dictionary, distribution F candidate
                        randDistr:  distribution F, stats object
                        sDistr : name distribution
                    dictG: dictionary, distribution G candidate
                        randDistr:  distribution F, stats object
                        sDistr: name distribution        
    mY :            matrix, iT x iRep, observations for which the scoring rule will be calculated
    fScore :        function, selected scoring rule
    dictW :         dictionary,  weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    iRidx :         integer, index r grid                              
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
        
    Returns
    -------
    vDMStat :              vector, DM statistics for each value of r [horizontal axis power plot]
    """
    
    mDiff = fScore(dictCand['dictF'], mY, iRidx, dictW, vParamsS, dictPreCalc) - fScore(dictCand['dictG'], mY, iRidx, dictW, vParamsS, dictPreCalc)
    if mDiff.ndim == 1: mDiff = mDiff.reshape((mDiff.size, 1))  # allow for vectors
    iT, iRep = mDiff.shape
    
    return mDiff.mean(axis=0)/np.sqrt(mDiff.var(axis=0)/iT)

###########################################################  
def DMCalcWithIS(dictCand, mYandISWeights, iT, fScore, iRidx, dictW, vParamsS, dictPreCalc):
    """
    Purpose
    ----------
    Calculate Diebold Mariano Statistics with IS

    Parameters
    ----------
    dictCand :      dictionary, contains dictionary for F and G distribution:
                        dictF: dictionary, distribution F candidate
                            randDistr:  distribution F, stats object
                            sDistr : name distribution
                        dictG: dictionary, distribution G candidate
                            randDistr:  distribution F, stats object
                            sDistr: name distribution        
    mYandISWeights : array, 2,iT x iRep, observations for which the scoring rule will be calculated
                    note: includes NaNs if IS is applied
    iRidx :         integer, index r grid                
    fScore :        function, selected scoring rule
    dictW :         dictionary,  weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    vParamsS :      vector, parameter vector scoring rule: alpha parameter
    dictPreCalc :   dictionary, precalculated values
        
    Returns
    -------
    vDMStat :              vector, DM statistics for each value of r [horizontal axis power plot]
    """
    
    mY = mYandISWeights[0,][~np.isnan(mYandISWeights[0,]).any(axis=1)]
    mISWeights = mYandISWeights[1,][~np.isnan(mYandISWeights[1,]).any(axis=1)] 

    mDiff = fScore(dictCand['dictF'], mY, iRidx, dictW, vParamsS, dictPreCalc) - fScore(dictCand['dictG'], mY, iRidx, dictW, vParamsS, dictPreCalc)
       
    mDiffWeighted = mISWeights * mDiff
    vDiffMean = mDiffWeighted.sum(axis=0)
    mDiffSquaredWeighted = mISWeights * mDiff**2
    vDiffVar = mDiffSquaredWeighted.sum(axis=0) - vDiffMean**2 # not HAC
    
    # Note: Elements of the variance vector can be numerically zero, e.g. when using indicator function
    # and in a particular simultation, there are zero observations in the region of interest.
    # Another example is that the scores are exactly the same, in which case the score differences are zero.
    vDiffVar[vDiffVar < 10**-25] = 10**-25
    
    return vDiffMean / np.sqrt(vDiffVar/iT) 

###########################################################
def DMTestMPI(dictCand, dictScores, dictW, lSettings, dictPreCalc, iC, iTupper, iRep, iRank, vInt, vC, dLeftTailPerc):
    """
    Purpose
    ----------
    Calculate DM statistics for replications with indices in vIntRep [without expected number of observations correction]
    
    Parameters
    ----------
    dictCand :      dictionary, contains dictionary for F and G distribution:
        dictF:      dictionary, distribution F candidate
                        randDistr: distribution F, stats object
                        sDistr: name distribution
        dictG:      dictionary, distribution G candidate
                        randDistr: distribution F, stats object
                        sDistr: name distribution
    dictScores :    dictionary, keys are scoring rule names: contains dictionary for each scoring rule with keys
                        fScore: function Scoring rule 
                        vParamsS: parameter vector scoring rule
                        bWeighted: boolean, weighted scoring rule logical
    dictW :         dictionary, weight function: 
                        fW : selected weighted function
                        dictParamsW : dictionary, parameters of weight function
    lSettings :     list, settings F and G candidate
    dictPreCalc :   dictionary, precalculated values
    iC :            integer, expected number of observations {w>0}
    iTupper :       integer or boolean, sample size upper bound for IS or False if no IS
    iRep :          integer, number of MC replications
    iRank :         integer, index of current MPI process
    vInt :          vector, selected indices of threshold grid for process with rank iRank
    vC :            vector, number of observations per threshold index for c-grid
    dLeftTailPerc : double, proportion used to compute truncation point for c-grid
    
    Returns
    ----------
    mDMCalc : matrix, size 2 x iS x iRep x len(vIntR) where the `2' refers to rejection in favour of f or g
    """

    lTrueKeys = list(dictCand.keys())
    lTrueKeysReverse = lTrueKeys.copy()
    lTrueKeysReverse.reverse()
    vR= dictW['vR']
    lScoresNames = list(dictScores.keys())
    mTmax = np.load('mDataAndWeights/mTmax_' +  lSettings[0] + '_fW_' + dictW['fW'].__name__ + '.npy') 
    
    iS = len(lScoresNames) # number of selected scoring rules
    lCandKeys = list(dictCand.keys())
    lCandKeysReverse = lCandKeys.copy()
    lCandKeysReverse.reverse()
    
    iInt = vInt.size
    mDMCalc = np.zeros((2, iS, iRep, iInt))  # dgp x scoring rules x rep x rgrid rank
   
    for i in range(iInt):
        if iRank == 0: print(round(i/iInt*100), '% completed', sep='') # status printed on first node
        
        dictDataAndWeightsR = {lTrueKeys[0]: ReadHDF('mDataAndWeights/mDataAndWeights_' +  lSettings[0] + '_fW_' + dictW['fW'].__name__ + '_iRidx_'+str(vInt[i]), 'DataAndWeights' + str(0) + str(vInt[i])),
                              lTrueKeys[1]: ReadHDF('mDataAndWeights/mDataAndWeights_' +  lSettings[1] + '_fW_' + dictW['fW'].__name__ + '_iRidx_'+str(vInt[i]), 'DataAndWeights' + str(1) + str(vInt[i]))}
        for d in range(2):
            #print((d,dictDataAndWeights[lTrueKeys[d]][:,vIntR[i],0:5,0:5]))
            for s in range(iS):   
                if iTupper != False and vC.sum() == 0: # r grid with IS
                    if dictScores[lScoresNames[s]]['bWeighted'] or vR[vInt[i]] == vR[-1]:
                        
                        # DM test
                        mDMCalc[d,s,:,i] = DMCalcWithIS(dictCand, dictDataAndWeightsR[lTrueKeys[d]], int(mTmax[d,vInt[i]]), dictScores[lScoresNames[s]]['fScore'], vInt[i], dictW, dictScores[lScoresNames[s]]['vParamsS'], dictPreCalc)  
                        
                    else: 
                        continue
                elif iTupper == False and vC.sum() == 0: # r grid without IS
                    iTc = int(np.round(iC/dictCand[lCandKeys[d]]['randDistr'].cdf(vR[vInt[i]])))
                    iT = iTc *  int(dictScores[lScoresNames[s]]['bWeighted']) + iC * int((1 - dictScores[lScoresNames[s]]['bWeighted']))
                    
                    # DM test
                    mDMCalc[d,s,:,i] = DMCalc(dictCand, dictDataAndWeightsR[lTrueKeys[d]][0:iT,:], dictScores[lScoresNames[s]]['fScore'], vInt[i], dictW, dictScores[lScoresNames[s]]['vParamsS'], dictPreCalc)
                    
                else: # c grid:
                    iT = int(vC[vInt[i]]/dLeftTailPerc * int(dictScores[lScoresNames[s]]['bWeighted']) + vC[vInt[i]] * int((1 - dictScores[lScoresNames[s]]['bWeighted'])))
                    
                    # DM test
                    mDMCalc[d,s,:,i] = DMCalc(dictCand, dictDataAndWeightsR[lTrueKeys[d]][0:iT,:], dictScores[lScoresNames[s]]['fScore'], d, dictW, dictScores[lScoresNames[s]]['vParamsS'], dictPreCalc)
        
    return  mDMCalc[0,:], mDMCalc[1,:]

###########################################################
def RejRates(mDMCalc, dictCritVal, bPower=True):
    """
    Purpose
    ----------
    Calculate Rejection Rates

    Parameters
    ----------
    mDMCalc :            matrix, size iF x iS x iRep x iLenPiGrid
    dictCritVal :        dictionary, keys: 
                                FavF:  H_1: E d > 0 [rej in favour of F]
                                FavG:  H_1 E d < 0 [rej in favour of G]
    Returns
    ----------
    dictRejRates :       dictionary, with per keySide
                               mRejRates : matrix, size iF x iS x iLenPiGrid
    """
    
    iTrues, iS, iRep, iLenRGrid = mDMCalc.shape
    if bPower: # power study case
        dictRejRates = {'TrueF': {'FavF': None, 'FavG': None}, 'TrueG': {'FavF': None, 'FavG': None}}
    else:
        dictRejRates = {'TrueP': {'FavF': None, 'FavG': None}}
    lTrueKeys = list(dictRejRates.keys())
    
    for keySide in list(dictCritVal.keys()): 
        for i in range(len(lTrueKeys)):
            if keySide == 'FavF':
                dictRejRates[lTrueKeys[i]][keySide] = []
                for k in range(len(dictCritVal[keySide])):
                    dictRejRates[lTrueKeys[i]][keySide].append(np.mean(mDMCalc[i,:,:,:] > dictCritVal[keySide][k], axis=1))
            elif keySide == 'FavG':
                dictRejRates[lTrueKeys[i]][keySide] = []
                for k in range(len(dictCritVal[keySide])):
                    dictRejRates[lTrueKeys[i]][keySide].append(np.mean(mDMCalc[i,:,:,:] < dictCritVal[keySide][k], axis=1))
    
    return  dictRejRates

###########################################################
def iTMaxCalc(dictCand, dictW, iC):
    """
    Purpose
    ----------
    Calculate iTMax

    Parameters
    ----------
    dictCand :  dictionary, contains dictionary for F and G distribution:
                    dictF: dictionary, distribution F candidate
                            randDistr:  distribution F, stats object
                            sDistr : name distribution
                    dictG: dictionary, distribution G candidate
                            randDistr:  distribution F, stats object
                            sDistr: name distribution
    dictW :     dictionary,  weight function: 
                    fW : selected weighted function
                    vR : vector, threshold grid
                    vParamsW : vector, parameters of weight function     
    iC :        integer, expected number of observations in {w>0}                   
        
                
    Returns
    ----------
    integer, iTmax = iC / smallest tail probability
    """

    vR = dictW['vR']
    lCandKeys = list(dictCand.keys())
    lProbW0 = []
    for d in range(2):
        lProbW0.append(dictCand[lCandKeys[d]]['randDistr'].cdf(vR.min()))
  
    return int(np.ceil(iC/np.min(lProbW0)))  
