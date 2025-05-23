#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Calculate Scores based on estimated temperature-GARCH models 
"""

## Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration

# MPI
from mpi4py import MPI

# Dependencies
from ScoreBasis import *       # elementary functions
from ScoringRules import *     # scoring rules
from Weightfunctions import *  # weight functions

###########################################################  
def main():    
    
    ## Magic numbers 
    bWarm = False 
    iTest = 8*365+2                       # window length parameter estimation    
    vH = np.array([1,2,3])           # forecast horizons
    iHmax = vH.max()                      # largest forecast horizon
    iSeed = 1234
    np.random.seed(iSeed)                 # set random seed
    iPrecision = 4                        # precision LaTeX tables
    bOrderMPI = False                     # order task distribution [sequentially if true]
    iMatrixSizeMax = 50
    lRq = [1, 2, 4]                 # centre example fixed bandwidths 
    dRTarget = 18                         # median of optimal tuber range [14, 22], Struik (2009)
                         
    sVersion = 'vFinal'
    sVersionGARCH = 'clim' 
    sModels = 'TempGARCH-norm-t'
    
    iYearStart = 2003 
    iYearEnd = 2022
    iT= 7305
    
    sSettingsModels = sModels + '_iT_' + str(iT) + '_iTest_' +  str(iTest) 
    
    # Number of windows
    iNumEstWindows = iT - iTest - iHmax + 1
    
    # Methods
    dictMethods = {
        'GARCH-Normal' : {'Model': '','sDistr': 'normal', 'sName': 'QGARCHI-$\\mathcal{N}$'},
        'GARCH-Std(nu)' : {'Model': '','sDistr': 't', 'sName': 'QGARCHI-$t$'},
        'QGARCHI-I-Normal' : {'Model': '','sDistr': 'normal', 'sName': 'QGARCHI-$\\mathcal{N}$'},
        'QGARCHI-I-Std(nu)' : {'Model': '','sDistr': 't', 'sName': 'QGARCHI-$t$'},
        'QGARCHI-II-Normal' : {'Model': '','sDistr': 'normal', 'sName': 'QGARCHI-$\\mathcal{N}$'},
        'QGARCHI-II-Std(nu)' : {'Model': '','sDistr': 't', 'sName': 'QGARCHI-$t$'}
        }
    iM = len(list(dictMethods.keys()))
    
    ####################
    # Scoring rules
    ####################
    dictNumIntSettings = {'dMinInf':-100, 'dPlusInf': 100, 'dDeltaZ' : 0.001}
    dictCRPSSettings = {'dMinInf':dictNumIntSettings['dMinInf'], 'dPlusInf': dictNumIntSettings['dPlusInf'], 'dDeltaZ' : dictNumIntSettings['dDeltaZ'], 'iMatrixSizeMax':iMatrixSizeMax, 'iSeed': 2314,'CensDistParams': 0}
    
    vAlpha = np.array([2])
    dAlpha = 2 # selected alpha norm  
    dictScores = {
              'LogSSharp': {'fScore': LogSSharp,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\sharp$'},
              'LogSFlat': {'fScore': LogSFlat,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\flat$'},
              'LogSSharpslog': {'fScore': LogSSharpslog,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\sharp + $slog'},
              'LogSSharpsbar': {'fScore': LogSSharpsbar,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\sharp + $sbar'},
              'PsSphSSharp': {'fScore': PsSphSSharp,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\sharp$'}, 
              'PsSphSFlat': {'fScore': PsSphSFlat,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\flat$'},
              'PsSphSSharpslog': {'fScore': PsSphSSharpslog,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\sharp + $slog'}, 
              'PsSphSSharpsbar': {'fScore': PsSphSSharpsbar,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\sharp + $sbar'}, 
              'PowSSharp': {'fScore': PowSSharp,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\sharp$'},
              'PowSFlat': {'fScore': PowSFlat,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\flat$'},
              'PowSSharpslog': {'fScore': PowSSharpslog,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\sharp + $slog'},
              'PowSSharpsbar': {'fScore': PowSSharpsbar,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\sharp + $sbar'},
              'twCRPS':{'fScore': twCRPS,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'twCRPS'}, 
              'CRPSSharp':{'fScore': CRPSSharp,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp$'},
              'CRPSGenFlatGammaEstim':{'fScore': CRPSGenFlatGamma,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$_{\\rm {gen}}^\\flat$'},
              'CRPSSharpslog':{'fScore': CRPSSharpslog,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp + $slog'},
              'CRPSSharpsbar':{'fScore': CRPSSharpsbar,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp + $sbar'},
              }
    lScoreNames = list(dictScores.keys())
    iS = len(lScoreNames)
          
    ####################
    # Weight function
    ####################
    dictW = {'fW': fWIndicatorC, 'vR': None,  'vParamsW': dRTarget}
    
    sSettings = sSettingsModels + '_iS_' +  str(iS) + '_' + str(dictW['fW'].__name__)  + '_' + sVersion 
     
    # Load data
    lYh = []
    lParamsDFh = []
    for i in range(len(vH)):
        iH = vH[i]
        mParamsDFhSelect = np.load('mParamsDF/mParamsDF_h' + str(iH) +'_' + sSettingsModels + '_bWarm_' + str(bWarm) + '.npy')
        lParamsDFh.append(mParamsDFhSelect)
        
        # Load vectors y_i = y_{t+h}, for t= iTest - 1, ..., iT-1-iHmax, i = 0, ..., iT-1-iHmax-iTest+1 = number of estimation windows minus 1
        vYhSelect = np.load('mParamsDF/vYh' + str(iH) +'_' + sSettingsModels + '_bWarm_' + str(bWarm) + '.npy')
        lYh.append(vYhSelect)
    
    mRhat = mRhatFixed(lRq, iNumEstWindows) # for fixed bands
    mGammaHat = np.load('mParamsDF/mGammaHat_' + sSettingsModels + '_bWarm_' + str(bWarm) +  '.npy')
    
    ######################################################################    
    ## Calculate average scores
        
    ## Initialisation MPI device
    comm = MPI.COMM_WORLD
    iRank = comm.Get_rank()  # rank process
    iProc = comm.Get_size()  # total number of processes 
        
    # Each estimation window implies one forecast 
    vIntScoresByProc = MPITaskDistributor(iRank, iProc, iNumEstWindows, bOrder=bOrderMPI) # calculate indices windows assigned to this process
      
    # Calculation: density forecast parameters
    mScoresProc = RollingScoresMPI(dictMethods, dictScores, dictW, lYh, lParamsDFh, mRhat, vIntScoresByProc, dictNumIntSettings, mGammaHat)
    lScores = comm.gather(mScoresProc, root=0)
        
    comm.barrier() # wait for everyone

    ## Combine data on root 
    if iRank == 0:
        mScores = np.ones((len(vH), len(lRq), iM, iS,iNumEstWindows)) * np.nan
        for rank in range(len(lScores)):
            vIntRank = MPITaskDistributor(rank, iProc, iNumEstWindows, bOrder=bOrderMPI)
            for i in range(vIntRank.size): 
                mScores[:,:,:,:,vIntRank[i]] = lScores[rank][:,:,:,:,i]
 
        # Save scores per horizon in separate files
        for i in range(len(vH)):
            iH = vH[i]
            np.save('mScores/mScores_h' + str(iH) +'_' + sSettings + '_bWarm_' + str(bWarm) + '.npy' , mScores[i,:])
                        
###########################################################
### start main
if __name__ == "__main__":
    main()
