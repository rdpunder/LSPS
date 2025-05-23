#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Risk management application: Construct density forecasts and calculate scores.
"""

## Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration

# MPI
from mpi4py import MPI

# Pandas
import pandas as pd

# Dependencies
from ScoreBasis import *         # elementary functions
from ScoringRules import *       # scoring rules
from Weightfunctions import *    # weight functions

###########################################################  
def main():    
    
    ## Magic numbers
    bTest = False
    dRq = 0.25
    iTest = 1000                            # window length parameter estimation
    
    vH = np.array([1, 5])                   # forecast horizons
    lQVaR = [0.01, 0.05, 0.10]              # quantile VaR evaluation
    lRq = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25] 
    lH = [1,5]                              # forecast horizons
    iSeed = 1234
    np.random.seed(iSeed)                   # set random seed
  
    bOrderMPI = False                       # order task distribution [sequentially if true]
    iMatrixSizeMax = 50

    bRollingScores = False
    bVaR = False
    bDM = False
    
    # Estimation
    iP = 0 # AR order 
    bQML=False
    bQMLFull = False 
    bImpStat = False # Imposes stationarity when true
    bPackage = False # TGARCH can be compared with package by Kevind Sheppard
    
    # Plotting
    bPlotVaR = False
    bPlotVol= False
    bPlotRollingScores = False
    bPlotHits = False
    
    sVersion = 'vFinal'
    sVersionGARCH = 'vTGFinal' 
    sModels = 'TGARCH-RGARCH-norm-t'
    
    #### Data ####
    dfData = pd.read_csv('Data/SP500andRealVol1995Xiu.csv', sep=',')
    dfData.index = dfData['Date']
    lDataIndex = dfData.index
    vY = dfData['Adj Close'].values  * 100
    vRV= dfData['rv5'].values * 100**2 # better scale for numerical optimisation
    iHmax = vH.max()
    if bTest:
        iTest = 1000 + 1000                            # window length parameter estimation
        iTw = 10
        iTmax = iTest + iTw + iHmax - 1 + 4 
        iStart=1000
        vY = dfData['Adj Close'].values[iStart:iTmax+iStart]  * 100
        vRV= dfData['rv5'].values[iStart:iTmax+iStart] * 100**2 # better scale for numerical optimisation
    iT = len(vY)
    iTw = 0                                 # window length optimisation weights: unused
       
    # Number of windows
    iNumEstWindows = iT - iTest - iHmax + 1
    iNumDFWindows = iT - iTest -iTw - iHmax + 1 
    print('Number of esitmation windows:',iNumEstWindows )
    print('Number of density forecast windows:',iNumDFWindows )
    
    ####################
    # Methods
    ####################

    # Note: #windows_estim = iT - iTest - iHmax + 1  and #windows_estim = iT - iTest - iHmax + 1 - iT_w,
    # equivalent to Opschoor et al. (2017)
    
    dictMethods = {
        'GARCH-Normal' : {'Model': '', 'iP': iP, 'sDistr': 'Normal', 'sName': 'GARCH-$\\mathcal{N}$'}, 
        'GARCH-Std(nu)':  {'Model': '', 'iP': iP, 'sDistr': 'Student-t', 'sName': 'GARCH-$t(\\nu)$'},
        'TGARCH-Normal' : {'Model': '', 'iP': iP,'sDistr': 'Normal', 'sName': 'TGARCH-$\\mathcal{N}$'}, 
        'TGARCH-Std(nu)':  {'Model': '','iP': iP, 'sDistr': 'Student-t', 'sName': 'TGARCH-$t(\\nu)$'},
        'RGARCH-Normal':  {'Model': '','iP': iP, 'sDistr': 'Normal', 'sName': 'RGARCH-$\\mathcal{N}$'},
       'RGARCH-Std(nu)':  {'Model': '','iP': iP, 'sDistr': 'Student-t', 'sName': 'RGARCH-$t(\\nu)$'},
        }
    iM = len(list(dictMethods.keys())) 

    ####################
    # Scoring rules
    ####################
    dictCRPSSettings = {'dMinInf':-20, 'dPlusInf': 20, 'dDeltaZ' : 0.001, 'iMatrixSizeMax':iMatrixSizeMax, 'iSeed': 2314,'CensDistParams': 0}
   
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
              'twCRPS':{'fScore': twCRPS,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'twCRPS'}, #equiv to flat 
              'CRPSSharp':{'fScore': CRPSSharp,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp$'},
              'CRPSSharpslog':{'fScore': CRPSSharpslog,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp + $slog'},
              'CRPSSharpsbar':{'fScore': CRPSSharpsbar,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp + $sbar'},
              }
    lScoreNames = list(dictScores.keys())
    iS = len(lScoreNames)
    dictW = {'fW': fWIndicatorL, 'vR': None,  'vParamsW': 0} #vR will be filled  for different values of q
    sSettingsModels = sModels + '_iT_' + str(iT) + '_iTest_' +  str(iTest) + '_' + sVersionGARCH +'_iP_'+ str(iP) + '_p_' + str(int(bPackage)) + '_qml_' + str(int(bQML+bQMLFull)) 
    sSettings = sSettingsModels + '_iTw_' +  str(iTw) + '_dRq_' +  str(int(dRq*100)) + '_iS_' +  str(iS) +'_' + sVersion 
    
    ## Load calculated h-step ahead forecasts        
        
    # Create vectors y_i = y_{t+h}, for t= iTest - 1, ..., iT-1-iHmax, i = 0, ..., iT-1-iHmax-iTest+1 = number of estimation windows minus 1
    vYh1 = vY[iTest - 1 + 1: iT-iHmax + 1] 
    vYh5 = vY[iTest - 1 + 5: iT-iHmax + 5]
    lDataIndexh1 = lDataIndex[iTest - 1 + 1: iT-iHmax + 1] 
    lDataIndexh5 = lDataIndex[iTest - 1 + 5: iT-iHmax + 5]
    dfYh1 = pd.DataFrame(data=vYh1, index=lDataIndexh1)
    dfYh5 = pd.DataFrame(data=vYh5, index=lDataIndexh5)
    pd.DataFrame(data=vYh1, index=lDataIndexh1)

    # iM x 3 [mu, sig2, nu] x iNumEstWindows
    mParamsDFh1 = np.load('mParamsDF/mParamsDF_h1_' + sSettingsModels + '.npy')
    mParamsDFh5 = np.load('mParamsDF/mParamsDF_h5_' + sSettingsModels + '.npy')
    mRhat = np.load('mParamsDF/mRhat_' + sSettingsModels + '.npy')

    ######################################################################    
    # Calculate scores
        
    ## Initialisation MPI device
    comm = MPI.COMM_WORLD
    iRank = comm.Get_rank()  # rank process
    iProc = comm.Get_size()  # total number of processes 
        
    vIntScoresByProc = MPITaskDistributor(iRank, iProc, iNumEstWindows, bOrder=bOrderMPI) # calculate indices windows assigned to this process
    print((iRank,vIntScoresByProc))
      
    # Calculation: density forecast parameters
    mScoresProc = RollingScoresMPI(dictMethods, dictScores, dictW, [vYh1, vYh5], [mParamsDFh1, mParamsDFh5], mRhat, vIntScoresByProc)
    lScores = comm.gather(mScoresProc, root=0)
        
    comm.barrier() # wait for everyone

    ## Combine data on root 
    if iRank == 0:
            
        mScores = np.ones((len(vH), len(lRq), iM, iS,iNumEstWindows)) * np.nan
        for rank in range(len(lScores)):
            vIntRank = MPITaskDistributor(rank, iProc, iNumEstWindows, bOrder=bOrderMPI)
            for i in range(vIntRank.size): 
                mScores[:,:,:,:,vIntRank[i]] = lScores[rank][:,:,:,:,i]

        # Save h=1 and h=5 in separate files        
        np.save('mScores/mScores_h1_' + sSettings + '.npy' , mScores[0,:]) 
        np.save('mScores/mScores_h5_' + sSettings + '.npy' , mScores[1,:])
   
###########################################################
### start main
if __name__ == "__main__":
    main()
