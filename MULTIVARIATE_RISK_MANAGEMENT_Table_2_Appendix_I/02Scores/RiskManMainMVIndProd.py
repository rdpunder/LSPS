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
from RiskManBasisMV import *        # elementary functions
from ScoringRulesMV import *        # scoring rules
from WeightfunctionsMV import *     # weight functions
from BivariateT import *            # bivariate Student-t

###########################################################  
def main():    
    
    ## Magic numbers
    bTest = False
    dRq = 0.25
    iTest = 1000                            # window length parameter estimation
    
    vH = np.array([1, 5])                   # forecast horizons
    lRq = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25] 
    lH = [1,5]                              # forecast horizons
    bBivariate = True
    iSeed = 1234
    np.random.seed(iSeed)                   # set random seed
                             
    bOrderMPI = False                       # order task distribution [sequentially if true]
    iMatrixSizeMax = 50
    bRollingScores = False

    # Estimation
    iP = 0 # AR order 
    bQML=False
    bQMLFull = False 
    bImpStat = False # Imposes stationarity when true
    bPackage = False # TGARCH can be compared with package by Kevind Sheppard
    
    sVersion = 'vFinal' + 'XLEF'
    sVersionGARCH = 'vTGFinal' + 'XLEF'
    sModels = 'DCC'
    sWeight = 'IndProd'
    
    #### Data ####

    # XLE data
    dfXLE = pd.read_csv('Data/XLEandRealVolXiu.csv', sep=',')
    dfXLE.index = pd.to_datetime(dfXLE['Date'])  # Ensure Date is in datetime format
    vY1 = dfXLE['XLE'].values * 100
    vRV1 = dfXLE['rv'].values * 100**2  # Better scale for numerical optimization
    
    # XLF data
    dfXLF = pd.read_csv('Data/XLFandRealVolXiu.csv', sep=',')
    dfXLF.index = pd.to_datetime(dfXLF['Date'])  # Ensure Date is in datetime format
    vY2 = dfXLF['XLF'].values * 100
    vRV2 = dfXLF['rv'].values * 100**2  # Better scale for numerical optimization
    
    # Find intersection of dates
    dateCommon = dfXLE.index.intersection(dfXLF.index)
    
    # Filter dataframes to only include common dates
    dfXLE_filtered = dfXLE.loc[dateCommon]
    dfXLF_filtered = dfXLF.loc[dateCommon]
    
    # Update the relevant arrays based on the filtered data
    vY1 = dfXLE_filtered['XLE'].values * 100
    vRV1 = dfXLE_filtered['rv'].values * 100**2
    
    vY2 = dfXLF_filtered['XLF'].values * 100
    vRV2 = dfXLF_filtered['rv'].values * 100**2
    
    mY = np.vstack((vY1, vY2)).T
    mRV = np.vstack((vRV1, vRV2)).T

    vY= vY1
    vRV = vRV1
    
    iHmax = vH.max()
    
    iT = len(vY1)
    iTw = 0                                 # window length optimisation weights: unused
    iPredParams = 3 + int(3*bBivariate)
    
    # Number of windows
    iNumEstWindows = iT - iTest - iHmax + 1
    iNumDFWindows = iT - iTest -iTw - iHmax + 1 
    print('Number of esitmation windows:',iNumEstWindows)
    print('Number of density forecast windows:',iNumDFWindows ) 
   
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
              'CRPSFlat':{'fScore': CRPSFlatBivariateL,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPSFlat'}, 
              'CRPSSharp':{'fScore': CRPSSharpBivariate,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp$'},
              'CRPSSharpslog':{'fScore': CRPSSharpslogBivariate,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp + $slog'},
              'CRPSSharpsbar':{'fScore': CRPSSharpsbarBivariate,'vParamsS': dictCRPSSettings, 'bWeighted': True, 'Label': 'CRPS$^\\sharp + $sbar'},
              }
    
    lScoreNames = list(dictScores.keys())
    iS = len(lScoreNames)
    if sWeight == 'IndProd':
        dictW = {'fW': fWIndicatorLBivProd, 'vR': None,  'vParamsW': 0} #vR will be filled  for different values of q
    elif sWeight == 'LogProd2':
        dictW = {'fW': fWLogisticLBivProd, 'vR': None,  'vParamsW': 2} #vR will be filled  for different values of q    
    elif sWeight == 'LogProd3':
        dictW = {'fW': fWLogisticLBivProd, 'vR': None,  'vParamsW': 3} #vR will be filled  for different values of q
    elif sWeight == 'LogProd4':
        dictW = {'fW': fWLogisticLBivProd, 'vR': None,  'vParamsW': 4} #vR will be filled  for different values of q    
    elif sWeight == 'IndSum':
        dictW = {'fW': fWIndicatorLBivSum, 'vR': None,  'vParamsW': 0} #vR will be filled  for different values of q 
          
    sSettingsModels = sModels + '_iT_' + str(iT) + '_iTest_' +  str(iTest) + '_' + sVersionGARCH +'_iP_'+ str(iP) + '_p_' + str(int(bPackage)) + '_qml_' + str(int(bQML+bQMLFull)) 
    sSettings = sSettingsModels + '_iTw_' +  str(iTw) + '_dRq_' +  str(int(dRq*100)) + '_iS_' +  str(iS) +'_' + sVersion 

    ## Load calculated h-step ahead forecasts        
    
    # Create vectors y_i = y_{t+h}, for t= iTest - 1, ..., iT-1-iHmax, i = 0, ..., iT-1-iHmax-iTest+1 = number of estimation windows minus 1
    mYh1 = mY[iTest - 1 + 1: iT-iHmax + 1,:] 
    mYh5 = mY[iTest - 1 + 5: iT-iHmax + 5,:]

    # univariate: iM x 3 [mu, sig2, nu] x iNumEstWindows
    # bivariate: iM x 6 [mu1, mu2, sig1, sig12, sig2, nu]
    mParamsDFh1 = np.load('mParamsDF/mParamsDF_h1_' + sSettingsModels + '.npy')
    mParamsDFh5 = np.load('mParamsDF/mParamsDF_h5_' + sSettingsModels + '.npy')
    mRhat = np.load('mParamsDF/mRhat_' + sSettingsModels + '.npy')
  
    ######################################################################    
    # Calculate scores

    ## Initialisation MPI device
    comm = MPI.COMM_WORLD
    iRank = comm.Get_rank()  # rank process
    iProc = comm.Get_size()  # total number of processes 
        
    # Each estimation window implies one forecast and hence one 
    vIntScoresByProc = MPITaskDistributor(iRank, iProc, iNumEstWindows, bOrder=bOrderMPI) # calculate indices windows assigned to this process
    print((iRank,vIntScoresByProc))
      
    # Calculation: density forecast parameters
    mScoresProc = RollingScoresMPI(dictMethods, dictScores, dictW, [mYh1, mYh5], [mParamsDFh1, mParamsDFh5], mRhat, vIntScoresByProc, bBivariate)
    lScores = comm.gather(mScoresProc, root=0)
        
    comm.barrier() # wait for everyone
    print(lScores)

    ## Combine data on root 
    if iRank == 0:
            
        mScores = np.ones((len(vH), len(lRq), iM, iS,iNumEstWindows)) * np.nan
        for rank in range(len(lScores)):
            vIntRank = MPITaskDistributor(rank, iProc, iNumEstWindows, bOrder=bOrderMPI)
            for i in range(vIntRank.size): 
                mScores[:,:,:,:,vIntRank[i]] = lScores[rank][:,:,:,:,i]

        # Save h=1 and h=5 in separate files        
        np.save('mScores/mScores_h1_' + sWeight + sSettings + '.npy' , mScores[0,:]) # settings instead of sSettingsModels
        np.save('mScores/mScores_h5_' + sWeight + sSettings + '.npy' , mScores[1,:])
   
###########################################################
### start main
if __name__ == "__main__":
    main()
