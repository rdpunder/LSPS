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

# System
import os
from os import path

# Dependencies
from TGARCHmodel import *
from DCCmodel import *
from RiskManBasisMV import *
from BivariateT import *

###########################################################  
def main():    
    
    ## Magic numbers
    bTest = False
    iGoal = 0                              # 0: density forecast params, 11: approach 1, 12 approach 2
    dRq = 0.25
    iTest = 1000                            # window length parameter estimation
    
    vH = np.array([1, 5])                   # forecast horizons
    lRq = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25] 
    lH = [1,5]                              # forecast horizons
    bBivariate = True
    iSeed = 1234
    np.random.seed(iSeed)                   # set random seed
    
    iPrecision = 4                          # precision LaTeX tables
    iPrecisionDM = 2
                             
    bOrderMPI = False                       # order task distribution [sequentially if true]
    iMatrixSizeMax = 50
    # Temp
    bRollingScores = False
    bVaR = False
    bDM = False
    
    # Estimation
    iP = 0 # AR order 
    bQML=False
    bQMLFull = False 
    bImpStat = False # Imposes stationarity when true
    bPackage = False # TGARCH can be compared with package by Kevind Sheppard
    
    sVersion = 'vFinal' + 'XLEF'
    sVersionGARCH = 'vTGFinal' + 'XLEF'
    sModels = 'DCC'
    
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
    
    ####################
    # Methods
    ####################

    # Note: #windows_estim = iT - iTest - iHmax + 1  and #windows_estim = iT - iTest - iHmax + 1 - iT_w,
    # equivalent to Opschoor et al. (2017)
    
    # Bivariate Methods
    def DCC_GARCH_N(mY, iP, vH, mRV=None):
        return DCCEstimAndForecast(mY, vH, iP, bGamma=False, bRealized=False, sDistr='normal', mRV=None)
        
    def DCC_GARCH_t(mY, iP, vH, mRV=None):
        return DCCEstimAndForecast(mY, vH, iP, bGamma=False, bRealized=False, sDistr='t', mRV=None)

    def DCC_TGARCH_N(mY, iP, vH, mRV=None):
        return DCCEstimAndForecast(mY, vH, iP, bGamma=True, bRealized=False, sDistr='normal', mRV=None)
    
    def DCC_TGARCH_t(mY, iP, vH, mRV=None):
        return DCCEstimAndForecast(mY, vH, iP, bGamma=True, bRealized=False, sDistr='t', mRV=None)
    
    def DCC_RGARCH_N(mY, iP, vH, mRV=None):
        return DCCEstimAndForecast(mY, vH, iP, bGamma=False, bRealized=True, sDistr='normal', mRV=mRV)
    
    def DCC_RGARCH_t(mY, iP, vH, mRV=None):
        return DCCEstimAndForecast(mY, vH, iP, bGamma=False, bRealized=True, sDistr='t', mRV=mRV)
            
    dictMethods = {
        'GARCH-Normal' : {'Model': DCC_GARCH_N, 'iP': iP, 'sDistr': 'Normal', 'sName': 'GARCH-$\\mathcal{N}$'}, 
        'GARCH-Std(nu)':  {'Model': DCC_GARCH_t, 'iP': iP, 'sDistr': 'Student-t', 'sName': 'GARCH-$t(\\nu)$'},
        'TGARCH-Normal' : {'Model': DCC_TGARCH_N, 'iP': iP,'sDistr': 'Normal', 'sName': 'TGARCH-$\\mathcal{N}$'}, 
        'TGARCH-Std(nu)':  {'Model': DCC_TGARCH_t,'iP': iP, 'sDistr': 'Student-t', 'sName': 'TGARCH-$t(\\nu)$'},
        'RGARCH-Normal':  {'Model': DCC_RGARCH_N,'iP': iP, 'sDistr': 'Normal', 'sName': 'RGARCH-$\\mathcal{N}$'},
       'RGARCH-Std(nu)':  {'Model': DCC_RGARCH_t,'iP': iP, 'sDistr': 'Student-t', 'sName': 'RGARCH-$t(\\nu)$'},
        }
    iM = len(list(dictMethods.keys())) 

    sSettingsModels = sModels + '_iT_' + str(iT) + '_iTest_' +  str(iTest) + '_' + sVersionGARCH +'_iP_'+ str(iP) + '_p_' + str(int(bPackage)) + '_qml_' + str(int(bQML+bQMLFull)) 
    
    #########################################################################
    ### Bivariate Density Forecast [params] for all horizons
    #########################################################################
    
    ## Initialisation MPI device
    comm = MPI.COMM_WORLD
    iRank = comm.Get_rank()  # rank process
    iProc = comm.Get_size()  # total number of processes 
           
    vIntWindows = MPITaskDistributor(iRank, iProc, iNumEstWindows, bOrder=bOrderMPI) # calculate indices windows assigned to this process
    print((iRank,vIntWindows))
    
    # Calculation: density forecast parameters
    mParamsDFProc  = PredDistrWindowMPI(dictMethods, vH, mY, mRV, dRq, iTest, vIntWindows, bBivariate)
    lParamsDF = comm.gather(mParamsDFProc, root=0)
        
    comm.barrier() # wait for everyone

    ## Combine data on root 
    if iRank == 0:
        mParamsDF = np.zeros((len(vH), iM,iPredParams,iNumEstWindows))
        for rank in range(len(lParamsDF)):
            vIntRank = MPITaskDistributor(rank, iProc, iNumEstWindows, bOrder=bOrderMPI)
            for i in range(vIntRank.size): 
                mParamsDF[:,:,:,vIntRank[i]] = lParamsDF[rank][:,:,:,i]
            
        mRhat = RhatMV(mY, lRq, iTest, iNumEstWindows)  

        # Save h=1 and h=5 in separate files        
        np.save('mParamsDF/mRhat_' + sSettingsModels + '.npy' , mRhat)
        np.save('mParamsDF/mParamsDF_h1_' + sSettingsModels + '.npy' , mParamsDF[0,:])
        np.save('mParamsDF/mParamsDF_h5_' + sSettingsModels + '.npy' , mParamsDF[1,:])
   
###########################################################
### start main
if __name__ == "__main__":
    main()
