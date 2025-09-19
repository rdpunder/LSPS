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
from TGARCHmodel import *
from RiskManBasis import *

###########################################################  
def main():    
     
    ## Magic numbers
    bTest = False
    iGoal = 0                              # 0: density forecast params, 11: approach 1, 12 approach 2
    dRq = 0.25
    iTest = 1250                            # window length parameter estimation
    
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
    
    iT = len(vY)
    iTw = 0  # window length optimisation weights: unused
       
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
    
    # Methods
    def GARCH_N(vY, iP, vH, vRV=None):
        return TGARCHEstimAndForecast(vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=bPackage, vH=vH, bQML=bQML, bQMLFull=bQMLFull, bImpStat=bImpStat)
    
    def GARCH_t(vY, iP, vH, vRV=None):
        return TGARCHEstimAndForecast(vY, iP, bGamma=False, bRobust=False, bTheta=True, vThetaFixed=None, sDistr='t', bPackage=bPackage, vH=vH, bQML=bQML,  bQMLFull=bQMLFull, bImpStat=bImpStat)

    def TGARCH_N(vY, iP, vH, vRV=None):
        return TGARCHEstimAndForecast(vY, iP, bGamma=True, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=bPackage, vH=vH, bQML=bQML, bQMLFull=bQMLFull, bImpStat=bImpStat)
    
    def TGARCH_t(vY, iP, vH, vRV=None):
        return TGARCHEstimAndForecast(vY, iP, bGamma=True, bRobust=False, bTheta=True, vThetaFixed=None, sDistr='t', bPackage=bPackage, vH=vH, bQML=bQML, bQMLFull=bQMLFull, bImpStat=bImpStat)
    
    def RGARCH_N(vY, iP, vH, vRV=None):
        return RGARCHEstimAndForecast(vY, iP, bGamma=False, bRobust=False, bTheta=False, vThetaFixed=None, sDistr='normal', bPackage=False, vH=np.array([1,5]), bQML=bQML, bQMLFull=True, vRV=vRV, bImpStat=bImpStat)
    
    def RGARCH_t(vY, iP, vH, vRV=None):
        return RGARCHEstimAndForecast(vY, iP, bGamma=False, bRobust=False, bTheta=True, vThetaFixed=None, sDistr='t', bPackage=False, vH=np.array([1,5]), bQML=bQML, bQMLFull=bQMLFull, vRV=vRV, bImpStat=bImpStat)
        
    # Change distr names later
    dictMethods = {
        'GARCH-Normal' : {'Model': GARCH_N, 'iP': iP, 'sDistr': 'Normal', 'sName': 'GARCH-$\\mathcal{N}$'}, 
        'GARCH-Std(nu)':  {'Model': GARCH_t, 'iP': iP, 'sDistr': 'Student-t', 'sName': 'GARCH-$t(\\nu)$'},
        'TGARCH-Normal' : {'Model': TGARCH_N, 'iP': iP,'sDistr': 'Normal', 'sName': 'TGARCH-$\\mathcal{N}$'}, 
        'TGARCH-Std(nu)':  {'Model': TGARCH_t,'iP': iP, 'sDistr': 'Student-t', 'sName': 'TGARCH-$t(\\nu)$'},
        'RGARCH-Normal':  {'Model': RGARCH_N,'iP': iP, 'sDistr': 'Normal', 'sName': 'RGARCH-$\\mathcal{N}$'},
       'RGARCH-Std(nu)':  {'Model': RGARCH_t,'iP': iP, 'sDistr': 'Student-t', 'sName': 'RGARCH-$t(\\nu)$'},
        }
    iM = len(list(dictMethods.keys())) 

    sSettingsModels = sModels + '_iT_' + str(iT) + '_iTest_' +  str(iTest) + '_' + sVersionGARCH +'_iP_'+ str(iP) + '_p_' + str(int(bPackage)) + '_qml_' + str(int(bQML+bQMLFull)) 
    
    #########################################################################
    ### Univariate Density Forecast [params] for all horizons
    #########################################################################
    
    ## Construct h-step ahead density forecasts, that is, the underlying parameters [fully parametric methods only]
    if iGoal == 0 : 
        
        ## Initialisation MPI device
        comm = MPI.COMM_WORLD
        iRank = comm.Get_rank()  # rank process
        iProc = comm.Get_size()  # total number of processes 
           
        vIntWindows = MPITaskDistributor(iRank, iProc, iNumEstWindows, bOrder=bOrderMPI) # calculate indices windows assigned to this process
        print((iRank,vIntWindows))
      
        # Calculation: density forecast parameters
        mParamsDFProc  = PredDistrWindowMPI(dictMethods, vH, vY, vRV, dRq, iTest, vIntWindows)
        lParamsDF = comm.gather(mParamsDFProc, root=0)
        
        comm.barrier() # wait for everyone
        print(lParamsDF)

        ## Combine data on root 
        if iRank == 0:
            mParamsDF = np.zeros((len(vH), iM,3,iNumEstWindows))
            for rank in range(len(lParamsDF)):
                vIntRank = MPITaskDistributor(rank, iProc, iNumEstWindows, bOrder=bOrderMPI)
                for i in range(vIntRank.size): 
                    mParamsDF[:,:,:,vIntRank[i]] = lParamsDF[rank][:,:,:,i]
            
            # Calculate Rhats
            mRhat = Rhat(vY, lRq, iTest, iNumEstWindows)  
            
            # Save h=1 and h=5 in separate files        
            np.save('mParamsDF/mParamsDF_h1_' + sSettingsModels + '.npy' , mParamsDF[0,:])
            np.save('mParamsDF/mParamsDF_h5_' + sSettingsModels + '.npy' , mParamsDF[1,:])
            np.save('mParamsDF/mRhat_' + sSettingsModels + '.npy' , mRhat)
   
###########################################################
### start main
if __name__ == "__main__":
    main()
