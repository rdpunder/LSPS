#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Estimate temperature-GARCH models and produce forecasts
"""

## Imports

# Fundamentals
import numpy as np  

# MPI
from mpi4py import MPI

# Pandas
import pandas as pd

# Visualisation
import matplotlib.pyplot as plt
from matplotlib import rc

# Dependencies
from ClimateLocMeanSinGARCH import * 
from ClimateLocMeanSinQGARCHI import * 
from ClimateLocMeanSinQGARCHII import * 
from ClimateBasis import *

###########################################################  
def main():    
    
    ## Magic numbers  
    bDayNumbers = True                    # use T_t = t if true
    bWarm = False                         # warm start not used for results in paper
    iTest = 8*365+2                       # window length parameter estimation    
    vH = np.array([1,2,3])                # forecast horizons 
    iHmax = vH.max()                      # largest forecast horizon
    iSeed = 1234
    np.random.seed(iSeed)                 # set random seed
    bOrderMPI = False                     # order task distribution [sequentially if true]
    sVersion = 'vFinal'
    sModels = 'TempGARCH-norm-t'
    
    lRq = [0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99] # quantile grid empirical quantiles
    lR = [1,2,4]                                      # grid fixed thresholds
    dC = 18                                           # center region of interest for center case
    
    # Daily data
    sStart = '2003-02-01'
    sEnd = '2023-01-31'
    sDep = 'TempAvg' # options: Tmax, Tmin, Tavg
    dfDataFull = pd.read_excel('ClimateKNMI_Temp.xlsx')
    dfDataFull['Date'] = pd.to_datetime(dfDataFull['Date'], format='%Y%m%d') # Convert the 'date' column to datetime format
    dfDataFull.set_index('Date', inplace=True)
    dfData = dfDataFull[sDep]/10
    dfDataSelect = dfData[sStart:sEnd]

    vY = dfDataSelect.values
    vT = fWeekNumbers(dfDataSelect, False, bDayNumbers)/20
    iT = len(vY)
    
    sSettingsModels = sModels + '_iT_' + str(iT) + '_iTest_' +  str(iTest) + '_bWarm_' + str(bWarm)
            
    # Number of windows
    iNumEstWindows = iT - iTest - iHmax + 1 
    
    # Methods
    dictMethods = {
        'GARCH-Normal' : {'Model': GARCHEstimAndForecast,'sDistr': 'normal', 'sName': 'QGARCHI-$\\mathcal{N}$'},
        'GARCH-Std(nu)' : {'Model': GARCHEstimAndForecast,'sDistr': 't', 'sName': 'QGARCHI-$t$'},
        'QGARCHI-I-Normal' : {'Model': QGARCHIEstimAndForecast,'sDistr': 'normal', 'sName': 'QGARCHI-$\\mathcal{N}$'},
        'QGARCHI-I-Std(nu)' : {'Model': QGARCHIEstimAndForecast,'sDistr': 't', 'sName': 'QGARCHI-$t$'},
        'QGARCHI-II-Normal' : {'Model': QGARCHIIEstimAndForecast,'sDistr': 'normal', 'sName': 'QGARCHI-$\\mathcal{N}$'},
        'QGARCHI-II-Std(nu)' : {'Model': QGARCHIIEstimAndForecast,'sDistr': 't', 'sName': 'QGARCHI-$t$'}
        }
    iM = len(list(dictMethods.keys()))
   
    #######################################################
    ## Construct h-step ahead density forecasts, that is, the underlying parameters
        
    ## Initialisation MPI device
    comm = MPI.COMM_WORLD
    iRank = comm.Get_rank()  # rank process
    iProc = comm.Get_size()  # total number of processes 
           
    vIntWindows = MPITaskDistributor(iRank, iProc, iNumEstWindows, bOrder=bOrderMPI) # calculate indices windows assigned to this process
    print((iRank,vIntWindows))
      
    # Calculation: density forecast parameters
    mParamsDFProc  = PredDistrWindowMPI(dictMethods, vH, vY, vT, iTest, vIntWindows, bWarm)
    lParamsDF = comm.gather(mParamsDFProc, root=0)
        
    comm.barrier() # wait for everyone

    ## Combine data on root 
    if iRank == 0:
        mParamsDF = np.zeros((len(vH),iM,3,iNumEstWindows))
        for rank in range(len(lParamsDF)):
            vIntRank = MPITaskDistributor(rank, iProc, iNumEstWindows, bOrder=bOrderMPI)
            for i in range(vIntRank.size): 
                mParamsDF[:,:,:,vIntRank[i]] = lParamsDF[rank][:,:,:,i]
            
        # Calculate Rhats
        mRhat = Rhat(vY, lRq, iTest, iNumEstWindows) 
        mGammaHat = GammaHat(vY, lR, dC, iTest, iNumEstWindows)
            
        # Save results per horizon
        for i in range(len(vH)):
            iH = vH[i]
            np.save('mParamsDF/mParamsDF_h' + str(iH) +'_' + sSettingsModels + '.npy' , mParamsDF[i,:])
            # Create vectors y_i = y_{t+h}, for t= iTest - 1, ..., iT-1-iHmax, i = 0, ..., iT-1-iHmax-iTest+1 = number of estimation windows minus 1
            vYhSelect = vY[iTest - 1 + iH: iT-iHmax + iH]
            np.save('mParamsDF/vYh' + str(iH) +'_' + sSettingsModels + '.npy' , vYhSelect)
                
        #np.save('mParamsDF/mParamsDF_h5_' + sSettingsModels + '.npy' , mParamsDF[1,:])
        np.save('mParamsDF/mRhat_' + sSettingsModels + '.npy' , mRhat)
        np.save('mParamsDF/mGammaHat_' + sSettingsModels + '.npy' , mGammaHat)
     
###########################################################
### start main
if __name__ == "__main__":
    main()