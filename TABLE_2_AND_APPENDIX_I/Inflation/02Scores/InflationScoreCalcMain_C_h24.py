#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Calculate scores of density forecasts 
"""

## Imports

# Fundamentals
import numpy as np  
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration

# MPI
from mpi4py import MPI

# Dependencies
from ScoreBasisInflation import *          # elementary functions 
from ScoringRules import *                 # scoring rules
from Weightfunctions import *              # weight functions

###########################################################  
def main():    

    ## Magic numbers
    bTest = False
    lRq = [1,1.5,2]               # difference from target 
    dRTarget = 2                  # inflation target
    lH = [1]                      # forecast horizons
    iSeed = 1234
    np.random.seed(iSeed)         # set random seed
    iIdxHorizonInflation = 1
    iHorizonInflation= [6,24][iIdxHorizonInflation]
    bOrderMPI = False            # order task distribution [sequentially if true]
    iMatrixSizeMax = 50
    bPlotPred = False
    
    sVersion = 'vFinal'  
    sVersionGARCH = 'Infl' 
    sDistr = 'tpnorm' 
    sModels = 'ML-' + sDistr 
    
    #### Data ####
    mY = np.load('Data/mYAccOut.npy', allow_pickle=True) # note: this is already vYh1 [adjusted in R]
    vY = mY[:,iIdxHorizonInflation]
    vYh1 = np.copy(vY)
    vYh = np.copy(vY)
        
    # Number of windows
    iT = len(vY)
    mYFull=np.load('Data/mYAcc.npy',allow_pickle=True)
    mGammaHat = GammaHat(mYFull, lRq, dRTarget, iHorizonInflation, iT)
    
    ####################
    # Methods
    ####################

    # Models are estimated in R, the following is only used for variable definitions
    sDistr = 'tpnorm'
    sDistrFull = 'tpnorm'
        
    # Change distr names later
    lMeanNames = ['rw','ar','bagging','csr','lasso','rf' ]
        
    dictMethods = {
            lMeanNames[0] + sDistr : {'Model': None, 'iP': None, 'sDistr': sDistrFull, 'sName': 'rw-' + sDistr}, 
            lMeanNames[1] + sDistr :  {'Model': None, 'iP': None, 'sDistr': sDistrFull, 'sName': 'ar-' + sDistr},
            lMeanNames[2] + sDistr  : {'Model': None, 'iP': None,'sDistr': sDistrFull, 'sName': 'bagging-' + sDistr}, 
            lMeanNames[3] + sDistr :  {'Model': None,'iP': None, 'sDistr': sDistrFull, 'sName': 'csr-' + sDistr},
            lMeanNames[4] + sDistr :  {'Model': None,'iP': None, 'sDistr': sDistrFull, 'sName': 'lasso-' + sDistr},
            lMeanNames[5] + sDistr :  {'Model': None,'iP': None, 'sDistr': sDistrFull, 'sName': 'rf-' + sDistr},
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
    dictW = {'fW': fWIndicatorC, 'vR': None,  'vParamsW': dRTarget} #vR will be filled for different values of q
    
    sSettingsModels = sModels + '_iT_' + str(iT) 
    sSettings = sSettingsModels + '_iS_' +  str(iS) + '_' + str(dictW['fW'].__name__)  + '_' +   sVersion 
    
    #########################################################################
    ### Univariate Density Forecast [params] for all horizons
    #########################################################################
    
    ## Construct h-step ahead density forecasts, that is, the underlying parameters 
    mParamsDFh1, mRhat, iNumEstWindows = InflationParamsDFandmRhat(lMeanNames, sDistr, dRTarget, lRq, iHorizonInflation)
                  
    ######################################################################    
    # Calculate Scorers

    ## Initialization MPI device
    comm = MPI.COMM_WORLD
    iRank = comm.Get_rank()  # rank process
    iProc = comm.Get_size()  # total number of processes 
        
    # Each estimation window implies one forecast and hence one 
    vIntScoresByProc = MPITaskDistributor(iRank, iProc, iNumEstWindows, bOrder=bOrderMPI) # calculate indices windows assigned to this process
    print((iRank,vIntScoresByProc))
      
    # Calculation: density forecast parameters
    lYh = [vYh1]
    lParamsDFh = [mParamsDFh1]
            
    mScoresProc = RollingScoresMPI(dictMethods, dictScores, dictW, lYh, lParamsDFh, mRhat, vIntScoresByProc, mGammaHat)
    lScores = comm.gather(mScoresProc, root=0)
        
    comm.barrier() # wait for everyone

    ## Combine data on root 
    if iRank == 0:
        mScores = np.ones((1, len(lRq), iM, iS,iNumEstWindows)) * np.nan
        for rank in range(len(lScores)):
            vIntRank = MPITaskDistributor(rank, iProc, iNumEstWindows, bOrder=bOrderMPI)
            for i in range(vIntRank.size): 
                mScores[:,:,:,:,vIntRank[i]] = lScores[rank][:,:,:,:,i]

            np.save('mScores/mScores_h_' + str(iHorizonInflation) + '_' + sSettings + '.npy' , mScores[0,:]) 
           
###########################################################
### start main
if __name__ == "__main__":
    main()