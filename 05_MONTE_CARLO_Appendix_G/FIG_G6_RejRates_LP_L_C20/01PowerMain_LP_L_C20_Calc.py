#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Main file power experiments
"""
###########################################################
### Imports

# Fundamentals
import numpy as np  
import pandas as pd
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration

# MPI
from mpi4py import MPI

# System
import os

# Plots
import matplotlib.pyplot as plt
from matplotlib import rc

# LaTeX formatting
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':'15'})
rc('text', usetex=True)
#%matplotlib qt

# Dependencies
from PowerBasis import *
from ScoringRulesMC  import *           
from WeightFunctionsMC  import *     
from PowerPlots import *

###########################################################  
### main
def main():    
    
    ########################################
    ### Magic numbers
    ########################################
    
    ## Goal
    iGoal = 1                   # 1: simulate data and calculate DM statistics, 2: plotting
    bLeftTail = True            # boolean, left tail indicator example if true, centre indicator if false
    sDensClass = 'Laplace'      # StudentNorm or Laplace
    
    ## Fundamentals
    iRtot = 500             # steps Rgrid [64, 288]
    iRep = 10000            # number of Monte Carlo replications
    
    iAxis = 0               # 0: threshold r
    iC = 20                 # expected number of observations in {w>0} [ r grid]
    dLeftTailPerc =0.01     # left tail percentile [c grid]
    
    dISWeightLB = 0.1       # observations assigned with weights below this threshold are included in IS
    iSeed = 4132            # random seed
    bOrderMPI = False       # order task distribution [sequentially if true]
    iTupper  = 5*iC         # upper bound sample size for IS, no IS if False
    iMatrixSizeMax = 5      # maximal matrix size CRPS computations
    
    ## Candidate distributions
    # Normal and Student-t
    if sDensClass == 'StudentNorm' :
        dNuF = 'Normal'
        dNuG = 5
        if dNuF == 'Normal':
            dictF = {'randDistr' : stats.norm(0,1), 'sDistr': 'Normal'}
        else:
            dictF = {'randDistr' : stats.t(dNuF,loc=0, scale=1/np.sqrt(dNuF/(dNuF-2))), 'sDistr' :'Student' + str(dNuF)}
        dictG = {'randDistr' : stats.t(dNuG,loc=0, scale=1/np.sqrt(dNuG/(dNuG-2))), 'sDistr' :'Student' + str(dNuG)}
        dictCand = {'dictF': dictF, 'dictG': dictG}
    # Laplace
    elif sDensClass == 'Laplace' :
        dMuF = -1
        dThetaF = 1
        dictF = {'randDistr' : stats.laplace(loc=dMuF, scale=dThetaF), 'sDistr' :'Lp' + str(dMuF)+'_'+str(dThetaF)} 
        dMuG = 1
        dThetaG = 1.1
        dictG = {'randDistr' : stats.laplace(loc=dMuG, scale=dThetaG), 'sDistr' :'Lp' + str(dMuG)+'_'+str(dThetaG)} 
        dictCand = {'dictF': dictF, 'dictG': dictG}   
    
    ## vRGrid [for iAxis=0]
    iRmin = int(bLeftTail) * -4 - 2 * int(sDensClass == 'Laplace') + (1-int(bLeftTail) ) * 0.1   # start iR grid
    iRmax = 4   -1 * int(sDensClass == 'Laplace')           # end iR grid
    
    vR = np.linspace(iRmin, iRmax, iRtot)
    
    ## vCGrid [for iAxis=1]
    vC = np.array([0]) # depreciated
  
    ## Rejection rates DM statistics
    lNominalSize = [0.05]
    dictCritVal = { 'FavG': [stats.norm.ppf(dNominalSize) for dNominalSize in lNominalSize], 'FavF': [stats.norm.ppf(1-dNominalSize) for dNominalSize in lNominalSize]} # H_1 E d < (>) 0
    dictCritValCal = { 'L': [stats.norm.ppf(dNominalSize/2) for dNominalSize in lNominalSize], 'R': [stats.norm.ppf(1-dNominalSize/2) for dNominalSize in lNominalSize]} # H_1 E d < (>) 0
   
    ## Plots
    lStyles = ['-', '--', ':', '-.']
    lCycle = plt.rcParams['axes.prop_cycle'].by_key()['color'] # blue, orange, green, red, brown, purple

    ## Scoring rules
    dictCRPSNumInt = {'dMinInf':-20, 'dPlusInf': 20, 'dDeltaZ' : 0.001, 'iMatrixSizeMax':iMatrixSizeMax, 'iSeed': 2314,'CensDistParams': 0}
    vAlpha = np.array([2])
    dAlpha = 2 # selected alpha norm  
    dictScores = {
                'LogSSharp': {'fScore': LogSSharp,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\sharp$', 'Colour': lCycle[0], 'Symbol': lStyles[0]},
                'LogSFlat': {'fScore': LogSFlat,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\flat$', 'Colour': lCycle[1], 'Symbol': lStyles[0]},
                'LogSsbar': {'fScore': LogSsbar,'vParamsS': None, 'bWeighted': True, 'Label': 'LogSsbar', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
                'LogSslog': {'fScore': LogSslog,'vParamsS': None, 'bWeighted': True, 'Label': 'LogSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]},
                'PsSphSSharp': {'fScore': PsSphSSharp,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\sharp$', 'Colour': lCycle[0], 'Symbol': lStyles[0]}, 
                'PsSphSFlat': {'fScore': PsSphSFlat,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphS$^\\flat$', 'Colour': lCycle[1], 'Symbol': lStyles[0]},
                'PsSphSsbar': {'fScore': PsSphSsbar,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphSsbar', 'Colour': lCycle[2], 'Symbol': lStyles[0]}, 
                'PsSphSslog': {'fScore': PsSphSslog,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'SphSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]},
                'PowSSharp': {'fScore': PowSSharp,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\sharp$', 'Colour': lCycle[0], 'Symbol': lStyles[0]},
                'PowSFlat': {'fScore': PowSFlat,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QS$^\\flat$', 'Colour': lCycle[1], 'Symbol': lStyles[0]},
                'PowSsbar': {'fScore': PowSsbar,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QSsbar', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
                'PowSslog': {'fScore': PowSslog,'vParamsS': dAlpha, 'bWeighted': True, 'Label': 'QSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]},
                'CRPSSharp':{'fScore': CRPSSharp,'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'CRPS$^\\sharp$', 'Colour': lCycle[0], 'Symbol': lStyles[0]},
                'CRPSFlat':{'fScore': twCRPS,'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'CRPS$^\\flat$', 'Colour': lCycle[1], 'Symbol': lStyles[0]}, #equiv. has also been verified numerically
                'CRPSsbar':{'fScore': CRPSsbar,'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'CRPSsbar', 'Colour': lCycle[2], 'Symbol': lStyles[0]},
                'CRPSslog':{'fScore': CRPSslog,'vParamsS': dictCRPSNumInt, 'bWeighted': True, 'Label': 'CRPSslog', 'Colour': lCycle[3], 'Symbol': lStyles[0]}, 
                 }
    
    ## Selected weight function
    if bLeftTail: 
        dictW = {'fW': fWIndicatorL, 'vR': vR, 'vParamsW': ''} # left tail indicator
    else:
        dictW = {'fW': fWIndicatorC, 'vR': vR, 'vParamsW': ''} # center indicator
    sWeightFunction = dictW['fW'].__name__
    
    ## Save settings
    sCurrentFileName = 'PowerMain' 
    
    # Tmax for fWIndicatorL
    if iAxis ==0:
        if sWeightFunction == 'fWIndicatorL':
            vTmax = np.array([int(np.round(iC/dictF['randDistr'].cdf(vR.min()))),int(np.round(iC/dictG['randDistr'].cdf(vR.min())))]) #    # Calculate maximum number of observations [to comply with iC for all iR and both candidates]
        else:
            vTmax = np.array([int(np.round(iC/(1-DistrBar(dictF, dictW, np.arange(0,vR.size)).max()))), int(np.round(iC/(1-DistrBar(dictG, dictW, np.arange(0,vR.size)).max())))])
    else:
        vTmax = np.array([int(np.round(iC/(1-DistrBar(dictF, dictW, np.array([0]))[0]))), int(np.round(iC/(1-DistrBar(dictG, dictW, np.array([1]))[1])))])
    
    sSelectedScores = 'AllScores' #ScoringRuleCollectionName(dictScores)
    sSettingsHorizontalGrid =  '_iC'  + str(iC) + '_iRmin' +  str(int(abs(iRmin*100))) + '_iRmax' +  str(int(abs(iRmax*100))) + '_iRtot' +  str(int(iRtot)) 
    sSettings = 'CandF_' + dictCand['dictF']['sDistr'] + '_CandG_' + dictCand['dictG']['sDistr'] + sSettingsHorizontalGrid + '_fW_' + sWeightFunction +'_iTmax' +  str(int(vTmax.max())) + '_iTupper' +  str(iTupper) + '_iRep' +  str(iRep) + '_'+ sCurrentFileName 
    sSettingsF = 'CandF_' + dictCand['dictF']['sDistr'] +  sSettingsHorizontalGrid + '_fW_' + sWeightFunction + '_iTmax' +  str(vTmax[0])  + '_iTupper' +  str(iTupper) + '_iRep' +  str(iRep)  + '_'+ sCurrentFileName 
    sSettingsG = 'CandG_' + dictCand['dictG']['sDistr'] +  sSettingsHorizontalGrid + '_fW_' + sWeightFunction + '_iTmax' +  str(vTmax[1]) + '_iTupper' +  str(iTupper) + '_iRep' +  str(iRep) + '_'+ sCurrentFileName 
   
    ########################################
    ### Calculate DM statistics and Rejection rates
    ########################################
    if iGoal == 1:
  
        ## Load data and IS weights
        iRorCSize = int(vR.size * (iAxis==0) + vC.size * (iAxis==1)) 
        
        ## Initialisation MPI device
        comm = MPI.COMM_WORLD
        iRank = comm.Get_rank()  # rank process
        iProc = comm.Get_size()  # total number of processes 

        # Distribute
        vInt = MPITaskDistributor(iRank, iProc, iRorCSize, bOrder=bOrderMPI) # calculate indices simulations assigned to this process
        print((iRank,vInt))
        vIntR = vInt

        # Simulate data
        if iRank ==0:
            SimulateData(dictCand, dictW, np.arange(0,iRtot), iC, iTupper, dISWeightLB, iRep, iSeed, [sSettingsF,sSettingsG], vTmax, True, True, True, False)
        comm.barrier()
        
        # Precalculate norms and Fbar for vR grid and vAlpha
        dictPreCalc = {'AlphaNormsF': {dictF['sDistr']: AlphaNormsF(dictF, dictW, vAlpha), dictG['sDistr']: AlphaNormsF(dictG, dictW, vAlpha)},
                       'AlphaNormsFw': {dictF['sDistr']: AlphaNormsFw(dictF, dictW, vAlpha, vIntR), dictG['sDistr']: AlphaNormsFw(dictG, dictW, vAlpha, vIntR)},
                       'DistrBar': {dictF['sDistr']: DistrBar(dictF, dictW, vIntR), dictG['sDistr']: DistrBar(dictG, dictW, vIntR)},
                       }
        
        comm.barrier() # wait for everyone
        if iRank ==0: print('Pre-calculations completed')
       
        # DM calculations
        mDM = np.zeros((2,len(list(dictScores.keys())),iRep,iRorCSize)) # dgps x scores x reps x rgrid
        
        # Split due to MPI memory issues
        mDMProcFDM, mDMProcGDM = DMTestMPI(dictCand, dictScores, dictW,  [sSettingsF,sSettingsG], dictPreCalc, iC, iTupper, iRep, iRank, vInt, vC, dLeftTailPerc)
        lFDM = comm.gather(mDMProcFDM, root=0)
        lGDM = comm.gather(mDMProcGDM, root=0)
        comm.barrier() # wait for everyone

        ## Combine data on root 
        if iRank == 0:
            iCount = 0
            for rank in range(len(lFDM)):
                vIntRank = MPITaskDistributor(rank, iProc, iRorCSize, bOrder=bOrderMPI)
                for i in range(vIntRank.size): 
                    mDM[0,:,:,vIntRank[i]] = lFDM[rank][:,:,i]
                    mDM[1,:,:,vIntRank[i]] = lGDM[rank][:,:,i]
                    
            if iTupper != False and vC.sum() == 0:    
                for s in range(len(list(dictScores.keys()))):
                    if not dictScores[list(dictScores.keys())[s]]['bWeighted']:
                        for i in range(int(iRorCSize-1)):
                            mDM[:,s,:,i] = mDM[:,s,:,-1]
            
            print('DM statistics:\n', np.round(mDM[0,:,0:5,0],2))
            np.save('mDMCalc/mDM_' + sSettings + '_fW_' + dictW['fW'].__name__ + '_' + sSelectedScores +  '.npy' , mDM)
    
        # Clean folder
        comm.barrier()
        if iRank==0:
            try:
                os.remove('mDataAndWeights/mTmax_' +  sSettingsF + '_fW_' + dictW['fW'].__name__+'.npy')
            except:
                None
                              
            for d in range(2):
                for r in range(iRtot):
                    try:
                        os.remove('mDataAndWeights/mDataAndWeights_' +  [sSettingsF,sSettingsG][d] + '_fW_' + dictW['fW'].__name__+ '_iRidx_'+str(r) + '.npy')
                    except:
                        None
                    
    ########################################
    ### Generate plots
    ########################################
    if iGoal==2:
        
            ## Enable interactive mode
            plt.ion()
        
            ## Cleaning
            plt.close('all')
            
            ## Load DM statistics and calculate rejection rates   
            mDM = np.load('mDMCalc/mDM_' + sSettings + '_fW_' + dictW['fW'].__name__ + '_' +  sSelectedScores + '.npy')
            
            ## Calculate rejection rates
            dictRejRatesDM = RejRates(mDM, dictCritVal)
            
            ## Select scores
            vIdxSelectedScoresDens = np.arange(0,12)
            vIdxSelectedScoresCRPS = np.arange(12,16)
            
            if iAxis == 0:
                if sDensClass == 'StudentNorm' :
                    dictRIdxGridPlots = {'TrueF': {'FavF': np.arange(0,iRtot)[vR >= -3][vR[vR >= -3] <= 3], 'FavG': np.arange(0,iRtot)[vR > -3][vR[vR > -3] < 3]}, 
                                      'TrueG': {'FavF': np.arange(0,iRtot)[vR > -4][vR[vR > -4] < 4], 'FavG': np.arange(0,iRtot)[vR > -4][vR[vR > -4] < 4]}}
                elif sDensClass == 'Laplace' :    
                    dictRIdxGridPlots = {'TrueF': {'FavF': np.arange(0,iRtot)[vR >= -6][vR[vR >= -6] <= 3], 'FavG': np.arange(0,iRtot)[vR > -6][vR[vR > -6] < 3]}, 
                                      'TrueG': {'FavF': np.arange(0,iRtot)[vR > -6][vR[vR > -6] < 3], 'FavG': np.arange(0,iRtot)[vR > -6][vR[vR > -6] < 4]}}
                
                PlotRejRates({'Values': vR, 'Label': 'threshold $r$'}, dictRejRatesDM, dictScores, np.arange(0,4), dictRIdxGridPlots, sSettings + '_DensLog', iLegCol=4, iStepX=1)
                dictScores['LogSFlat'] = {'fScore': LogSFlat,'vParamsS': None, 'bWeighted': True, 'Label': 'LogS$^\\flat$', 'Colour': lCycle[1], 'Symbol': lStyles[1]} 
                PlotRejRates({'Values': vR, 'Label': 'threshold $r$'}, dictRejRatesDM, dictScores, np.concatenate((np.arange(4,8),[1])), dictRIdxGridPlots, sSettings + '_DensSphS', iLegCol=5, iStepX=1)
                PlotRejRates({'Values': vR, 'Label': 'threshold $r$'}, dictRejRatesDM, dictScores, np.concatenate((np.arange(8,12),[1])), dictRIdxGridPlots, sSettings + '_DensQS', iLegCol=5, iStepX=1)
                PlotRejRates({'Values': vR, 'Label': 'threshold $r$'}, dictRejRatesDM, dictScores, np.concatenate((np.arange(12,16),[1])), dictRIdxGridPlots, sSettings + '_DensCRPS', iLegCol=5, iStepX=1)
                
            else:
                print('iAxis=1 is not supported')
                
            ## Disable interactive mode
            plt.ioff()
              
###########################################################
### start main
if __name__ == "__main__":
    main()
