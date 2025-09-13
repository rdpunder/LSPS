#### Imports

# Fundamentals
import numpy as np  
import pandas as pd
from scipy import stats # pre-programmed random variables
from scipy import integrate # numerical integration
from scipy.integrate import quad

# MPI
from mpi4py import MPI

# System
import sys, os

# Dependencies
from ScoringRulesLocalDiv import *
from DivergencesBasis import *
from DivergencesPlot import *

# Plots
import matplotlib.pyplot as plt
from matplotlib import rc
os.environ["PATH"] += os.pathsep + '/usr/local/bin' 
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':'15'})
rc('text', usetex=True)
#%matplotlib qt

###########################################################  
### main
def main():    
    
    ###########################################################
    ## Settings
    iGoal = 1          # 1: calculate divergences, 2: plotting
    bSwitch = True     # apply distributions in reversed order if true: G, F 
 
    ## Distributions
    # NormT5, LP10LP11, NormNorm
    sCandidates = 'NormT5'
    
    randF, randG = funcDistributionSelection(sCandidates, bSwitch)

    ## Scoring rules
    vParamsS = 2 # alpha parameter for PsSphS and PowS
    lFuncScoringRules = [funcLogS, funcPsSphS, funcPowS]
    lFuncScoringRulesLocal = [funcLogSLocal, funcPsSphSLocal, funcPowSLocal, funCRPSLocal]
    sScoringRuleReference = 'LogSSphSQSCRPS' # reference scoring rule
    lScoringRuleNames = ['LogS', 'SphS', 'QS', 'CRPS']
    
    ## Weight functions 
    # Grid for r
    iRTotal = 300 # total number of r values
    lSides = ['C']
    dictWeightFunction =  funcWeightFuncDict(lSides, iRTotal)
    # Add twCRPS
    if lSides[0] == 'C':
        lFuncScoringRulesLocal += [funtwCRPS]
        lScoringRuleNames += ['twCRPS']
    
    ## Restricted correlation
    bRestricted = True
    
    # Settings numerical integration
    dDeltaY = 1e-4
    dNumInf = 20
    vYGrid = np.arange(-dNumInf, dNumInf, dDeltaY)
    
    # MPI
    bOrderMPI = False # order task distribution [sequentially if true]

    # Version
    sVersion = 'bSwitch'+str(bSwitch)
    
    ###########################################################    
    if iGoal == 1:
        
        ## Initialisation MPI device
        comm = MPI.COMM_WORLD
        iRank = comm.Get_rank()  # rank process
        iProc = comm.Get_size()  # total number of processes 
    
        # Distribute tasks
        vInt = MPITaskDistributor(iRank, iProc, iRTotal, bOrder=bOrderMPI) # calculate indices simulations assigned to this process
        
        # Calculate moments  
        mMomentsProc = funcMomentsScoreDiff(randF, randG, vInt, lFuncScoringRulesLocal, vYGrid, vParamsS, dictWeightFunction, dNumInf, dDeltaY)
        lMoments = comm.gather(mMomentsProc, root=0) # gather moments from all processes    
        comm.barrier() # wait for everyone
    
        # Combine data on root
        if iRank == 0:
            mMoments = np.ones((len(dictWeightFunction) ,len(lScoringRuleNames),iRTotal,10))* np.nan
            iCount = 0
            for rank in range(len(lMoments)):
                vIntRank = MPITaskDistributor(rank, iProc, iRTotal, bOrder=bOrderMPI) # calculate indices assigned to this process
                for i in range(len(vIntRank)):
                    mMoments[:,:,vIntRank[i],:] = lMoments[rank][:,:,i,:]
    
            # Save output as dataframe
            for w in range(len(dictWeightFunction)):
                sSide = list(dictWeightFunction.keys())[w]
                vRProc = dictWeightFunction[sSide]['vR'][vInt]
                vParamsW = dictWeightFunction[sSide]['vParamsW']
                for s in range(len(lScoringRuleNames)):
                    sScoringRule = lScoringRuleNames[s]
                    funcDataFrameMomentsSDiff(mMoments[w,s,:,:], sScoringRule, dictWeightFunction, sSide, sCandidates, sVersion)
           
    ###########################################################    
    if iGoal == 2:
       
        # Dataframes
        dictDataFramesSDiff = {}
        for sSide in lSides:
            dictDataFramesSDiff[sSide] = {}
            for sScoringRule in lScoringRuleNames:
                vParamsW = dictWeightFunction[sSide]['vParamsW']
                sFileNameSDiff = 'OutputDataFrames/mMomentsSDiff_RTot'+str(iRTotal)+'_'+sScoringRule+'_'+str(sSide)+str(vParamsW)+'_'+sCandidates+'_' + 'bSwitch'+'False'+'.xlsx'
                dictDataFramesSDiff[sSide][sScoringRule] = pd.read_excel(sFileNameSDiff)
        
        dictDataFramesSDiffSwitch = {}
        for sSide in lSides:
            dictDataFramesSDiffSwitch[sSide] = {}
            for sScoringRule in lScoringRuleNames:
                vParamsW = dictWeightFunction[sSide]['vParamsW']
                sFileNameSDiff = 'OutputDataFrames/mMomentsSDiff_RTot'+str(iRTotal)+'_'+sScoringRule+'_'+str(sSide)+str(vParamsW)+'_'+sCandidates+'_'+'bSwitch'+'True'+'.xlsx'
                dictDataFramesSDiffSwitch[sSide][sScoringRule] = pd.read_excel(sFileNameSDiff)
        
        # Figures of shares (XiS)
        if lSides[0] == 'C':
            funcPlotsPerWAllS(dictDataFramesSDiff,dictDataFramesSDiffSwitch, lScoringRuleNames[:-1], sCandidates,'bSwitch'+'False', 'XiS',lKeys=['XiSslog', 'XiSsbar'])
        else:
            funcPlotsPerWAllS(dictDataFramesSDiff,dictDataFramesSDiffSwitch, lScoringRuleNames, sCandidates,'bSwitch'+'False', 'XiS',lKeys=['XiSslog', 'XiSsbar'])
          
        # Figures of Standardized Divergences
        lLabels = ['${S^\\sharp}$', '${S^\\flat}$', '${Sslog}$', '${Ssbar}$']
        lKeys = [ 'StandDivSSharp','StandDivSFlat', 'StandDivSSharpslog', 'StandDivSSharpsbar']
        sFolder = 'StandDiv'
        sTitle = 'Standardized Divergences'
        if lSides[0] == 'C':
            funcPlotsPerWPerSInclTwCRPS(dictDataFramesSDiff, dictDataFramesSDiffSwitch, lScoringRuleNames, sCandidates, sVersion, sFolder, lKeys, lLabels, sFolder)
            funcPlotsPerWPerS(dictDataFramesSDiff, dictDataFramesSDiffSwitch, lScoringRuleNames[:-1], sCandidates, sVersion, sFolder, lKeys, lLabels, sFolder)
        else:
            funcPlotsPerWPerS(dictDataFramesSDiff, dictDataFramesSDiffSwitch, lScoringRuleNames, sCandidates, sVersion, sFolder, lKeys, lLabels, sFolder)

###########################################################
### start main
if __name__ == "__main__":
    main()
