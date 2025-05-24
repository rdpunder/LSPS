#### Imports

# Fundamentals
import numpy as np  
import pandas as pd
from scipy import stats # pre-programmed random variables

# System
import sys, os

# Dependencies
from DivergencesBasis import *

# Plots
import matplotlib.pyplot as plt
from matplotlib import rc
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':'14'})
rc('text', usetex=False)
#%matplotlib qt

###########################################################    
def funcPlotsPerWAllS(dictDataFrames,dictDataFramesSwitched, lScoringRules, sCandidates, sVersion, sFolder='XiS',lKeys=['XiSslog', 'XiSsbar'], iCol=4):
    """
    Purpose
    ----------
    Generate and save plots of score divergence corrections across scoring rules and sides.
    Each scoring rule is plotted twice with a consistent color: once with a solid line for the first key in lKeys,
    and once with a dashed line for the second key.
    
    Parameters
    ----------
    dictDataFrames :        dictionary, contains data frames per side and scoring rule (for F to G)
    dictDataFramesSwitched :dictionary, contains data frames per side and scoring rule (for G to F)
    lScoringRules :         list of strings, scoring rules to be plotted
    sCandidates :           string, identifier for the candidate distribution pair
    sVersion :              string, version label used in plot filenames
    sFolder :               optional, string, folder name for saving plots 
    lKeys :                 optional, list of strings, keys in the data frames to be plotted 
    iCol :                  optional, integer, number of columns in the legend layout
    
    Returns
    ----------
    None :                  plots are saved to disk; function returns nothing
    """

    for sSide in dictDataFrames.keys():
        
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        iCount = 0
        for i in range(len(lKeys)):
            sKey = lKeys[i]
            
            sStyle = '-'
            if iCount >0:
                sStyle = '--'
                axs[0].set_prop_cycle(None)
                axs[1].set_prop_cycle(None)
            for sScoringRule in lScoringRules:
                dfDiv = dictDataFrames[sSide][sScoringRule]
                dfDivSwitch = dictDataFramesSwitched[sSide][sScoringRule]                
                axs[0].plot(dfDiv['r'], dfDiv[sKey], label=sScoringRule+['slog','sbar'][iCount], linestyle=sStyle)
                axs[1].plot(dfDivSwitch['r'], dfDivSwitch[sKey], label=sScoringRule+['slog','sbar'][iCount], linestyle=sStyle)
            iCount += 1
            
        axs[0].set_title('$\\xi_{S,s}$ for F to G')
        axs[1].set_title('$\\xi_{S,s}$ for G to F')
        axs[0].set_xlabel('threshold $r$')
        axs[1].set_xlabel('threshold $r$')        
        plt.xlabel('threshold $r$')
        
        handles, labels = axs[0].get_legend_handles_labels()
        fig.tight_layout()
        fig.subplots_adjust(bottom=0.12, hspace=0.45)
       
        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=iCol, frameon=True)
        plt.savefig('Figures'+'/'+sFolder+'/'+sFolder+'_'+sSide+'_'+sCandidates+'_'+sVersion+'.pdf', bbox_inches='tight')
        plt.close()

###########################################################
def funcPlotsPerWPerS(dictDataFrames, dictDataFramesSwitched, lScoringRules, sCandidates, sVersion, sFolder, lKeys, lLabels, sReference):
    """
    Purpose
    ----------
    Generate and save side-by-side plots of selected statistics per scoring rule and weight function side, 
    with D(F||G) and D(G||F) shown in separate subplots

    Parameters
    ----------
    dictDataFrames :           dictionary, contains per side and scoring rule the corresponding DataFrame with D(F||G) results
    dictDataFramesSwitched :   dictionary, contains per side and scoring rule the corresponding DataFrame with D(G||F) results
    lScoringRules :            list, scoring rules to be plotted
    sCandidates :              string, name of candidate distributions
    sVersion :                 string, version label for file name
    sFolder :                  string, folder in which plots are stored
    lKeys :                    list, column names of DataFrame to be plotted
    lLabels :                  list, legend labels corresponding to lKeys
    sReference :               string, reference label used in file name

    Returns
    ----------
    Saves figures as PDF files in the specified folder
    """

    for sSide in dictDataFrames.keys():
        
        for sScoringRule in lScoringRules:
            iCol = 2
            lLabels = [sScoringRule+'$^\\sharp$', sScoringRule+'$^\\flat$', sScoringRule+'slog', sScoringRule + 'sbar' ]
            dfDiv = dictDataFrames[sSide][sScoringRule]
            dfDivSwitch = dictDataFramesSwitched[sSide][sScoringRule]

            fig, axs = plt.subplots(1, 2, figsize=(8, 4))
            
            for i in range(len(lKeys)):
                sKey = lKeys[i]
                if (sScoringRule == 'LogS') * (sKey == 'StandDivSSharpslog'):
                    continue
                axs[0].plot(dfDiv['r'], dfDiv[sKey], label=lLabels[i])
                axs[1].plot(dfDivSwitch['r'], dfDivSwitch[sKey], label=lLabels[i])
            
            # Add censored likelihood reference
            if sScoringRule != 'LogS':
                iCol += 1
                axs[0].plot(dfDiv['r'],dictDataFrames[sSide]['LogS']['StandDivSFlat'], label='LogS$^\\flat$', linestyle='--', color='orange')
                axs[1].plot(dfDivSwitch['r'], dictDataFramesSwitched[sSide]['LogS']['StandDivSFlat'], label='LogS$^\\flat$', linestyle='--', color='orange')
                
            axs[0].set_title('Local Divergence F to G')
            axs[1].set_title('Local Divergence G to F')
            axs[0].set_xlabel('threshold $r$')
            axs[1].set_xlabel('threshold $r$')

            handles, labels = axs[0].get_legend_handles_labels()
            fig.tight_layout()
            fig.subplots_adjust(bottom=0.14, hspace=0.45)  # More space below for the legend
            
            fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=iCol, frameon=True)

            sFileName = 'Figures' + '/' + sFolder + '/' + sReference + '_' + sSide + '_' + sScoringRule + '_' + sCandidates + '_' + sVersion + '.pdf'
            plt.savefig(sFileName, bbox_inches='tight')
            plt.close()

###########################################################    
def funcPlotsPerWPerSInclTwCRPS(dictDataFrames, dictDataFramesSwitched, lScoringRuleNames, sCandidates, sVersion, sFolder, lKeys, lLabels, sReference):
    """
    Purpose
    ----------
    Generate and save plot for CRPS variants including twCRPS in the center case where truncation and censoring differ

    Parameters
    ----------
    dictDataFrames :   dictionary, contains per side and scoring rule the corresponding DataFrame with results
    sCandidates :      string, name of candidate distributions
    sVersion :         string, version label for file name
    sFolder :          string, folder in which plots are stored
    lKeys :            list, column names of DataFrame to be plotted
    lLabels :          list, legend labels corresponding to lKeys
    sReference :       string, reference label used in file name

    Returns
    ----------
    Saves figure as PDF file in the specified folder
    """
    
    # Center case: twCRPS and censoring do not coincide
    sSide = 'C'
    sScoringRule = 'CRPS'
    dfSelect = dictDataFrames['C']['CRPS'] 
    dfTwCRPS = pd.read_excel('OutputDataFrames/mMomentsSDiff_RTot300_twCRPS_C0_NormT5_bSwitchFalse.xlsx')['StandDivSFlat'] # 'StandDivSFlat' is to be able to reuse code for this instance
    dfTwCRPSSwitch = pd.read_excel('OutputDataFrames/mMomentsSDiff_RTot300_twCRPS_C0_NormT5_bSwitchTrue.xlsx')['StandDivSFlat'] # 'StandDivSFlat' is to be able to reuse code for this instance
    
    dfDiv = dictDataFrames[sSide][sScoringRule]
    dfDivSwitch = dictDataFramesSwitched[sSide][sScoringRule]

    lLabels = [sScoringRule+'$^\\sharp$', sScoringRule+'$^\\flat$', sScoringRule+'slog', sScoringRule + 'sbar' ]

    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    
    for i in range(len(lKeys)):
        sKey = lKeys[i]
        axs[0].plot(dfDiv['r'], dfDiv[sKey], label=lLabels[i])
        axs[1].plot(dfDivSwitch['r'], dfDivSwitch[sKey], label=lLabels[i])
    
    # Add censored likelihood reference
    iCol = 3
    
    axs[0].plot(dfDiv['r'], dfTwCRPS, label='twCRPS')
    axs[1].plot(dfDiv['r'], dfTwCRPSSwitch, label='twCRPS')
    
    axs[0].plot(dfDiv['r'],dictDataFrames[sSide]['LogS']['StandDivSFlat'], label='LogS$^\\flat$', linestyle='--', color='orange')
    axs[1].plot(dfDivSwitch['r'], dictDataFramesSwitched[sSide]['LogS']['StandDivSFlat'], label='LogS$^\\flat$', linestyle='--', color='orange')
    
    axs[0].set_title('Local Divergence F to G')
    axs[1].set_title('Local Divergence G to F')
    axs[0].set_xlabel('threshold $r$')
    axs[1].set_xlabel('threshold $r$')

    handles, labels = axs[0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.14, hspace=0.45)  # More space below for the legend
    
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.25), ncol=iCol, frameon=True)

    sFileName = 'Figures' + '/' + sFolder + '/' + sReference + '_' + sSide + '_' + sScoringRule + '_' + sCandidates + '_' + sVersion + '.pdf'
    plt.savefig(sFileName, bbox_inches='tight')
    
    plt.close()

