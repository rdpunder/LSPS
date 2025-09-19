#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Generate plots power experiments
"""

# System
import os

# Fundamentals
import numpy as np  

# Visualisation
import matplotlib.pyplot as plt

lFigureSettings= {'figsize':(8,8), 'dpi':70, 'titlefontsize':16, 'axisfontsize':14} 

###########################################################  
def PlotRejRates(dictX, dictRejRates, dictScores, vIdxScores, dictRIdxGridPlots, sSettings=None, iLegCol=3, iStepX=1):
    """
    Purpose
    ----------
    Plot weights for both horizons in two subfigures. Repeat for all focussing types and save save figures as separate files.

    Parameters
    ----------
    dictX :                 dictionary with 'Values' (x-axis values) and 'Label' (LaTeX-formatted label)
    dictRejRates :          dictionary of empirical rejection rates by true model and favored alternative
    dictScores :            dictionary of scoring rules with plotting specifications and labels
    vIdxScores :            array of integers indicating which scoring rules to include
    dictRIdxGridPlots :     dictionary specifying which grid indices to plot per model configuration
    sSettings :             optional string to include in filename when saving figure
    iLegCol :               integer number of columns in legend layout
    iStepX :                integer step size for x-axis tick marks

    Returns
    -------
    Saves 1 figure with iF x 1 subfigures

    """

    
    lKeyTrue = list(dictRejRates.keys())
    lKeySides = list(dictRejRates[lKeyTrue[0]].keys())
    iS, iLenRGrid = dictRejRates[lKeyTrue[0]][lKeySides[0]][0].shape
    lScores = list(dictScores.keys())
    
    vFigSize = (lFigureSettings['figsize'][0], lFigureSettings['figsize'][1])
    sFileLoc = os.getcwd()
    sFileLocation = os.path.join(sFileLoc, 'Figures/') 

    fig, ax = plt.subplots(2, 2, figsize=vFigSize)
   
    for k in range(2):
        for d in range(2):
            for s in vIdxScores:
                vIdx = dictRIdxGridPlots[lKeyTrue[d]][lKeySides[k]]
                ax[k,d].plot(dictX['Values'][vIdx], dictRejRates[lKeyTrue[d]][lKeySides[k]][0][s,][vIdx], c= dictScores[lScores[s]]['Colour'], ls=dictScores[lScores[s]]['Symbol'], label = dictScores[lScores[s]]['Label'])
                ax[k,d].xaxis.set_ticks(np.arange(int(dictX['Values'][vIdx].min().round()), int(dictX['Values'][vIdx].max().round()+1), iStepX))
            ax[k,d].set_xlabel(dictX['Label'])
            ax[k,d].set_ylabel('rejection rate')
            ax[k,d].yaxis.set_ticks(np.arange(0, 1.2, 0.2))
            if lKeySides[k] == 'FavF':
                if lKeyTrue[d] == 'TrueF':
                    ax[k,d].set_title('${\\rm H}_1: {\\rm E}_{f_t} (D_{t+1}) > 0$')
                else:
                    ax[k,d].set_title('${\\rm H}_1: {\\rm E}_{g_t} (D_{t+1}) > 0$')
            elif lKeySides[k] == 'FavG':
                if lKeyTrue[d] == 'TrueF':
                    ax[k,d].set_title('${\\rm H}_1: {\\rm E}_{f_t} (D_{t+1}) < 0$')
                else:
                    ax[k,d].set_title('${\\rm H}_1: {\\rm E}_{g_t} (D_{t+1}) < 0$')
            else:
                print('Key Error: keySide not found')
                
   
    handles, labels = ax[0,0].get_legend_handles_labels()
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.09+0.05*int(np.ceil(vIdxScores.size/iLegCol)), hspace=0.45)     
    fig.legend(handles, labels, loc='lower center', ncol=iLegCol, frameon=True) # mode='expand'
        
    #plt.show()    
    sName = 'PlotRejRates' + sSettings + '.pdf'
    plt.savefig(sFileLocation + sName, bbox_inches='tight')
    plt.show(block=False)

