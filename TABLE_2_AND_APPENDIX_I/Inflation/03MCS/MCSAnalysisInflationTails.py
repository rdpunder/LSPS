#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Analyse MCS results for Climate application
"""

## Imports
import numpy as np
import pandas as pd

###########################################################  
def main():    
    
    ###########################################################  
    ## Magic numbers
    
    lH = [6, 24]               # Horizons
    bTails = True              # Indicates tail-related files
    lLevels = [0.90, 0.75]
    lRq = [1, 1.5, 2]
    sQLabel = 'r'
    
    lMethods = [
        'Random Walk',
        'AR',
        'Bagging',
        'CSR',
        'LASSO',
        'Random Forest',
    ]
    
    ###########################################################  
    ## Data
    
    # Define file path components
    sBasePath = 'MCSTables/'
    sFilePrefix = 'Inflation_MCSTable_h'
    sFileSuffix = '_TR_iK5_iTest48_q{}_tails{}.xlsx'
    sAppendixTable = 'TableI7'
    sTable2 = 'Table2_InflationPanel'
    sTableI1 = 'TableI1_InflationPanel'
    
    # Combine all files into one DataFrame
    lDataFrames = []
    for dRq in lRq:
        for iH in lH:
            # Construct the file path dynamically
            sFilePath = f"{sBasePath}{sFilePrefix}{iH}{sFileSuffix.format(int(dRq*100), int(bTails))}"
            
            # Load the data
            dfTemp = pd.read_excel(sFilePath)
            
            # Add hierarchical information
            dfTemp[sQLabel] = dRq
            dfTemp['h'] = iH
            dfTemp['Method'] = lMethods  
            
            # Append to the list
            lDataFrames.append(dfTemp)
    
    # Concatenate all data into one DataFrame
    dfMCSPvals = pd.concat(lDataFrames, ignore_index=True)
    
    # Replace NaN values with 1
    dfMCSPvals.fillna(1, inplace=True)
    
    # Reorder columns so q, h, and Method come first
    lCols = [sQLabel, 'h', 'Method'] + [col for col in dfMCSPvals.columns if col not in [sQLabel, 'h', 'Method']]
    dfMCSPvals = dfMCSPvals[lCols]
    
    # Set hierarchical index
    dfMCSPvals.set_index([sQLabel, 'h', 'Method'], inplace=True)
    
    
    # Save the final DataFrame to an Excel file
    dfMCSPvals2dec = dfMCSPvals.round(2).applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x)
    dfMCSPvals4dec = dfMCSPvals.round(4).applymap(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)
    dfMCSPvals2dec.to_excel(sAppendixTable+'/MCSResults_Tails'+str(int(bTails))+'_Dec2.xlsx')
    dfMCSPvals4dec.to_excel(sAppendixTable+'/MCSResults_Tails'+str(int(bTails))+'_Dec4.xlsx')
    
    ###########################################################  
    ## Compute cardinalities for levels

    # Filter rows where p-values are above the given thresholds
    lCardinalities = []
    lScoringRules = dfMCSPvals.columns
    dfCardinality = pd.DataFrame()
    lScoringRulesFlat = ['LogSflat', 'QSflat', 'SphSflat', 'CRPSflat']
    lScoringRulesSharp = ['LogSsharp', 'QSsharp', 'SphSsharp', 'CRPSsharp']
    lScoringRulesSharpslog = ['LogSsharpslog', 'QSsharpslog', 'SphSsharpslog', 'CRPSslog']
    lScoringRulesSharpsbar = ['LogSsharpsbar', 'QSsharpsbar', 'SphSsharpsbar', 'CRPSsbar']
    lScoringRulesSharpVersions = [lScoringRulesSharp, lScoringRulesSharpsbar, lScoringRulesSharpslog]
    
    for dLevel in lLevels:
        dfOut = pd.DataFrame(data=None, columns=['<=', '<', 'Sharp/Flat', '<=', '<', 'Sharpsbar/Flat', '<=', '<', 'Sharpslog/Flat'], index=lH)
        for sRule in lScoringRules:
            dfCardinality[sRule] = dfMCSPvals[sRule].groupby(['h',sQLabel]).apply(lambda x: (x >= 1-dLevel).sum())
        for h in lH:
            dfCardinalityH = dfCardinality.xs(key=h, level='h', drop_level=False)
            lResults = []
            for lSharpVersion in lScoringRulesSharpVersions:
                lResults.append(np.round(np.mean(dfCardinalityH[lScoringRulesFlat].values <= dfCardinalityH[lSharpVersion].values)*100).astype(int))
                lResults.append(np.round(np.mean(dfCardinalityH[lScoringRulesFlat].values < dfCardinalityH[lSharpVersion].values)*100).astype(int))
                lResults.append(np.round(np.mean(dfCardinalityH[lSharpVersion].values / dfCardinalityH[lScoringRulesFlat].values),2))
            dfOut.loc[h] = lResults
        if dLevel == 0.9:
            dfOut.to_excel(sTable2+'/'+f'MCSCardinality_{dLevel:.2f}_Summary_Tails'+str(int(bTails))+'.xlsx')
        else:
            dfOut.to_excel(sTableI1+'/'+f'MCSCardinality_{dLevel:.2f}_Summary_Tails'+str(int(bTails))+'.xlsx')

###########################################################
### Start main
if __name__ == "__main__":
    output_file = main()
