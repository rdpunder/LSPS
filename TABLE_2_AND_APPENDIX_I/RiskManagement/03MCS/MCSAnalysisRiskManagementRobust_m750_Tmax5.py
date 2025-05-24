#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose:
    Analyse MCS results
"""

## Imports

# Fundamentals
import numpy as np  

# Pandas
import pandas as pd

###########################################################  
def main():    
    
    ## Magic numbers
    iTest = 750           
    iK = 5
    sStat = 'Tmax'
    
    
    lLevels = [0.90]               
    lRq = [0.01, 0.05, 0.10, 0.15, 0.20, 0.25] 
    lH = [1,5]
    bTails = True
    lMethodsInputOrder = [
        'RGARCH-Std(nu)',
        'TGARCH-Std(nu)',
        'GARCH-Std(nu)',
        'RGARCH-Normal',
        'TGARCH-Normal',
        'GARCH-Normal'
    ]
    lMethodsOutputOrder = [
        'RGARCH-t',
        'TGARCH-t',
        'GARCH-t',
        'RGARCH-N',
        'TGARCH-N',
        'GARCH-N'
    ]
    
    # Index mapping: Input indices to output indices
    lIndexMapping = [0, 1, 2, 3, 4, 5]
    
    ###########################################################  
    ## Data
    
    # Define file path components
    sBasePath = 'MCSTables/'
    sFilePrefix = 'RiskManL_MCSTableC'
    sFileSuffix = f'_{sStat}_iK{iK}_iTest{iTest}_q{{}}.xlsx'
    sTable2 = 'TableI3'
    
    # Dynamically create file paths
    lFilePaths = [f"{sBasePath}{sFilePrefix}{sFileSuffix.format(int(dRq*100))}" for dRq in lRq]

    # Combine all files into one DataFrame
    lDataFrames = []
    for dRq, sFilePath in zip(lRq, lFilePaths):
        # Load the data
        dfTemp = pd.read_excel(sFilePath)
        
        # Get all columns for h=1 and h=5
        lColumnsH1 = [col for col in dfTemp.columns if not col.endswith('5')]
        lColumnsH5 = [col for col in dfTemp.columns if col.endswith('5')]
        
        # For h=5, strip the suffix to standardize column names
        dfH1 = dfTemp[lColumnsH1].copy()
        dfH5 = dfTemp[lColumnsH5].copy()
        dfH5.columns = [col.rstrip('h5') for col in dfH5.columns]
        
        # Add hierarchical information
        for iH, df in zip([1, 5], [dfH1, dfH5]):
            df['q'] = dRq
            df['h'] = iH
            
            # Reorder rows based on lIndexMapping
            df = df.iloc[lIndexMapping].reset_index(drop=True)
            
            # Add reordered method names
            df['Method'] = lMethodsOutputOrder
            lDataFrames.append(df)
    
    # Concatenate all data into one DataFrame
    dfMCSPvals = pd.concat(lDataFrames, ignore_index=True)
    
    # Replace NaN values with 1
    dfMCSPvals.fillna(1, inplace=True)
    
    # Reorder columns so q, h, and Method come first
    lCols = ['q', 'h', 'Method'] + [col for col in dfMCSPvals.columns if col not in ['q', 'h', 'Method']]
    dfMCSPvals = dfMCSPvals[lCols]
    
    # Set hierarchical index
    dfMCSPvals.set_index(['q', 'h', 'Method'], inplace=True)
    
    # Save the final DataFrame to an Excel file
    dfMCSPvals2dec = dfMCSPvals.round(2).applymap(lambda x: f"{x:.2f}" if isinstance(x, (float, int)) else x)
    dfMCSPvals4dec = dfMCSPvals.round(4).applymap(lambda x: f"{x:.4f}" if isinstance(x, (float, int)) else x)
   
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
            dfCardinality[sRule] = dfMCSPvals[sRule].groupby(['h','q']).apply(lambda x: (x >= 1-dLevel).sum())
        for h in lH:
            dfCardinalityH = dfCardinality.xs(key=h, level='h', drop_level=False)
            lResults = []
            for lSharpVersion in lScoringRulesSharpVersions:
                lResults.append(round(np.mean(dfCardinalityH[lScoringRulesFlat].values <= dfCardinalityH[lSharpVersion].values)*100))
                lResults.append(round(np.mean(dfCardinalityH[lScoringRulesFlat].values < dfCardinalityH[lSharpVersion].values)*100))
                lResults.append(round(np.mean(dfCardinalityH[lSharpVersion].values / dfCardinalityH[lScoringRulesFlat].values),2))
            dfOut.loc[h] = lResults
        dfOut.to_excel(sTable2+'/'+f'MCSCardinality_{dLevel:.2f}_Summary_Tails'+str(int(bTails))+'_'+sStat+str(iK)+'_iTest'+str(int(iTest))+'.xlsx')
       
    
###########################################################
### Start main
if __name__ == "__main__":
    main()
