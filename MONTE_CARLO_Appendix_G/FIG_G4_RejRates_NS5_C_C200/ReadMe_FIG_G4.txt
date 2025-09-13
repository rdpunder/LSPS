Steps to reproduce Figure G.4: Rejection rates N(0,1) versus Student-t_5: Center (c=200)
------
1. Run 01PowerMain_NS5_C_C200_Calc_A.py. This script simulates and saves the data in the folder mDataAndWeights.  
2. Run 01PowerMain_NS5_C_C200_Calc_B.py. The computation time for 10,000 replications is large. A sample bash script, S1_PowerMain_NS5_C_C200_A.sh, is provided for your convenience. The script computes the DM test statistics and saves the resulting .npy file in the DMCalc directory. 
3. Run 02PowerMain_NS5_C_C200_Plot.py. The file generates four figures saved in the folder Figures.

Other files in this directory
------
ScoringRulesMC.py: Functions for scoring rules
WeightFunctionsMC.py: Functions for weight functions
PowerPlots.py: Functions for plotting
PowerBasis.py: Other required functions

Other remarks
------ 
In case no TeX distribution is available, replace True by False in rc('text', usetex=True). 