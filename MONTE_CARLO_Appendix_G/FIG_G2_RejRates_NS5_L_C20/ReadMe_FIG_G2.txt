Steps to reproduce Figure G.2: N(0,1) versus Student-t_5: Left-tail (c=20)
------
1. Run 01PowerMain_NS5_L_C20_Calc.py. The computation time for 10,000 replications is large. A sample bash script, S2_PowerMain_NS5_L_C20.sh, is provided for your convenience. The script computes the DM test statistics and saves the resulting .npy file in the DMCalc directory. 
2. Run 02PowerMain_NS5_L_C20_Plot.py. The file generates four figures saved in the folder Figures.

Other files in this directory
------
ScoringRulesMC.py: Functions for scoring rules
WeightFunctionsMC.py: Functions for weight functions
PowerPlots.py: Functions for plotting
PowerBasis.py: Other required functions

Other remarks
------ 
In case no TeX distribution is available, replace True by False in rc('text', usetex=True). 