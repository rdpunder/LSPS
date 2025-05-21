Steps to reproduce Figure G.1: Size properties of the DM test
------
1. Run 01SizeMain_Calc.py. The computation time for 10,000 replications is large. A sample bash script, S1_SizeMain.sh, is provided for your convenience. The script computes the DM test statistics and saves the resulting .npy file in the DMCalc directory. 
2. Run 02SizeMain_Plot.py. The file generates a figure saved in the folder Figures.

Other files in this directory
------
ScoringRulesMC.py: Functions for scoring rules
WeightFunctionsMC.py: Functions for weight functions
SizePlots.py: Functions for plotting
SizeBasis.py: Other required functions

Other remarks
------ 
In case no TeX distribution is available, replace True by False in rc('text', usetex=True). 