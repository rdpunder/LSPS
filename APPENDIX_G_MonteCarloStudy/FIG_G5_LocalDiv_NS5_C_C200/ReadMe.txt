Steps to reproduce Figure G.5: Standardized local divergences Normal-Student-t_5 (c=200)
------
1. Run 01_A_DivergencesMain_Calc.py. A sample bash script, S1_Divergences_A.sh, is provided for your convenience. The script computes the standardized divergences and saves the resulting .xlsx file in the OutputDataFrames directory. 
2. Run 01_B_DivergencesMain_Calc.py. See 1., script runs for switched order of distributions. See S1_Divergences_B.sh for an example bash script.
3. Run 02DivergencesMain_Plot.py. The file generates five figures saved in the folder Figures.

Other files in this directory
------
ScoringRulesLocalDiv.py: Functions for scoring rules
DivergencesPlot.py: Functions for plotting
DivergencesBasis.py: Other required functions

Other remarks
------ 
In case no TeX distribution is available, replace True by False in rc('text', usetex=True). 