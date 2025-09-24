Steps to reproduce results Risk Management Application
------
1. Run 01DensityForecasts/RiskManMain.py. A sample bash script, S1_RiskManMain.sh, is provided for your convenience. The script calculates the (parameters of) the density forecasts and saves them as a .npy file in the mParamsDF folder. After completion, copy the entire folder mParamsDF to 02Scores for step 2.

2. Run 02Scores/RiskManScoreCalcMain.py. The script calculates the scores based on the density forecast (parameters) in the folder mParamsDF. The scores are saved as a .npy file in the folder mScores. After completion, copy the entire folder mScores to 03MCS for step 3.

3. Run 03MCS/MCSTables_RiskMan.R in R. MCS Tables are calculated based on the R package MCS by Bernardi and Catania (2018). Running the R scripts produces MCS tables, saved as .xlsx files in the folder MCSTables.  After completion, run 03MCS/MCSAnalysisRiskManagement.py. The MCS tables in the folder MCSTables will be translated into the table with MCS p-values, Table I.2, and the risk management panel in Table 2 and Table I.1. 

4.1 Redo step 1 for 01DensityForecasts/RiskManMainTest750.py and 01DensityForecasts/RiskManMainTest1250.py.
4.2 Redo step 2 for 02Scores/RiskManScoreCalcMainTest750.py and 02Scores/RiskManScoreCalcMainTest1250.py. 
4.3 Run 03MCS/MCSTables_RiskManRobust.R, 03MCS/MCSTables_RiskManRobust750.R and 03MCS/MCSTables_RiskManRobust1250.R in R. Subsequently, run 03MCS/MCSAnalysisRiskManagementRobust_m750_Tmax5.py, 03MCS/MCSAnalysisRiskManagementRobust_m750_TR5.py, 03MCS/MCSAnalysisRiskManagementRobust_m1000_Tmax20.py, 03MCS/MCSAnalysisRiskManagementRobust_m1000_TR20.py, 03MCS/MCSAnalysisRiskManagementRobust_m1250_Tmax5.py, 03MCS/MCSAnalysisRiskManagementRobust_m1250_TR5.py. The MCS tables in the folder MCSTables will be translated into the corresponding rows of Table I.3.