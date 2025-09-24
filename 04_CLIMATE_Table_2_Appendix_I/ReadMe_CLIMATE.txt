Steps to reproduce results Climate Application
------
1. Run 01DensityForecasts/ClimateMain.py. A sample bash script, S1_ClimateMain.sh, is provided for your convenience. The script calculates the (parameters of) the density forecasts and saves them as a .npy file in the mParamsDF folder. After completion, copy the entire folder mParamsDF to 02Scores for step 2.

2. Run 02Scores/ClimateScoreCalcMain_C.py and run 02Scores/ClimateScoreCalcMain_R.py. The script calculates the scores based on the density forecast (parameters) in the folder mParamsDF. The scores are save as a .npy file in the folder mScores. After completion, copy the entire folder mScores to 03MCS for step 3.

3. Run 03MCS/MCSTables_ClimateCenter.R and run 03MCS/MCSTables_ClimateTails.R in R. MCS Tables are calculated based on the R package MCS by Bernardi and Catania. Running the R scripts produces MCS tables, saved as .xlsx files in the folder MCSTables.  After completion, run 03MCS/MCSAnalysisClimate_Center.py and MCSAnalysisClimate_Tails.py. The MCS tables in the folder MCSTables will be translated into the tables with MCS p-values, Table I.8 and Table I.9, and the climate panels in Table 2 and Table I.1. 
