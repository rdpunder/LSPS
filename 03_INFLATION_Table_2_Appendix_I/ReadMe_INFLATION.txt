Steps to reproduce results Inflation Application
------
1. The procedure of construction the mean of the density forecasts is inspired by the steps put forward by Medeiros et al. (2021) First, run 01_get_fred_data.R and 01_data_acc.R. Then, run the individual models 
 A. 03A_call_model_ar.R 
 B. 03B_call_model_bagging.R
 C. 03C_call_model_csr.R
 D. 03D_call_model_lasso.R
 E. 03E_call_model_rf.R
 F. 03F_call_model_rw.R

The individual scripts calculate the (parameters of) the density forecasts and save them as both .rda and .npy files in the mParamsDF folder. After completion, copy the entire folder mParamsDF to 02Scores for step 2.

2. Run 02Scores/InflationScoreCalcMain_C_h6.py, 02Scores/InflationScoreCalcMain_C_h24.py, 02Scores/InflationScoreCalcMain_T_h6.py and run 02Scores/InflationScoreCalcMain_T_h24.py. The scripts calculate the scores based on the density forecast (parameters) in the folder mParamsDF. The scores are save as a .npy file in the folder mScores. After completion, copy the entire folder mScores to 03MCS for step 3.

3. Run 03MCS/MCSTables_InflationCenter.R and run 03MCS/MCSTables_InflationTails.R in R. MCS Tables are calculated based on the R package MCS by Bernardi and Catania. Running the R scripts produces MCS tables, saved as .xlsx files in the folder MCSTables. After completion, run 03MCS/MCSAnalysisInflation_Center.py and MCSAnalysisInflation_Tails.py. The MCS tables in the folder MCSTables will be translated into the tables with MCS p-values, Table I.6 and Table I.7, and the inflation panels in Tables 2 and Table I.1. 
