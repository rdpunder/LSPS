Steps to reproduce results Inflation Application
------
1. The procedure of construction the mean of the density forecasts is inspired by the steps put forward by Medeiros et al. (2021) First, run 01_data_acc.R. Then, run the individual models 
 A. 02A_call_model_ar.R 
 B. 02B_call_model_bagging.R
 C. 02C_call_model_csr.R
 D. 02D_call_model_lasso.R
 E. 02E_call_model_rf.R
 F. 02F_call_model_rw.R

The individual scripts calculate the (parameters of) the density forecasts and save them as both .rda and .npy files in the mParamsDF directory. After completion, copy the entire directory mParamsDF to 02Scores for step 2.

2. Run 02Scores/InflationScoreCalcMain_C_h6.py, 02Scores/InflationScoreCalcMain_C_h24.py, 02Scores/InflationScoreCalcMain_T_h6.py and run 02Scores/InflationScoreCalcMain_T_h24.py. The scripts calculate the scores based on the density forecast (parameters) in the directory mParamsDF. The scores are save as a .npy file in the directory mScores. After completion, copy the entire directory mScores to 03MCS for step 3.

3. Run 03MCS/MCSTables_InflationCenter.R and run 03MCS/MCSTables_InflationTails.R in R. MCS Tables are calculated based on the R package MCS by Bernardi and Catania. Running the R scripts produces MCS tables, saved as .xlsx files in the directory MCSTables. After completion, run 03MCS/MCSAnalysisInflation_Center.py and MCSAnalysisInflation_Tails.py. The MCS tables in the directory MCSTables will be translated into the table with MCS p-values and the climate panels in Tables 2 and I1. 
