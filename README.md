# Localizing Strictly Proper Scoring Rules

This repository provides the data and code to reproduce all the empirical and Monte Carlo simulation results contained in the paper **Localizing Strictly Proper Scoring Rules**.

The repository contains 6 folders. First, each of the four empiricial applications: Risk Management, Multivariate Risk Management, Inflation and Climate has its own folder. The other two folders belong to the Monte Carlo study and the numerical calculation of Example 6.

For every empirical application, the MCS p-values reported in Appendix I, underlying Table 2, are obtained by following three steps:
* **01DensityForecasts**: Preprocess the data and estimate parameters using application specific forecast methods to construct density forecasts.
* **02Scores**: Compute the scores from each forecast method of step 1 under the scoring rules for which MCS p-values have to be calculated.
* **03MCS**: Apply the MCS procedure, relying on the R package MCS by Bernardi and Catania (2018), to the scores from step 2 and calculate the percentages and ratios reported in Table 2. 
 
Specific details per application are given below. The computation time of individual files can found in  `ComputationTimePerFile.xlsx`.


## RISK MANAGEMENT
Directory: `RISK_MANAGEMENT_Table_2_Appendix_I/`

### Data
Download the Realized Volatility measure via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting trades of `SPDR S & P 500 E T F TRUST` (symbol=`SPY`, PN=`843398`) for the period `All`. The downloaded file contains two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. Save the data as `.csv` file `RealisedVolatilityFullPeriodTrade.csv` in the subdirectory `01DensityForecasts/Data`. The S&P500 series (ticker: `SPY`) is downloaded from [Yahoo Finance](https://finance.yahoo.com/quote/SPY/) through the `yfinance` module by running the Python script `01DensityForecasts/Data/EmpiricalDataRiskMan.py`, which also transforms the prices into log returns and merges the S&P500 data with the deannualized realized measure.  

### Code
The code is organized in the three folders introduced above.
1. Navigate to the directory `01DensityForecasts`. The main script is `RiskManMain.py`, for which a sample bashscript `S1_RiskManMain.sh` is provided to facilitate parallel computation. The main script depends on `RiskManBasis.py`, which implements fundamental routines including the rolling window estimation procedure, and `TGARCHmodel.py`, which contains functions for the individual forecast methods. Executing `RiskManMain.py` produces the parameter estimates of the density forecasts based on the observations in the `Data` folder and stores them as `.npy` files in the `mParamsDF` subdirectory. After completion, the folders `Data` and `mParamsDF` should be manually copied to `02Scores`.

2. Navigate to the directory `02Scores`. The main script `RiskManScoreCalcMain.py` depends on the functions in `ScoreBasis.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of `RiskManScoreCalcMain.py`, e.g. using the sample bash script `S1_RiskManScores.sh`, produces the scores of the density forecasts built on the parameters in `mParamsDF` and the associated observations in `Data` and saves them as `.npy` files into the folder `mScores`, which should be manually copied to `03MCS` upon completion.
 
3. Navigate to the directory `03MCS`. Running the R script `MCSTables_RiskMan.R` produces the MCS p-values based on the scores in `mScores` and saves them as `.xlsx` files in the subdirectory `MCSTables`. 

### Output
Navigate to the directory `03MCS`. Run the script `MCSAnalysisRiskManagement.py`. The MCS results in the directory `MCSTables` will be translated into the table with MCS p-values in **Table I2** and the summary values in the Risk Management Panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

### Robustness Analysis
The remaining files in the three folders contribute to the robustness analysis with respect to the choice of the test statistic and the length of the estimation window (*m*). For reproduction, follow:
1. Redo step 1 for the alternative window lengths *m=750* and *m=1250* by running `01DensityForecasts/RiskManMainTest750.py` and `01DensityForecasts/RiskManMainTest1250.py` (e.g. by using the bash scripts `S1_RiskManScoresTest750.sh` and `S1_RiskManScoresTest1250.sh`, respectively).
2. Redo step 2 for the alternative window lengths *m=750* and *m=1250* by running `02Scores/RiskManScoreCalcMainTest750.py` and `02Scores/RiskManScoreCalcMainTest1250.py` (e.g. by using the bash scripts `S1_RiskManScoresTest750.sh` and `S1_RiskManScoresTest1250.sh`, respectively).
3. Similar to step 3 above, run `03MCS/MCSTables_RiskManRobust.R`, `03MCS/MCSTables_RiskManRobust750.R` and `03MCS/MCSTables_RiskManRobust1250.R`.
  
The output of the robustness analysis is generated by navigating to the directory `03MCS` and then running the scripts `MCSAnalysisRiskManagementRobust_m750_Tmax5.py`, `MCSAnalysisRiskManagementRobust_m750_TR5.py`, `MCSAnalysisRiskManagementRobust_m1000_Tmax20.py`, `MCSAnalysisRiskManagementRobust_m1000_TR20.py`, `MCSAnalysisRiskManagementRobust_m1250_Tmax5.py`, `MCSAnalysisRiskManagementRobust_m1250_TR5.py`. The MCS tables in the directory `MCSTables` will be translated into the corresponding rows of **Table I3**.


## MULTIVARIATE RISK MANAGEMENT
Directory: `MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/`

### Data
Download the Realized Volatility measures via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting, for the period `All`, trades of (i) `SELECT SECTOR SPDR TRUST` (symbol=`XLE`, PN=`86454`), and (ii) trades of `SELECT SECTOR SPDR TRUST` (symbol=`XLF`, PN=`86455`) for the period `All` The downloaded files contains two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. Save the data as the `.csv` files `RVFullPeriodXLETrade.csv` and `RVFullPeriodXLFTrade.csv` in the folder `01DensityForecasts/Data`. The series Energy Select Sector SPDR Fund (ticker: `XLE`) and Financial Select Sector SPDR (ticker: `XLF`) are downloaded from [Yahoo Finance](https://finance.yahoo.com/quote/XLE/) through the `yfinance` module by running the Python script `01DensityForecasts/Data/EmpiricalDataMultivariateRiskMan.py`, which also transforms the prices into log returns and merges the XLE and XLF data with the corresponding deannualized realized measure.  

### Code
The code follows the same structure as above.
1. Navigate to the directory `01DensityForecasts`. The main script is `RiskManMainMV.py`, for which a sample bashscript `S1_RiskManBivariate.sh` is provided to facilitate parallel computation. The main script relies on `RiskManBasisMV.py`, which implements fundamental routines including the rolling window estimation procedure, and on `DCCmodel.py` together with `TGARCHmodel.py` and `BivariateT.py`, which implement the individual forecast methods. Running `RiskManMainMV.py` generates parameter estimates of the density forecasts based on the observations in the `Data` folder and stores them as `.npy` files in the `mParamsDF` subdirectory. After completion, the folders `Data` and `mParamsDF` should be manually copied to `02Scores`.

2. Navigate to the directory `02Scores`. Separate main scripts for the indicator product and logistic product weight function are included as `RiskManMainMVIndProd.py` and `RiskManMainMVLogProd3.py`, respectively, with example bash scripts `S6_RiskManScoresMVIndProd.sh` and `S6_RiskManScoresMVLogProd3.sh`. The main scripts depend on the functions in `RiskManBasisMV.py`, `BivariateT.py`, `ScoringRulesMV.py` and `WeightfunctionsMV.py`, including fundamental supporting functions, a custom function for the bivariate t distribution, scoring rules and weight functions, respectively. Execution of the main scripts, e.g. using the sample bash script `S1_RiskManScores.sh`, produces the scores of the density forecasts built on the parameters in `mParamsDF` and the associated observations in `Data` and saves them as `.npy` files into the folder `mScores`, which should be manually copied to `03MCS` upon completion.
 
3. Navigate to the directory `03MCS`. We use the same split as in step 2 per weight function. Running the R scripts `RiskManMCSBivariateIndProd.R` and `RiskManMCSBivariateLogProd3.R` produces the MCS p-values based on the scores in `mScores` and saves them as `.xlsx` files in the subdirectory `MCSTables`. 

### Output
Navigate to the directory `03MCS`. Run the scripts `MCSAnalysisRiskManIndProd.py` and `MCSAnalysisRiskManLogProd3.py`. The MCS results in the directory `MCSTables` will be translated into the table with MCS p-values in **Table I4** (for the weight function IndicatorProduct) and **Table I5** (for the weight function LogisticProduct) and the summary values in the Risk Management panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.


## INFLATION
Directory: `INFLATION_Table_2_Appendix_I/`

### Data
The inflation data is sourced from the code provided by Medeiros et al. (2021) and stored as `01DensityForecasts/Data/Data.Rdata`. Run the R script `01_data_acc.R` to construct the accumulated inflation for each horizon. The resulting datasets are saved in the `01DensityForecasts/Data` directory as both `mYAcc.Rdata` and `mYAcc.npy` file. In addition, the last 180 months of observations are saved separately in the same formats, as the files `YAccOut.Rdata` and `mYAccOut.npy`. 

### Code
The code is organized in the three folders introduced above.
1. Navigate to the directory `01DensityForecasts`. The procedure of construction the mean of the density forecasts is relies on Medeiros et al. (2021), hence each individual forecast method now has its own R script:
    * AR model: `02A_call_model_ar.R`
    * Bagging: `02B_call_model_bagging`
    * Complete Subset Regression: `02C_call_model_csr.R`
    * LASSO: `02D_call_model_lasso.R`
    * Random Forest: `02E_call_model_rf.R`
    * Random Walk: `02F_call_model_rw.R`
The individual scripts rely on the supporting functions in the files `Functions/functions.R` and `rolling_window_tpnorm.R`, calculate the (parameters of) the density forecasts and save them as both `.rda` and `.npy` files in the `mParamsDF` directory.
After completion, the folders `Data` and `mParamsDF` should be manually copied to `02Scores`.

2. Navigate to the directory `02Scores`. We have devided the computation tasks per horizon (*h=6* and *h=24*) and weight function (tails (*T*) and center (*C*)), corresponding to the main scripts
    * `InflationScoreCalcMain_T_h6.py` (`S1_InflationScores_T_h6.sh`)
    * `InflationScoreCalcMain_T_h24.py` (`S1_InflationScores_T_h24.sh`)
    * `InflationScoreCalcMain_C_h6.py` (`S1_InflationScores_C_h6.sh`)
    * `InflationScoreCalcMain_C_h24.py` (`S1_InflationScores_C_h24.sh`)
   with respective example bash scripts in between brackets. The main scripts depend  which depend on the functions in `ScoreBasisInflation.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of a main script produces the scores of the density forecasts built on the parameters in `mParamsDF` and the associated observations in `Data` and saves them as `.npy` files into the folder `mScores`, which should be manually copied to `03MCS` upon completion.
 
3. Navigate to the directory `03MCS`. Running the R scripts `MCSTables_InflationTails.R` and `MCSAnalysisInflationCenter.py` produces the MCS p-values for the tails and center indicator weight function based on the scores in `mScores` and saves them as `.xlsx` files in the subdirectory `MCSTables`. 

### Output
Navigate to the directory `03MCS`. Run the scripts `MCSAnalysisInflationTails.py` and `MCSAnalysisRiskManLogProd3.py`. The MCS results in the directory `MCSTables` will be translated into the table with MCS p-values in **Table I6** (for the tails indicator weight function) and **Table I7** (for the center indicator weight function) and the summary values in the Inflation panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

## CLIMATE
Directory: `CLIMATE_Table_2_Appendix_I/`

### Data
The temperature data is downloaded as a `.txt` file directly from [KNMI - Daily Weather Data De Bilt](https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_260.zip), starting on January 1, 1901, and updated regularly. From this file we extract the columns `YYYYMMDD` (=Date), `TG` (=TempAvg, average temperature), `TN` (=TempMin, minimum temperature), `TX` (=TempMax, maximum temperature), and save them to Excel, resulting in `01DensityForecasts/ClimateKNMI_Temp.xlsx`. In `01DensityForecasts/ClimateMain.py` we divide the raw values by ten to convert them to degrees Celcius, and select the required sample period (sStart =`2003-02-01`, sEnd = `2023-01-31`).

### Code

### Output



### TABLE 2 and APPENDIX I: Empirical Studies
In Table 2 of Section 4, we present summary results on differences in Model Confidence Set (MCS) cardinality for α=0.90 and in Table I1 we display the summary results for α=0.75. For every subsection, for every weight function, Appendix I includes a separate Table reporting the underlying MCS p-values. In directory TABLE_2_AND_APPENDIX_I, we have created the following subdirectories to reproduce all empirical results:

#### RiskManagement
- Table 2 panel Sec 4.1, I_L 
- Table I1 panel Sec 4.1, I_L
- Table I2 MCS p-values Sec 4.1, I_L
- Table I3 Robustness checks Sec 4.1, I_L

#### MultivariateRiskManagement
- Table 2 panels Sec 4.1, I_L^2 and Λ_3^2
- Table I1 panels Sec 4.1, I_L^2 and Λ_3^2
- Table I4 MCS p-values Sec 4.1, I_L^2
- Table I5 MCS p-values Sec 4.1, Λ_3^2

#### Inflation
- Table 2 panels Sec 4.2, I_C and I_C^c
- Table I1 panels Sec 4.2, I_C and I_C^c
- Table I6 MCS p-values Sec 4.2, I_L^2
- Table I7 MCS p-values Sec 4.2, Λ_3^2

#### Climate
- Table 2 panels Sec 4.3, I_R and I_C
- Table I1 panels Sec 4.3, I_R and I_C
- Table I8 MCS p-values Sec 4.3, I_R
- Table I9 MCS p-values Sec 4.3, I_C

Each subdirectory has three further subdirectories: *01DensityForecasts*, *02Scores*, *03MCS* and a ReadMe.txt file listing the specific steps per application for reproduction. In the *01DensityForecasts* directories, application-based data is taken as input to produce the (parameters of) the density forecasts, saved into *mParamsDF*. The *02Scores* directories calculate the scores based on the parameters of the density forecasts in *mParamsDF* and the data. In the *03MCS* directories, MCS Tables are calculated per threshold/quantile of the weight function by using the R package MCS developed by Bernardi and Catania (2018). MCS tables for Appendix I and summary results for Table 2 and Table I1 are then generated by a Python wrapper in the *MCSTables* subdirectories. 

### APPENDIX G: Monte Carlo Study
The Monte Carlo Study detailed in Appendix G includes a size experiment and three power experiments: Normal vs. Student-t(5) left-tail, Normal vs. Student-t(5) center and Laplace(-1,1) vs. Laplace(1,1.1). The power studies are supplemented with the analysis of the associated local divergences, for which we include separate folders per experiment. In total, this yields seven subdirectories for the reproduction of the figures in Appendix G, with titles indicating which figure is reproduced (e.g. *FIG_G1_Size* for the reproduction of Figure G.1). In each subdirectory, a ReadMe.txt file is included with more specific instructions. The common structure of the subdirectories is to first calculate the desired scores (e.g. by running *01SizeMainCalc*) and then generate the plots by a different script (e.g. *02SizeMain_Plot.py*).

### EXAMPLE 6: Mitchell and Weale (2023)
In Example 6 of Section 3.4, we provide a specific example for which the expected score difference based on the weighted scoring rule by Mitchell and Weale (2023) is negative for α>α_0. The aim of directory *EXAMPLE_6* is to reproduce the (rounded) number α_0 = 0.052 and to graphically verify the inequality α>α_0. We refer to the *ReadMe.txt* in this directory for further instructions.

## Requirements
The scripts contained in the directories listed above rely on a set of supporting libraries. To ensure consistency and reproducibility, we used a virtual environment that can be reconstructed by creating a novel virtual environment and installing all required libraries by the provided requirements.txt file using *pip install -r requirements.txt*.

## Other files
- requirements.txt: required libraries and packages
- ComputationTimePerFile.xlsx: Indicating computation time per file
- LICENSE: MIT License.
- README.md: This *README.md* file




