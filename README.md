# Localizing Strictly Proper Scoring Rules

This repository provides the data and code to reproduce all the empirical and Monte Carlo simulation results contained in the paper **Localizing Strictly Proper Scoring Rules**.

The repository contains 6 folders. First, each of the four empirical applications: Risk Management, Multivariate Risk Management, Inflation and Climate has its own folder. The other two folders contain the Monte Carlo study and the numerical calculation of Example 6.

For every empirical application, the MCS p-values reported in Appendix I, underlying Table 2, are obtained by following three steps:
* **01DensityForecasts**: Preprocess the data and estimate parameters using application specific forecast methods to construct density forecasts.
* **02Scores**: Compute the scores from each forecast method of step 1 under the scoring rules for which MCS p-values have to be calculated.
* **03MCS**: Apply the MCS procedure, relying on the R package MCS by Bernardi and Catania (2018), to the scores from step 2 and calculate the percentages and ratios reported in Table 2. 
 
Specific details per application are given below. The computation time of individual files can be found in  `ComputationTimePerFile.xlsx`. The scripts contained in the directories listed above rely on a set of supporting libraries. To ensure consistency and reproducibility, we used a virtual environment that can be reconstructed by creating a novel virtual environment and installing all required libraries by the provided `requirements.txt` file using `pip install -r requirements.txt`.


## RISK MANAGEMENT
Directory: `RISK_MANAGEMENT_Table_2_Appendix_I/`

### Data
Start with: The data can be found in folder etc.
Download the Realized Volatility measure via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting trades of `SPDR S & P 500 E T F TRUST` (symbol=`SPY`, PN=`843398`) for the period `All`. The downloaded file contains two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. Save the data as `.csv` file `RealisedVolatilityFullPeriodTrade.csv` in the subdirectory `01DensityForecasts/Data`. The S&P500 series (ticker: `SPY`) is downloaded from [Yahoo Finance](https://finance.yahoo.com/quote/SPY/) through the `yfinance` module by running the Python script `01DensityForecasts/Data/EmpiricalDataRiskMan.py`, which also transforms the prices into log returns and merges the S&P500 data with the deannualized realized measure.  

### Code
The code is organized in the three folders introduced above.
1. Navigate to the directory `01DensityForecasts`. The main script is `RiskManMain.py`, for which a sample bash script `S1_RiskManMain.sh` is provided to facilitate parallel computation. The main script depends on `RiskManBasis.py`, which implements fundamental routines including the rolling window estimation procedure, and `TGARCHmodel.py`, which contains functions for the individual forecast methods. Executing `RiskManMain.py` produces the parameter estimates of the density forecasts based on the observations in the `Data` folder and stores them as `.npy` files in the `mParamsDF` subdirectory. After completion, the folders `Data` and `mParamsDF` should be manually copied to `02Scores`.

2. Navigate to the directory `02Scores`. The main script `RiskManScoreCalcMain.py` depends on the functions in `ScoreBasis.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of `RiskManScoreCalcMain.py`, e.g. using the sample bash script `S1_RiskManScores.sh`, produces the scores of the density forecasts built on the parameters in `mParamsDF` and the associated observations in `Data` and saves them as `.npy` files into the folder `mScores`, which should be manually copied to `03MCS` upon completion.
 
3. Navigate to the directory `03MCS`. Running the R script `MCSTables_RiskMan.R` produces the MCS p-values based on the scores in `mScores` and saves them as `.xlsx` files in the subdirectory `MCSTables`. 

### Output
* **Table 2** and **Table I1**, Risk Management Panel, run `03MCS/MCSAnalysisRiskManagement.py`.

Navigate to the directory `03MCS`. Run the script `MCSAnalysisRiskManagement.py`. Running the script translates the MCS p-values in the directory `MCSTables` into the table with MCS p-values in **Table I2** and the summary values in the Risk Management Panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

### Robustness Analysis
The remaining files in the three folders contribute to the robustness analysis with respect to the choice of the test statistic and the length of the estimation window (*m*). For reproduction, follow:
1. Redo step 1 for the alternative window lengths *m=750* and *m=1250* by running `01DensityForecasts/RiskManMainTest750.py` and `01DensityForecasts/RiskManMainTest1250.py` (e.g. by using the bash scripts `S1_RiskManScoresTest750.sh` and `S1_RiskManScoresTest1250.sh`, respectively).
2. Redo step 2 for the alternative window lengths *m=750* and *m=1250* by running `02Scores/RiskManScoreCalcMainTest750.py` and `02Scores/RiskManScoreCalcMainTest1250.py` (e.g. by using the bash scripts `S1_RiskManScoresTest750.sh` and `S1_RiskManScoresTest1250.sh`, respectively).
3. Similar to step 3 above, run `03MCS/MCSTables_RiskManRobust.R`, `03MCS/MCSTables_RiskManRobust750.R` and `03MCS/MCSTables_RiskManRobust1250.R`.
  
The output of the robustness analysis is generated by navigating to the directory `03MCS` and then running the scripts `MCSAnalysisRiskManagementRobust_m750_Tmax5.py`, `MCSAnalysisRiskManagementRobust_m750_TR5.py`, `MCSAnalysisRiskManagementRobust_m1000_Tmax20.py`, `MCSAnalysisRiskManagementRobust_m1000_TR20.py`, `MCSAnalysisRiskManagementRobust_m1250_Tmax5.py`, `MCSAnalysisRiskManagementRobust_m1250_TR5.py`. The MCS tables in the directory `MCSTables` will be translated into the corresponding rows of **Table I3**.


## MULTIVARIATE RISK MANAGEMENT
Directory: `MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/`

### Data
The data is located in the folder [01DensityForecasts/Data](MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data), consisting of the files `XLEandRealVolXiu.csv` and `XLFandRealVolXiu.csv` containing the log returns and realized measures of the ETFs Energy Select Sector SPDR Fund (ticker: `XLE`) and Financial Select Sector SPDR (ticker: `XLF`), respectively.

To reconstruct the data files, navigate to the folder [01DensityForecasts/Data/DataConstruction](MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction). Download the realized volatility measures via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting, for the period `All`, trades of (i) `SELECT SECTOR SPDR TRUST` (symbol=`XLE`, PN=`86454`), and (ii) trades of `SELECT SECTOR SPDR TRUST` (symbol=`XLF`, PN=`86455`) for the period `All`. The downloaded files contain two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. Save the data as the `.csv` files `RVFullPeriodXLETrade.csv` and `RVFullPeriodXLFTrade.csv` in the folder [01DensityForecasts/Data/DataConstruction](MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction). The series Energy Select Sector SPDR Fund (ticker: `XLE`) and Financial Select Sector SPDR (ticker: `XLF`) are downloaded from [Yahoo Finance](https://finance.yahoo.com/quote/XLE/) through the `yfinance` module by running the Python script `EmpiricalDataMultivariateRiskMan.py` inside the [DataConstruction](MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction) folder., which also transforms the prices into log returns and merges the XLE and XLF data with the corresponding deannualized realized measure.  

### Code
The code follows the same structure as above.
1. Navigate to the directory `01DensityForecasts`. The main script is `RiskManMainMV.py`, for which a sample bash script `S1_RiskManBivariate.sh` is provided to facilitate parallel computation. The main script relies on `RiskManBasisMV.py`, which implements fundamental routines including the rolling window estimation procedure, and on `DCCmodel.py` together with `TGARCHmodel.py` and `BivariateT.py`, which implement the individual forecast methods. Running `RiskManMainMV.py` generates parameter estimates of the density forecasts based on the observations in the `Data` folder and stores them as `.npy` files in the `mParamsDF` subdirectory. After completion, the folders `Data` and `mParamsDF` should be manually copied to `02Scores`.

2. Navigate to the directory `02Scores`. Separate main scripts for the indicator product and logistic product weight function are included as `RiskManMainMVIndProd.py` and `RiskManMainMVLogProd3.py`, respectively, with example bash scripts `S6_RiskManScoresMVIndProd.sh` and `S6_RiskManScoresMVLogProd3.sh`. The main scripts depend on the functions in `RiskManBasisMV.py`, `BivariateT.py`, `ScoringRulesMV.py` and `WeightfunctionsMV.py`, including fundamental supporting functions, a custom function for the bivariate t distribution, scoring rules and weight functions, respectively. Execution of the main scripts, e.g. using the sample bash script `S1_RiskManScores.sh`, produces the scores of the density forecasts built on the parameters in `mParamsDF` and the associated observations in `Data` and saves them as `.npy` files into the folder `mScores`, which should be manually copied to `03MCS` upon completion.
 
3. Navigate to the directory `03MCS`. We use the same split as in step 2 per weight function. Running the R scripts `RiskManMCSBivariateIndProd.R` and `RiskManMCSBivariateLogProd3.R` produces the MCS p-values based on the scores in `mScores` and saves them as `.xlsx` files in the subdirectory `MCSTables`. 

### Output
Navigate to the directory `03MCS`. Run the scripts `MCSAnalysisRiskManIndProd.py` and `MCSAnalysisRiskManLogProd3.py`. The MCS results in the directory `MCSTables` will be translated into the table with MCS p-values in **Table I4** (for the weight function IndicatorProduct) and **Table I5** (for the weight function LogisticProduct) and the summary values in the Risk Management panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.


## INFLATION
Directory: `INFLATION_Table_2_Appendix_I/`

### Data
For the data construction, we adopt the procedure of Medeiros et al. (2021), made available through the GitHub repository [gabrielrvsc/ForecastingInflation](https://github.com/gabrielrvsc/ForecastingInflation/tree/main).  Download the current monthly vintage `current.csv` of the FRED-MD monthly data via the [McCracken Database](https://www.stlouisfed.org/research/economists/mccracken/fred-databases). Run the R script `01_get_fred_data.R` to convert the raw file into `data.rda`, which will be stored in the `01DensityForecasts/Data` folder. Subsequently run the R script `01_data_acc.R` to construct the accumulated inflation based on `CPIAUCSL` for each horizon. The resulting datasets are saved in the `01DensityForecasts/Data` directory as both `mYAcc.Rdata` and `mYAcc.npy` files. In addition, the last 180 months of observations are saved separately in the same formats, as the files `YAccOut.Rdata` and `mYAccOut.npy`. 

### Code
The code is organized in the three folders introduced above.
1. Navigate to the directory `01DensityForecasts`. The construction of the mean of the density forecasts also strongly relies on Medeiros et al. (2021), hence each individual forecast method now has its own R script:
    * AR model: `03A_call_model_ar.R`
    * Bagging: `03B_call_model_bagging`
    * Complete Subset Regression: `03C_call_model_csr.R`
    * LASSO: `03D_call_model_lasso.R`
    * Random Forest: `03E_call_model_rf.R`
    * Random Walk: `03F_call_model_rw.R`
The individual scripts rely on the supporting functions in the files `Functions/functions.R` and `rolling_window_tpnorm.R`, calculate the (parameters of) the density forecasts and save them as both `.rda` and `.npy` files in the `mParamsDF` directory.
After completion, the folders `Data` and `mParamsDF` should be manually copied to `02Scores`.

2. Navigate to the directory `02Scores`. We have divided the computation tasks per horizon (*h=6* and *h=24*) and weight function (tails (*T*) and center (*C*)), corresponding to the main scripts
    * `InflationScoreCalcMain_T_h6.py` (`S1_InflationScores_T_h6.sh`)
    * `InflationScoreCalcMain_T_h24.py` (`S1_InflationScores_T_h24.sh`)
    * `InflationScoreCalcMain_C_h6.py` (`S1_InflationScores_C_h6.sh`)
    * `InflationScoreCalcMain_C_h24.py` (`S1_InflationScores_C_h24.sh`)
   with respective example bash scripts in between brackets. The main scripts depend on the functions in `ScoreBasisInflation.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of a main script produces the scores of the density forecasts built on the parameters in `mParamsDF` and the associated observations in `Data` and saves them as `.npy` files into the folder `mScores`, which should be manually copied to `03MCS` upon completion.
 
3. Navigate to the directory `03MCS`. Running the R scripts `MCSAnalysisInflationCenter.py` and `MCSTables_InflationTails.R` produces the MCS p-values for the tails and center indicator weight function based on the scores in `mScores` and saves them as `.xlsx` files in the subdirectory `MCSTables`. 

### Output
Navigate to the directory `03MCS`. Run the scripts `MCSAnalysisInflationCenter.py` and `MCSAnalysisInflationTails.py` . The MCS results in the directory `MCSTables` will be translated into the table with MCS p-values in **Table I6** (for the center indicator weight function) and **Table I7** (for the tails indicator weight function) and the summary values in the Inflation panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

## CLIMATE
Directory: `CLIMATE_Table_2_Appendix_I/`

### Data
The temperature data is downloaded as a `.txt` file directly from [KNMI - Daily Weather Data De Bilt](https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_260.zip), starting on January 1, 1901, and updated regularly. From this file we extract the columns `YYYYMMDD` (=Date), `TG` (=TempAvg, average temperature), `TN` (=TempMin, minimum temperature), `TX` (=TempMax, maximum temperature), and save them to Excel, resulting in `01DensityForecasts/ClimateKNMI_Temp.xlsx`. In `01DensityForecasts/ClimateMain.py` we divide the raw values by ten to convert them to degrees Celsius, and select the required sample period (sStart =`2003-02-01`, sEnd = `2023-01-31`).

### Code
The code is organized in the three folders introduced above.
1. Navigate to the directory `01DensityForecasts`. The main script is `ClimateMain.py`, for which a sample bash script `S1_ClimateMain.sh` is provided to facilitate parallel computation. The main script depends on `ClimateBasis.py`, which implements fundamental routines including the rolling window estimation procedure. For the individual forecast methods, we have split the functions into three files labeled `ClimateLocMeanSinGARCH.py`, `ClimateLocMeanSinGARCHI.py` and `ClimateLocMeanSinGARCHII.py` for the different variance updating equations. Executing `ClimateMain.py` produces the parameter estimates of the density forecasts based on the observations in the `ClimateKNMI_Temp.xlsx` file and stores them as `.npy` files in the `mParamsDF` subdirectory. It additionally saves the out-of-sample observations in separate `.npy` files. After completion, the folder `mParamsDF` should be manually copied to `02Scores`.

2. Navigate to the directory `02Scores`. Separate main scripts for the right tail (*R*) and center (*C*) indicator weight function are included as `ClimateScoreCalcMain_R.py` and `ClimateScoreCalcMain_C.py`, respectively, with example bash scripts `S1_ClimateScores_C.sh` and `S1_ClimateScores_R.sh`. The main scripts depend on the functions in `ScoreBasis.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of the main scripts produces the scores of the density forecasts built on the parameters and out-of-sample observations in `mParamsDF` and saves them as `.npy` files into the folder `mScores`, which should be manually copied to `03MCS` upon completion.
 
3. Navigate to the directory `03MCS`. Running the R scripts `MCSTables_ClimateTails.R` and `MCSTables_ClimateCenter.R` produces the MCS p-values based on the scores in `mScores` and saves them as `.xlsx` files in the subdirectory `MCSTables`. 

### Output
Navigate to the directory `03MCS`. Run the scripts `MCSAnalysisClimate_Tails.py` and `MCSAnalysisClimate_Center.py`. The MCS results in the directory `MCSTables` will be translated into the table with MCS p-values in **Table I8** (for the right tail indicator weight function) and **Table I9** (for the center indicator weight function) and the summary values in the Climate panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

## MONTE CARLO
Directory: `MONTE_CARLO_Appendix_G/`

The Monte Carlo Study detailed in Appendix G includes a size experiment and three power experiments: Normal vs. Student-t(5) left-tail, Normal vs. Student-t(5) center and Laplace(-1,1) vs. Laplace(1,1.1). The power studies are supplemented with the analysis of the associated local divergences, for which we include separate folders per experiment. In total, this yields seven subdirectories for the reproduction of the figures in Appendix G, with titles indicating which figure is reproduced (e.g. *FIG_G1_Size* for the reproduction of Figure G.1). In each subdirectory, a ReadMe.txt file is included with more specific instructions. The common structure of the subdirectories is to first calculate the desired scores (e.g. by running *01SizeMainCalc*) and then generate the plots by a different script (e.g. *02SizeMain_Plot.py*).

## MITCHELL AND WEALE (EXAMPLE 6)
Directory: `MITCHELL_AND_WEALE_Example_6/`

In **Example 6** of Section 3.4, we provide a specific example for which the expected score difference based on the weighted scoring rule by Mitchell and Weale (2023) is negative for *α>α_0*. The aim of the current directory is to reproduce the (rounded) number *α_0 = 0.052* and to graphically verify the inequality *α>α_0*. Run `Example6.py`. The script generates a figure of the expected score differences, saved in the `Figures` subdirectory, and prints the alpha-root of the expected score differences.


## Other files
- requirements.txt: required libraries and packages
- ComputationTimePerFile.xlsx: Indicating computation time per file
- LICENSE: MIT License.
- README.md: This *README.md* file




