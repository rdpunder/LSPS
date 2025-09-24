# Localizing Strictly Proper Scoring Rules

This repository provides the data and code to reproduce all the empirical and Monte Carlo simulation results contained in the paper **Localizing Strictly Proper Scoring Rules**.

The repository contains six folders. Each of the four empirical applications: Risk Management, Multivariate Risk Management, Inflation, and Climate has its own folder. The other two folders contain the Monte Carlo study and the numerical calculations for Example 6.

For every empirical application, the corresponding MCS p-values reported in **Appendix I**, underlying **Table 2**, are obtained by following three steps:
* **01DensityForecasts**: Preprocess the data and estimate parameters using application-specific forecast methods to construct density forecasts.
* **02Scores**: Compute the scores for each forecast method of step 1 under the scoring rules for which MCS p-values are to be calculated.
* **03MCS**: Apply the MCS procedure, relying on the R package MCS by [Bernardi and Catania (2018)](https://doi.org/10.1504/IJCEE.2018.091037), to the scores from step 2 and calculate the percentages and ratios reported in **Table 2**. 
 
Specific details per application are given below. The computation time of individual files can be found in `ComputationTimePerFile.xlsx`. Intermediate results are provided as `.xlsx`, `.npy` and `.Rdata` files.

**Dependencies**: Code is written in Python unless we build on existing R code.
* Install Python dependencies with `pip install -r requirementsLocal.txt` (only freezing local dependencies). The package `mpi4py` requires and MPI implementation for which we use `Open MPI 4.1.5`, which can be installed through `brew install openmpi` on macOS. The main scripts can also be executed sequentially without `mpi4py` by commenting out the corresponding import statements in the main files. In that case, run `python3 mainfile.py` instead of `mpirun -n 16 python3 mainfile.py` (for each of the mainfiles listed below) when using a computer with 16 cores. For parallel computation on a computing cluster we refer to *Remark 1*.
* Install R dependencies by running `InstallPackages.R`.

*Remark 1*. As can be seen from `ComputationTimePerFile.xlsx`, we ran many files on a computing cluster. 
We have provided the associated bash scripts (`.sh`), which serve as an example for other computing clusters. 
For exact reproduction of our results, we ran all files in a virtual environment called `LSPS` (activated in every bash script).
To replicate the virtual environment, create a new virtual environment and run `pip install -r requirements.txt` to install all dependencies.

## RISK MANAGEMENT
Folder: [01_RISK_MANAGEMENT_Table_2_Appendix_I](01_RISK_MANAGEMENT_Table_2_Appendix_I)

### Data
The data are stored in the folder [01DensityForecasts/Data](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data). This folder contains the file `SP500andRealVol1995Xiu.csv`, which includes the log returns and realized measures of the S&P500 (ticker: `SPY`).

To reconstruct the data file, navigate to the folder [01DensityForecasts/Data/DataConstruction](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction). 
Download the realized volatility measures via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting trades of `SPDR S & P 500 E T F TRUST` (symbol=`SPY`, PN=`843398`). 
The downloaded file contains two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. 
Save the data as `.csv` file `RealisedVolatilityFullPeriodTrade.csv` in the [DataConstruction](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction) folder. 
The `SPY` series has been obtained from [Yahoo Finance](https://finance.yahoo.com/quote/SPY/) through the `yfinance` module. 
The script `EmpiricalDataRiskMan.py` located in the [DataConstruction](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction) folder, downloads the price series, transforms it into log returns, and merges it with the corresponding deannualized realized measure. 
The script produces the file `SP500andRealVol1995Xiu.csv`, which can then be copied to the [01DensityForecasts/Data](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data) folder.
For exact replication of results, we recommend using the provided file `SP500andRealVol1995Xiu.csv` in [01DensityForecasts/Data](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data).
Re-downloading the series may lead to small numerical differences (~10e-7) due to real time auto-adjustments for dividends and stock splits and/or other differences related to future changes to the `yfinance` module. 

### Code
The code is organized in the following three folders:
1. [01DensityForecasts](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts):  The main script is `RiskManMain.py`, for which a sample bash script `S1_RiskManMain.sh` is provided to facilitate parallel computation on a computing cluster. The main script depends on `RiskManBasis.py`, which implements fundamental routines including the rolling window estimation procedure, and `TGARCHmodel.py`, which contains functions for the individual forecast methods. Executing `RiskManMain.py` produces the parameter estimates of the density forecasts based on the observations in the [Data](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data) folder and stores them as `.npy` files in the [mParamsDF](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/mParamsDF) folder, together with the calculated empirical quantiles (mRhat) and tail proportion (mGammaHat). After completion, the folders [Data](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data) and [mParamsDF](01_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/mParamsDF) should be manually copied to the [02Scores](01_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores) folder.

2. [02Scores](01_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores): The main script `RiskManScoreCalcMain.py` depends on the functions in `ScoreBasis.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of `RiskManScoreCalcMain.py` (for which one can follow the structure of the sample bash script `S1_RiskManScores.sh` when using a computing cluster) produces the scores of the density forecasts built on the parameters in [mParamsDF](01_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores/mParamsDF) and the associated observations in the [Data](01_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores/Data) folder and saves them as `.npy` files into the folder [mScores](01_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores/mScores), which should be manually copied to the [03MCS](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS) folder upon completion.
 
3. [03MCS](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS): Running the R script `MCSTables_RiskMan.R` produces the MCS p-values based on the scores in [mScores](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS/mScores) and saves them as `.xlsx` files in the folder [MCSTables](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS/MCSTables). 

### Output
* **Table 2** and **Table I1**, first weight function in the Sec. 4.1 panel, and  **Table I2**, run `03MCS/MCSAnalysisRiskManagement.py`.
* **Table I3**, run `03MCS/MCSAnalysisRiskManagement.py`, run `03MCS/MCSAnalysisRiskManagementRobust_m750_Tmax5.py`, `03MCS/MCSAnalysisRiskManagementRobust_m750_TR5.py`, `03MCS/MCSAnalysisRiskManagementRobust_m1000_Tmax20.py`, `03MCS/MCSAnalysisRiskManagementRobust_m1000_TR20.py`, and `03MCS/MCSAnalysisRiskManagementRobust_m1250_Tmax5.py`, `03MCS/MCSAnalysisRiskManagementRobust_m1250_TR5.py`

Navigate to the folder [03MCS](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS). Run the script `MCSAnalysisRiskManagement.py`. Running the script translates the MCS p-values in the folder [MCSTables](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS/MCSTables) into the table with MCS p-values in **Table I2** and the summary values for the first weight function in the Sec. 4.1 panel, in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

The remaining files in the three folders contribute to the robustness analysis with respect to the choice of the test statistic and the length of the estimation window (*m*). For reproduction, follow:
1. Redo step 1 for the alternative window lengths *m=750* and *m=1250* by running `01DensityForecasts/RiskManMainTest750.py` and `01DensityForecasts/RiskManMainTest1250.py` (e.g. by using the bash scripts `S1_RiskManMainTest750.sh` and `S1_RiskManMainTest1250.sh`, respectively).
2. Redo step 2 for the alternative window lengths *m=750* and *m=1250* by running `02Scores/RiskManScoreCalcMainTest750.py` and `02Scores/RiskManScoreCalcMainTest1250.py` (e.g. by using the bash scripts `S1_RiskManScoresTest750.sh` and `S1_RiskManScoresTest1250.sh`, respectively).
3. Similar to step 3 above, run `03MCS/MCSTables_RiskManRobust.R`, `03MCS/MCSTables_RiskManRobust750.R` and `03MCS/MCSTables_RiskManRobust1250.R`.
  
The output of the robustness analysis is generated by navigating to the folder [03MCS](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS) and then running the scripts `MCSAnalysisRiskManagementRobust_m750_Tmax5.py`, `MCSAnalysisRiskManagementRobust_m750_TR5.py`, `MCSAnalysisRiskManagementRobust_m1000_Tmax20.py`, `MCSAnalysisRiskManagementRobust_m1000_TR20.py`, `MCSAnalysisRiskManagementRobust_m1250_Tmax5.py`, `MCSAnalysisRiskManagementRobust_m1250_TR5.py`. By running each script, the MCS tables in the folder [MCSTables](01_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS/MCSTables) will be translated into the corresponding rows of **Table I3**.


## MULTIVARIATE RISK MANAGEMENT
Folder: [02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I)

### Data
The data are located in the folder [01DensityForecasts/Data](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data). This folder contains the files `XLEandRealVolXiu.csv` and `XLFandRealVolXiu.csv`, which include the log returns and realized measures of the ETFs Energy Select Sector SPDR Fund (ticker: `XLE`) and Financial Select Sector SPDR (ticker: `XLF`), respectively.

To reconstruct the data files, navigate to the folder [01DensityForecasts/Data/DataConstruction](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction). Download the realized volatility measures via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting, for the period `All`, trades of (i) `SELECT SECTOR SPDR TRUST` (symbol=`XLE`, PN=`86454`), and (ii) trades of `SELECT SECTOR SPDR TRUST` (symbol=`XLF`, PN=`86455`). 
The downloaded files contain two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. 
Save the data as the `.csv` files `RVFullPeriodXLETrade.csv` and `RVFullPeriodXLFTrade.csv` in the folder [01DensityForecasts/Data/DataConstruction](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction). The series `XLE` and `XLF` have been obtained from [Yahoo Finance](https://finance.yahoo.com/quote/XLE/) through the `yfinance` module. 
The script `EmpiricalDataMultivariateRiskMan.py` located in the [DataConstruction](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data/DataConstruction) folder, downloads the price series, transforms them into log returns, and merges them with the corresponding deannualized realized measures. The script produces the files `XLEandRealVolXiu.csv` and `XLFandRealVolXiu.csv`, which can then be copied to the [01DensityForecasts/Data](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data) folder. 
For exact replication of results, we recommend using the provided files in [01DensityForecasts/Data](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data). 
Re-downloading the series may lead to small numerical differences (~10e-7) due to real time auto-adjustments for dividends and stock splits and/or other differences related to future changes to the `yfinance` module. 

### Code
The code is organized in the following three folders:
1. [01DensityForecasts](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts): The main script is `RiskManMainMV.py`, for which a sample bash script `S1_RiskManBivariate.sh` is provided to facilitate parallel computation on a computing cluster. The main script relies on `RiskManBasisMV.py`, which implements fundamental routines including the rolling window estimation procedure, and on `DCCmodel.py` together with `TGARCHmodel.py` and `BivariateT.py`, which implement the individual forecast methods. Running `RiskManMainMV.py` generates parameter estimates of the density forecasts based on the observations in the [Data](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data) folder and stores them as `.npy` files in the [mParamsDF](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/mParamsDF) folder, together with the calculated empirical quantiles (mRhat). After completion, the folders [Data](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/Data)  and [mParamsDF](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/01DensityForecasts/mParamsDF) should be manually copied to [02Scores](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores).

2. [02Scores](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores): Separate main scripts for the indicator product and logistic product weight function are included as `RiskManMainMVIndProd.py` and `RiskManMainMVLogProd3.py`, respectively, with example bash scripts `S6_RiskManScoresMVIndProd.sh` and `S6_RiskManScoresMVLogProd3.sh`. The main scripts depend on the functions in `RiskManBasisMV.py`, `BivariateT.py`, `ScoringRulesMV.py` and `WeightfunctionsMV.py`, including fundamental supporting functions, a custom function for the bivariate t distribution, scoring rules and weight functions, respectively. Execution of the main scripts produces the scores of the density forecasts built on the parameters in [mParamsDF](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores/mParamsDF) and the associated observations in the [Data](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores/Data) and saves them as `.npy` files into the folder [mScores](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/02Scores/mScores), which should be manually copied to [03MCS](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS) upon completion.
 
3. [03MCS](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS): We use the same split as in step 2 per weight function. Running the R scripts `RiskManMCSBivariateIndProd.R` and `RiskManMCSBivariateLogProd3.R` produces the MCS p-values based on the scores in [mScores](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS/mScores) and saves them as `.xlsx` files in the folder [MCSTables](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS/MCSTables). 

### Output
* **Table 2** and **Table I1**, last two weight functions of Sec. 4.1 panel,  **Table I4**, and **Table I5**, run  `MCSAnalysisRiskManIndProd.py` and `MCSAnalysisRiskManLogProd3.py`.

Navigate to the folder [03MCS](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS). Run the scripts `MCSAnalysisRiskManIndProd.py` and `MCSAnalysisRiskManLogProd3.py`. The MCS results in the folder [MCSTables](02_MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/03MCS/MCSTables) will be translated into the table with MCS p-values in **Table I4** (for the weight function IndicatorProduct) and **Table I5** (for the weight function LogisticProduct) and the summary values for the last two weight functions of Sec. 4.1 panel, in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.


## INFLATION
Folder: [03_INFLATION_Table_2_Appendix_I](03_INFLATION_Table_2_Appendix_I)

### Data
For the data construction, we adopt the procedure of [Medeiros et al. (2021)](https://doi.org/10.1080/07350015.2019.1637745), made available through the GitHub repository [gabrielrvsc/ForecastingInflation](https://github.com/gabrielrvsc/ForecastingInflation/tree/main).  Download the January 2025 vintage `2025-01.csv` of the FRED-MD monthly data via the [McCracken Database](https://www.stlouisfed.org/research/economists/mccracken/fred-databases). Run the R script `01_get_fred_data.R` to convert the raw file into `data.rda`, which will be stored in the `01DensityForecasts/Data` folder. Subsequently run the R script `02_data_acc.R` to construct the accumulated inflation based on `CPIAUCSL` for each horizon. The resulting datasets are saved in the `01DensityForecasts/Data` folder as both `mYAcc.Rdata` and `mYAcc.npy` files. In addition, the last 180 months of observations are saved separately in the same formats, as the files `YAccOut.Rdata` and `mYAccOut.npy`. 

### Code
The code is organized in the following three folders:
1. [01DensityForecasts](03_INFLATION_Table_2_Appendix_I/01DensityForecasts): The construction of the mean of the density forecasts also strongly relies on [Medeiros et al. (2021)](https://doi.org/10.1080/07350015.2019.1637745), hence each individual forecast method now has its own R script:
    * AR model: `03A_call_model_ar.R`
    * Bagging: `03B_call_model_bagging.R`
    * Complete Subset Regression: `03C_call_model_csr.R`
    * LASSO: `03D_call_model_lasso.R`
    * Random Forest: `03E_call_model_rf.R`
    * Random Walk: `03F_call_model_rw.R`
      
    The individual scripts rely on the supporting functions in the files `Functions/functions.R` and `Functions/rolling_window_tpnorm.R`, which calculate the (parameters of) the density forecasts based on the associated observations in the [Data](03_INFLATION_Table_2_Appendix_I/01DensityForecasts/Data) folder and save them as both `.rda` and `.npy` files in the [mParamsDF](03_INFLATION_Table_2_Appendix_I/01DensityForecasts/mParamsDF) folder.
After completion, the folders [Data](03_INFLATION_Table_2_Appendix_I/01DensityForecasts/Data) and [mParamsDF](03_INFLATION_Table_2_Appendix_I/01DensityForecasts/mParamsDF) should be manually copied to `02Scores`.

2. [02Scores](03_INFLATION_Table_2_Appendix_I/02Scores): We have divided the computation tasks per horizon (*h=6* and *h=24*) and weight function (tails (*T*) and center (*C*)), corresponding to the main scripts:
    * `InflationScoreCalcMain_T_h6.py` (`S1_InflationScores_T_h6.sh`)
    * `InflationScoreCalcMain_T_h24.py` (`S1_InflationScores_T_h24.sh`)
    * `InflationScoreCalcMain_C_h6.py` (`S1_InflationScores_C_h6.sh`)
    * `InflationScoreCalcMain_C_h24.py` (`S1_InflationScores_C_h24.sh`)
 
   with respective example bash scripts in between brackets. The main scripts depend on the functions in `ScoreBasisInflation.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of a main script produces the scores of the density forecasts built on the parameters in [mParamsDF](03_INFLATION_Table_2_Appendix_I/02Scores/mParamsDF) and the associated observations in [Data](03_INFLATION_Table_2_Appendix_I/02Scores/Data) and saves them as `.npy` files into the folder [mScores](03_INFLATION_Table_2_Appendix_I/02Scores/mScores), which should be manually copied to [03MCS](03_INFLATION_Table_2_Appendix_I/03MCS) upon completion.
 
3. [03MCS](03_INFLATION_Table_2_Appendix_I/03MCS): Running the R scripts `MCSTables_InflationCenter.R` and `MCSTables_InflationTails.R` produces the MCS p-values for the tails and center indicator weight function based on the scores in [mScores](03_INFLATION_Table_2_Appendix_I/03MCS/mScores) and saves them as `.xlsx` files in the folder [MCSTables](03_INFLATION_Table_2_Appendix_I/03MCS/MCSTables). 

### Output
* **Table 2** and **Table I1**, Sec. 4.2 panel, **Table I6**, and **Table I7**, run `03MCS/MCSAnalysisInflationCenter.py` and `03MCS/MCSAnalysisInflationTails.py`.
  
Navigate to the folder [03MCS](03_INFLATION_Table_2_Appendix_I/03MCS). Run the scripts `MCSAnalysisInflationCenter.py` and `MCSAnalysisInflationTails.py`. The MCS results in the folder [MCSTables](03_INFLATION_Table_2_Appendix_I/03MCS/MCSTables) will be translated into the table with MCS p-values in **Table I6** (for the center indicator weight function) and **Table I7** (for the tails indicator weight function) and the summary values in the Sec. 4.2 panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

## CLIMATE
Folder: [04_CLIMATE_Table_2_Appendix_I](04_CLIMATE_Table_2_Appendix_I)

### Data
The temperature data are downloaded as a `.txt` file directly from [KNMI - Daily Weather Data De Bilt](https://cdn.knmi.nl/knmi/map/page/klimatologie/gegevens/daggegevens/etmgeg_260.zip), starting on January 1, 1901, and updated regularly. From this file we extract the columns `YYYYMMDD` (=Date), `TG` (=TempAvg, average temperature), `TN` (=TempMin, minimum temperature), `TX` (=TempMax, maximum temperature), and save them as `.xlsx` file, resulting in `01DensityForecasts/ClimateKNMI_Temp.xlsx`. In `01DensityForecasts/ClimateMain.py` we divide the raw values by ten to convert them to degrees Celsius, and select the required sample period (sStart =`2003-02-01`, sEnd = `2023-01-31`).

### Code
The code is organized in the following three folders:
1. [01DensityForecasts](04_CLIMATE_Table_2_Appendix_I/01DensityForecasts): The main script is `ClimateMain.py`, for which a sample bash script `S1_ClimateMain.sh` is provided to facilitate parallel computation on a computing cluster. The main script depends on `ClimateBasis.py`, which implements fundamental routines including the rolling window estimation procedure. For the individual forecast methods, we have split the functions into three files labeled `ClimateLocMeanSinGARCH.py`, `ClimateLocMeanSinGARCHI.py` and `ClimateLocMeanSinGARCHII.py` for the different variance updating equations. Executing `ClimateMain.py` produces the parameter estimates of the density forecasts based on the observations in the `ClimateKNMI_Temp.xlsx` file and stores them as `.npy` files in the [mParamsDF](04_CLIMATE_Table_2_Appendix_I/01DensityForecasts/mParamsDF) folder, together with the calculated empirical quantiles (mRhat) and tail proportions (mGammaHat). It additionally saves the out-of-sample observations in separate `.npy` files. After completion, the folder [mParamsDF](04_CLIMATE_Table_2_Appendix_I/01DensityForecasts/mParamsDF) should be manually copied to [02Scores](04_CLIMATE_Table_2_Appendix_I/02Scores).

2. [02Scores](04_CLIMATE_Table_2_Appendix_I/02Scores): Separate main scripts for the right tail (*R*) and center (*C*) indicator weight function are included as `ClimateScoreCalcMain_R.py` and `ClimateScoreCalcMain_C.py`, respectively, with example bash scripts `S1_ClimateScores_C.sh` and `S1_ClimateScores_R.sh`. The main scripts depend on the functions in `ScoreBasis.py`, `ScoringRules.py` and `Weightfunctions.py`, including fundamental supporting functions, scoring rules and weight functions, respectively. Execution of the main scripts produces the scores of the density forecasts built on the parameters and out-of-sample observations in [mParamsDF](04_CLIMATE_Table_2_Appendix_I/02Scores/mParamsDF) and saves them as `.npy` files into the folder [mScores](04_CLIMATE_Table_2_Appendix_I/02Scores/mScores), which should be manually copied to [03MCS](04_CLIMATE_Table_2_Appendix_I/03MCS) upon completion.
 
3. [03MCS](04_CLIMATE_Table_2_Appendix_I/03MCS): Running the R scripts `MCSTables_ClimateTails.R` and `MCSTables_ClimateCenter.R` produces the MCS p-values based on the scores in [mScores](04_CLIMATE_Table_2_Appendix_I/03MCS/mScores) and saves them as `.xlsx` files in the folder [MCSTables](04_CLIMATE_Table_2_Appendix_I/03MCS/MCSTables). 

### Output
* **Table 2** and **Table I1**, Sec. 4.3 panel,  **Table I8**, and **Table I9**, run `03MCS/MCSAnalysisClimate_Tails.py` and `03MCS/MCSAnalysisClimate_Center.py`.
  
Navigate to the folder [03MCS](04_CLIMATE_Table_2_Appendix_I/03MCS). Run the scripts `MCSAnalysisClimate_Tails.py` and `MCSAnalysisClimate_Center.py`. The MCS results in the folder [MCSTables](04_CLIMATE_Table_2_Appendix_I/03MCS/MCSTables) will be translated into the table with MCS p-values in **Table I8** (for the right tail indicator weight function) and **Table I9** (for the center indicator weight function) and the summary values in the Sec. 4.3 panel in **Table 2** for MCS confidence level 0.90 and **Table I1** for MCS confidence level 0.75.

## MONTE CARLO
Folder: [05_MONTE_CARLO_Appendix_G](05_MONTE_CARLO_Appendix_G)

### Data
The data are simulated under different DGPs:
* Normal(-0.2,1) and Normal(0.2,1) in the size experiment.
* Normal(0,1) and Student-t(5) in the first two power experiments (**Figure G.2** and **G.4**).
* Laplace(-1,1) and Laplace(1,1.1) in the final power experiment (**Figure G.6**).

By running the `Calc` scripts under Code below, data are temporarily saved in the empty folder `mDataAndWeights`.

### Code
The Monte Carlo Study detailed in Appendix G includes a size experiment and three power experiments: Normal vs. Student-t(5) left-tail, Normal vs. Student-t(5) center and Laplace(-1,1) vs. Laplace(1,1.1). The power studies are supplemented with the analysis of the associated standardized local divergences, for which we include separate folders per experiment. In total, this yields seven subdirectories for the reproduction of the figures in *Appendix G*, with titles indicating which figure is reproduced (e.g. *FIG_G1_Size* for the reproduction of **Figure G.1**). 

* [FIG_G1_Size](05_MONTE_CARLO_Appendix_G/FIG_G1_Size): Size experiment. For reproduction, run
    1. `01SizeMain_Calc.py`, which computes the DM test statistics and saves them as `.npy` file in the [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G1_Size/mDMCalc) folder.
    2. `02SizeMain_Plot.py`, which calculates the rejection rates from the DM test statistics in [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G1_Size/mDMCalc), generates **Figure G.1**, and saves it as `.pdf` file in the [Figures](05_MONTE_CARLO_Appendix_G/FIG_G1_Size/Figures) folder.
   
  The files `ScoringRulesMC.py`, `WeightFunctionsMC.py`, `SizePlots.py` and `SizeBasis.py` provide functions for the required scoring rules, weight functions, plotting and other supporting functions, respectively. The script `S1_SizeMain.sh` is a sample bash script for running `01SizeMain_Calc.py` in parallel on a computing cluster. The file `ReadMe_FIG_G1.txt` summarizes details specific to the current folder. 
* [FIG_G2_RejRates_NS5_L_C20](05_MONTE_CARLO_Appendix_G/FIG_G2_RejRates_NS5_L_C20): Rejection rates for power study Normal(0,1) versus Student-t(5) where the region of interest is the left-tail, with *c=20* tail observations. For reproduction, run
    1. `01PowerMain_NS5_L_C20_Calc.py`, which computes the DM test statistics and saves them as `.npy` file in the [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G2_RejRates_NS5_L_C20/mDMCalc) folder.
    2. `02PowerMain_NS5_L_C20_Plot.py`, which calculates the rejection rates from the DM test statistics in [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G2_RejRates_NS5_L_C20/mDMCalc), generates the subfigures of **Figure G.2**, and saves them as `.pdf` files in the [Figures](05_MONTE_CARLO_Appendix_G/FIG_G2_RejRates_NS5_L_C20/Figures) folder.
   
  The files `ScoringRulesMC.py`, `WeightFunctionsMC.py`, `PowerPlots.py` and `PowerBasis.py` provide functions for the required scoring rules, weight functions, plotting and other supporting functions, respectively. The script `S2_PowerMain_NS5_L_C20.sh` is a sample bash script for running `01PowerMain_NS5_L_C20_Calc.py` in parallel on a computing cluster. The file `ReadMe_FIG_G2.txt` summarizes details specific to the current folder.  
* [FIG_G3_LocalDiv_NS5_L_C20](05_MONTE_CARLO_Appendix_G/FIG_G3_LocalDiv_NS5_L_C20): Standardized local divergences for Normal(0,1) versus Student-t(5) where the region of interest is the left-tail. For reproduction, run
    1. `01_A_DivergencesMain_Calc.py` and `01_B_DivergencesMain_Calc.py`, which compute the standardized divergences (where the B version switches the order of the distributions compared to A) and save the resulting `.xlsx` files in the [OutputDataFrames](05_MONTE_CARLO_Appendix_G/FIG_G3_LocalDiv_NS5_L_C20/OutputDataFrames) folder.
    2. `02DivergencesMain_Plot.py`, which generates the subfigures of **Figure G.3** from the values in the `.xlsx` files in the [OutputDataFrames](05_MONTE_CARLO_Appendix_G/FIG_G3_LocalDiv_NS5_L_C20/OutputDataFrames) folder, and saves them as `.pdf` files in the folders [Figures/StandDiv](05_MONTE_CARLO_Appendix_G/FIG_G3_LocalDiv_NS5_L_C20/Figures/StandDiv) and [Figures/XiS](05_MONTE_CARLO_Appendix_G/FIG_G3_LocalDiv_NS5_L_C20/Figures/XiS) for the standardized divergences in subfigures **(a)** to **(d)** and ratios in subfigure **(e)**, respectively.
       
  The files `ScoringRulesLocalDiv.py`, `WeightFunctionsMC.py`, `DivergencesPlots.py` and `DivergencesBasis.py` provide functions for the required scoring rules, plotting and other supporting functions, respectively. The scripts `S1_Divergences_A.sh` and `S1_Divergences_B.sh` are sample bash scripts for running `01_A_DivergencesMain_Calc.py` and `01_B_DivergencesMain_Calc.py` in parallel on a computing cluster. The file `ReadMe_FIG_G3.txt` summarizes details specific to the current folder. 
* [FIG_G4_RejRates_NS5_C_C200](05_MONTE_CARLO_Appendix_G/FIG_G4_RejRates_NS5_C_C200): Rejection rates for power study Normal(0,1) versus Student-t(5) where the region of interest is the center, with *c=200* observations in the region of interest. For reproduction, run
    1. `01_A_PowerMain_NS5_C_C200_Sim.py`, which simulates and saves the data in the folder [mDataAndWeights](05_MONTE_CARLO_Appendix_G/FIG_G4_RejRates_NS5_C_C200/mDataAndWeights).
    2. `01_B_PowerMain_NS5_C_C200_Calc.py`, which computes the DM test statistics and saves them as `.npy` file in the [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G4_RejRates_NS5_C_C200/mDMCalc) folder.
    3. `02PowerMain_NS5_C_C200_Plot.py`, which calculates the rejection rates from the DM test statistics in [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G4_RejRates_NS5_C_C200/mDMCalc), generates the subfigures of **Figure G.4**, and saves them as `.pdf` files in the [Figures](05_MONTE_CARLO_Appendix_G/FIG_G4_RejRates_NS5_C_C200/Figures) folder.
       
  The files `ScoringRulesMC.py`, `WeightFunctionsMC.py`, `PowerPlots.py` and `PowerBasis.py` provide functions for the required scoring rules, weight functions, plotting and other supporting functions, respectively. The scripts `S1_PowerMain_NS5_C_C200_A.sh` and `S1_PowerMain_NS5_C_C200_B.sh` is a sample bash script for running `01_A_PowerMain_NS5_C_C200_Sim.py` and  `01_B_PowerMain_NS5_C_C200_Calc.py` in parallel on a computing cluster. The file `ReadMe_FIG_G4.txt` summarizes details specific to the current folder.  
* [FIG_G5_LocalDiv_NS5_C_C200](05_MONTE_CARLO_Appendix_G/FIG_G5_LocalDiv_NS5_C_C200): Standardized local divergences for Normal(0,1) versus Student-t(5) where the region of interest is the center. For reproduction, run
    1. `01_A_DivergencesMain_Calc.py` and `01_B_DivergencesMain_Calc.py`, which compute the standardized divergences (where the B version switches the order of the distributions compared to A) and save the resulting `.xlsx` files in the [OutputDataFrames](05_MONTE_CARLO_Appendix_G/FIG_G5_LocalDiv_NS5_C_C200/OutputDataFrames)  folder.
    2. `02DivergencesMain_Plot.py`, which generates the subfigures of **Figure G.5** from the values in the `.xlsx` files in the [OutputDataFrames](05_MONTE_CARLO_Appendix_G/FIG_G5_LocalDiv_NS5_C_C200/OutputDataFrames) folder, and saves them as `.pdf` files in the folders [Figures/StandDiv](05_MONTE_CARLO_Appendix_G/FIG_G5_LocalDiv_NS5_C_C200/Figures/StandDiv) and [Figures/XiS](05_MONTE_CARLO_Appendix_G/FIG_G5_LocalDiv_NS5_C_C200/Figures/XiS) for the standardized divergences in subfigures **(a)** to **(d)** and ratios in subfigure **(e)**, respectively.
          
  The files `ScoringRulesLocalDiv.py`, `WeightFunctionsMC.py`, `DivergencesPlots.py` and `DivergencesBasis.py` provide functions for the required scoring rules, plotting and other supporting functions, respectively. The scripts `S1_Divergences_A.sh` and `S1_Divergences_B.sh` are sample bash scripts for running `01_A_DivergencesMain_Calc.py` and `01_B_DivergencesMain_Calc.py` in parallel on a computing cluster. The file `ReadMe_FIG_G5.txt` summarizes details specific to the current folder. 
* [FIG_G6_RejRates_LP_L_C20](05_MONTE_CARLO_Appendix_G/FIG_G6_RejRates_LP_L_C20): Rejection rates for power study Laplace(-1,1) versus Laplace(1,1.1) where the region of interest is the left-tail, with *c=20* tail observations. For reproduction, run
    1. `01PowerMain_LP_L_C20_Calc.py`, which computes the DM test statistics and saves them as `.npy` file in the [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G6_RejRates_LP_L_C20/mDMCalc) folder.
    2. `02PowerMain_LP_L_C20_Plot.py`, which calculates the rejection rates from the DM test statistics in [mDMCalc](05_MONTE_CARLO_Appendix_G/FIG_G6_RejRates_LP_L_C20/mDMCalc), generates the subfigures of **Figure G.6**, and saves them as `.pdf` files in the [Figures](05_MONTE_CARLO_Appendix_G/FIG_G6_RejRates_LP_L_C20/Figures) folder.
     
  The files `ScoringRulesMC.py`, `WeightFunctionsMC.py`, `PowerPlots.py` and `PowerBasis.py` provide functions for the required scoring rules, weight functions, plotting and other supporting functions, respectively. The script `S2_PowerMain_LP_L_C20.sh` is a sample bash script for running `01PowerMain_LP_L_C20_Calc.py` in parallel on a computing cluster. The file `ReadMe_FIG_G6.txt` summarizes details specific to the current folder. 
* [FIG_G7_LocalDiv_LP_L_C20](05_MONTE_CARLO_Appendix_G/FIG_G7_LocalDiv_LP_L_C20): Standardized local divergences for Laplace(-1,1) versus Laplace(1,1.1) where the region of interest is the left-tail. For reproduction, run
    1. `01_A_DivergencesMain_Calc.py` and `01_B_DivergencesMain_Calc.py`, which compute the standardized divergences (where the B version switches the order of the distributions compared to A) and save the resulting `.xlsx` files in the [OutputDataFrames](05_MONTE_CARLO_Appendix_G/FIG_G7_LocalDiv_LP_L_C20/OutputDataFrames) folder.
    2. `02DivergencesMain_Plot.py`, which generates the subfigures of **Figure G.7** from the values in the `.xlsx` files in the [OutputDataFrames](05_MONTE_CARLO_Appendix_G/FIG_G7_LocalDiv_LP_L_C20/OutputDataFrames) folder, and saves them as `.pdf` files in the folders [Figures/StandDiv](05_MONTE_CARLO_Appendix_G/FIG_G7_LocalDiv_LP_L_C20/Figures/StandDiv) and [Figures/XiS](05_MONTE_CARLO_Appendix_G/FIG_G7_LocalDiv_LP_L_C20/Figures/XiS) for the standardized divergences in subfigures **(a)** to **(d)** and ratios in subfigure **(e)**, respectively.

  The files `ScoringRulesLocalDiv.py`, `WeightFunctionsMC.py`, `DivergencesPlots.py` and `DivergencesBasis.py` provide functions for the required scoring rules, plotting and other supporting functions, respectively. The scripts `S1_Divergences_A.sh` and `S1_Divergences_B.sh` are sample bash scripts for running `01_A_DivergencesMain_Calc.py` and `01_B_DivergencesMain_Calc.py` in parallel on a computing cluster. The file `ReadMe_FIG_G7.txt` summarizes details specific to the current folder. 

### Output
* **Figure G.1**, run `FIG_G1_Size/02SizeMain_Plot.py`.
* **Figure G.2**, run `FIG_G2_RejRates_NS5_L_C20/02PowerMain_NS5_L_C20_Plot.py`.
* **Figure G.3**, run `FIG_G3_LocalDiv_NS5_L_C20/02DivergencesMain_Plot.py`.
* **Figure G.4**, run `FIG_G4_RejRates_NS5_C_C200/02PowerMain_NS5_C_C200_Plot.py`.
* **Figure G.5**, run `FIG_G5_LocalDiv_NS5_C_C200/02DivergencesMain_Plot.py`.
* **Figure G.6**, run `FIG_G6_RejRates_LP_L_C20/02PowerMain_LP_L_C20_Plot.py`.
* **Figure G.7**, run `FIG_G7_LocalDiv_LP_L_C20/02DivergencesMain_Plot.py`.

## MITCHELL AND WEALE (EXAMPLE 6)
Folder: [06_MITCHELL_AND_WEALE_Example_6](06_MITCHELL_AND_WEALE_Example_6)

In **Example 6** of Section 3.4, we provide a specific example for which the expected score difference based on the weighted scoring rule by [Mitchell and Weale (2023)](https://doi.org/10.1002/jae.2972) is negative for *α>α_0*. The aim of the current folder is to reproduce the (rounded) number *α_0 = 0.052* and to graphically verify the inequality *α>α_0*. Run `Example6.py`. The script generates a figure of the expected score differences, saved in the [Figures](06_MITCHELL_AND_WEALE_Example_6/Figures) folder, and prints the alpha root of the expected score differences. The file ReadMe_MITCHELL_AND_WEALE.txt summarizes details specific to the current folder.


## Other files
- `requirements.txt`: required Python libraries and packages (including dependencies installed on the computing cluster)
- `requirementsLocal.txt`: required Python libraries and packages (excluding dependencies installed on the computing cluster)
- `InstallPackages.R`: required R packages
- `ComputationTimePerFile.xlsx`: Indicating computation time per file
- `LICENSE`: MIT License.
- `README.md`: This *README.md* file




