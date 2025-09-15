# Localizing Strictly Proper Scoring Rules

This repository provides the data and code to reproduce all the empirical and Monte Carlo simulation results contained in the paper **Localizing Strictly Proper Scoring Rules**.


## Structure of the repository

## RISK MANAGEMENT
Directory: `RISK_MANAGEMENT_Table_2_Appendix_I/`

### Data
Download the Realized Volatility measure via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting trades of `SPDR S & P 500 E T F TRUST` (symbol=`SPY`, PN=`843398`) for the period `All`. The downloaded file contains two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. Save the data as `.csv` file `RealisedVolatilityFullPeriodTrade.csv` in the subdirectory `01DensityForecasts/Data`. The S&P500 series (ticker: `SPY`) is downloaded from [Yahoo Finance](https://finance.yahoo.com/quote/SPY/) through the `yfinance` module by running the Python script `01DensityForecasts/Data/EmpiricalDataRiskMan.py`, which also transforms the prices into log returns and merges the S&P500 data with the deannualized realized measure.  


### Code

### Output

## MULTIVARIATE RISK MANAGEMENT
Directory: `MULTIVARIATE_RISK_MANAGEMENT_Table_2_Appendix_I/`

### Data
Download the Realized Volatility measures via Dacheng Xiu's [Risk Lab](https://dachxiu.chicagobooth.edu/#risklab) by selecting, for the period `All`, trades of (i) `SELECT SECTOR SPDR TRUST` (symbol=`XLE`, PN=`86454`), and (ii) trades of `SELECT SECTOR SPDR TRUST` (symbol=`XLF`, PN=`86455`) for the period `All` The downloaded files contains two types: QMLE-Trades and QMLE-Quote, from which we select `QMLE-Trades`. Save the data as the `.csv` files `RVFullPeriodXLETrade.csv` and `RVFullPeriodXLFTrade.csv` in the folder `01DensityForecasts/Data`. The series Energy Select Sector SPDR Fund (ticker: `XLE`) and Financial Select Sector SPDR (ticker: `XLF`) are downloaded from [Yahoo Finance](https://finance.yahoo.com/quote/XLE/) through the `yfinance` module by running the Python script `01DensityForecasts/Data/EmpiricalDataMultivariateRiskMan.py`, which also transforms the prices into log returns and merges the XLE and XLF data with the corresponding deannualized realized measure.  


### Code

### Output

## INFLATION
Directory: `INFLATION_Table_2_Appendix_I/`

### Data
The inflation data is sourced from the code provided by Medeiros et al. (2021) and stored as `01DensityForecasts/Data/Data.Rdata`. Run the R script `01_data_acc.R` to construct the accumulated inflation for each horizon. The resulting datasets are saved in the `01DensityForecasts/Data` directory as both `mYAcc.Rdata` and `mYAcc.npy` file. In addition, the last 180 months of observations are saved separately in the same formats, as the files `YAcc.Rdata` and `mYAcc.npy`. 

### Code

### Output

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




