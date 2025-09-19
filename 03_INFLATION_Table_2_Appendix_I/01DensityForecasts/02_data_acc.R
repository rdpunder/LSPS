### must add package for specific models ###
packages <- c("roll","HDeconometrics","glmnet","randomForest","tidyverse","TTR","rugarch","reticulate","writexl","rje","devtools")

# Install any that are not already installed
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Apply the function to each package
sapply(packages, install_if_missing)
library(devtools)
install_github("gabrielrvsc/HDeconometrics") # provided by Medeiros et al. (2021)
library(HDeconometrics)
library(glmnet)
library(randomForest)
library(tidyverse)
library(TTR)
library(rugarch) # estimation and forecasting univariate GARCH models
library(reticulate)
library(writexl)
library(rje) # logit and expit
source("Functions/rolling_window_tpnorm.R")
source("Functions/functions.R") 

#####
## The file with the forecasts will be saved with model_name
model_name = "rwtpnorm"
## The function called to run models is model_function, which is a function from functions.R
model_function = runrwtpnorm
#####

# Other parameters
sDep = "CPIAUCSL"
nwindows = 180
lHorizon = c(6,24)

#Load data
load("Data/data.rda")

#Transform dependent variable to accumulative yearly
vYMonthly=as.matrix(data[sDep])


mYAcc = matrix(NA, nrow=length(vYMonthly), ncol=length(lHorizon))

for(j in 1:length(lHorizon)){
  h = lHorizon[j]
  for(i in h:length(vYMonthly)){
    mYAcc[i,j] =  12/h * sum(vYMonthly[(i-h+1):i])   
  }
}
mYAccOut = tail(mYAcc, nwindows)

# Save
save(mYAcc,file = paste("Data/mYAcc.rda",sep = ""))
reticulate::r_to_py(mYAcc)$dump(paste("Data/mYAcc.npy",sep = ""))
save(mYAccOut,file = paste("Data/mYAccOut.rda",sep = ""))
reticulate::r_to_py(mYAccOut)$dump(paste("Data/mYAccOut.npy",sep = ""))