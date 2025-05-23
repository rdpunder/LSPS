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
source("functions/rolling_window_tpnorm.R")
source("functions/functions.R") 

#####
## The file with the forecasts will be saved with model_name
model_name = "csrtpnorm"
## The function called to run models is model_function, which is a function from functions.R
model_function = runcsrtpnorm
#####

# Other parameters
sDep = "CPIAUCSL"
nwindows = 180
lHorizon = c(6,24)

#Load data
load("Data/data.rda")

#Transform dependent variable to accumulative yearly
vYMonthly=as.matrix(data[sDep])
load("Data/mYAcc.rda")
load("Data/mYAccOut.rda")

print((head(mYAcc, 28)))
print((head(mYAccOut, 28)))

for(i in 1:length(lHorizon)){
  iHorizon = lHorizon[i]
  print(iHorizon)

  #Load data
  load("Data/data.rda") # load dataframe with all data
  dates = data$date
  data = data%>%select(-date)%>%as.matrix()
  rownames(data) = as.character(dates)
  vYAcc = mYAcc[,i]
  data = cbind(data,vYAcc)
  sDepAcc = paste(sDep, as.character(iHorizon), sep="" )
  colnames(data)[colnames(data) == "vYAcc"] = sDepAcc
  
  sum(mYAccOut[,i] != tail(mYAcc[,i], nwindows))   
  
  ####### run rolling window ##########
  model = rolling_window(model_function,data,nwindows+iHorizon-1,iHorizon, sDep)
  
  # remark about timing: forecast vYPred[k] is the forecast for t=k
  mForecast = matrix(data=NA, nrow=nwindows, ncol=4)
  mForecast[,1] = model$forecast[1:nwindows]
  mForecast[,2] = model$forecastSig_1[1:nwindows]
  mForecast[,3] = model$forecastSig_2[1:nwindows]
  
  # Save
  sFileName = paste("mParamsDF/",model_name,"h",as.character(iHorizon),sep = "")
  save(mForecast,file = paste(sFileName,".rda",sep = ""))
  reticulate::r_to_py(mForecast)$dump(paste(sFileName,".npy",sep = ""))
  
}