#### gets out of sample y and computes random walk forecasts ###
packages <- c("roll","dplyr")

# Install any that are not already installed
install_if_missing <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    install.packages(pkg)
  }
}

# Apply the function to each package
sapply(packages, install_if_missing)

library(roll)
library(dplyr)
load("Data/data.rda")
dates = data$date
data = data%>%select(-date)%>%as.matrix()
rownames(data) = as.character(dates)

nwindows = 180

y = data[,"CPIAUCSL"]
y = cbind(y,(roll_prod(1+y/100,3)-1)*100,(roll_prod(1+y/100,6)-1)*100,(roll_prod(1+y/100,12)-1)*100)

yout = tail(y,nwindows)

# rw forecasts
iHorizon=1
rw=data[(nrow(data)-nwindows-iHorizon+1):(nrow(data)-iHorizon),"CPIAUCSL"]
vRes = yout-rw

vYoutAcc12 = yout[,4]

save(yout,file = "mParamsDF/yout.rda")
save(vYoutAcc12,file = "mParamsDF/youtacc12.rda")
save(rw,file = "mParamsDF/rw.rda")
