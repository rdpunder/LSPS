library(quantmod)

#################################
sTicker = "^GSPC" 
sStart = "2008-01-01" #yyyy-dd-mm
sEnd = "2022-12-31"   #yyyy-dd-mm
sFileName = 'YahooSP500.csv'
  
#################################
dfSPX <- getSymbols(sTicker,auto.assign = FALSE, from = sStart, to = sEnd)
write.zoo(dfSPX, file=sFileName, sep=",")