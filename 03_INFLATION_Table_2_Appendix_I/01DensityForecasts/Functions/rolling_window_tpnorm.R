rolling_window=function(fn,df,nwindow=1,horizon,variable,...){
  ind=1:nrow(df)
  window_size=nrow(df)-nwindow
  indmat=matrix(NA,window_size,nwindow)
  indmat[1,]=1:ncol(indmat)
  for(i in 2:nrow(indmat)){
    indmat[i,]=indmat[i-1,]+1
  }
  rw=apply(indmat,2,fn,df=df,horizon=horizon,variable=variable,...) # applied over columns [2]
  
  forecast=unlist(lapply(rw,function(x)x$forecast))
  forecastSig_1=unlist(lapply(rw,function(x)x$forecastSig_1))
  forecastSig_2=unlist(lapply(rw,function(x)x$forecastSig_2))

  outputs=lapply(rw,function(x)x$outputs)
  
  return(list(forecast=forecast, forecastSig_1 = forecastSig_1, forecastSig_2 =forecastSig_2))
}