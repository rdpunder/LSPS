library(MCS) # package by Bernardi and Catania
library(reticulate)
library("writexl") # save dataframe to excel
library("readxl") # load datadrame from excel
np <- import("numpy")

## Booleans to select application
bRiskMan = FALSE
bInflation = TRUE
bClimate = FALSE
bRiskManRobustCheck = FALSE

## Boolean to select tail(s) or center
bTails = FALSE

## Number of bootstrap replications
iB = 10000 

MCSpvalCustomRiskMan <- function(mLossh1, mLossh5, iB,  sStatistic, iK){
 #' MCS p-value table RiskMan Left-tail application
 #' 
 #' Given a matrix of losses construct matrix of MCS p-values
 #' Recall: MCS p-value of method j is smallest significance level for which
 #' method j is eliminated.
 #' The scores are: 'LogSsharp', 'LogSflat', 'QSsharp', 'QSflat', 'SphSsharp', 'SphSflat', 'CRPSsharp', 'CRPSflat'
 #' @param mLoss matrix of size iM x iS x iT with losses
 #' @param iB integer, number of bootstrap replications
 #' @param iCores integer, number of cores for parallel computations
 #' @param sStatistic statistic MCS procedure 'Tmax' or 'TR'
 #' @param iK block length bootstrap
 #'  description
  
  # Initialise MCS table 
  lScoringRules = c('LogSflat', 'LogSsharp',  'QSflat', 'QSsharp', 'SphSflat', 'SphSsharp', 'CRPSflat', 'CRPSsharp','LogSflath5', 'LogSsharph5',  'QSflath5', 'QSsharph5', 'SphSflath5', 'SphSsharph5', 'CRPSflath5', 'CRPSsharph5')
  mMCSTable = matrix(data=NA, nrow=6, ncol=2*8)
  dfMCStable = data.frame(data=mMCSTable, row.names = c('RGARCH-t','TGARCH-t','GARCH-t','RGARCH-N','TGARCH-N','GARCH-N')) #c('GARCH-N','GARCH-t','TGARCH-N','TGARCH-t','RGARCH-N','RGARCH-t'))
  colnames(dfMCStable) = lScoringRules

  iStatisticPval = 3 + 3 * (sStatistic=='TR')
  vIdx = c(2, 1, 4, 3, 6, 5, 7)
  vH = c(1,5)
  for(h in 1:length(vH)){
    # Select score matrix
    if(h==1){
      mLoss = mLossh1
    } else{
      mLoss = mLossh5#[3,,,]
    }
    for(i in 1:length(vIdx)){
      print(i)
      s = vIdx[i]
      mLossS = mLoss[,s,]
      dfLossS = data.frame(t(mLoss[,s,]))
      colnames(dfLossS) = c('GARCH-N','GARCH-t','TGARCH-N','TGARCH-t','RGARCH-N','RGARCH-t')
      dAlpha=-0.1 # initialisation
      while(dAlpha < 1 & ncol(dfLossS) > 1){
        lColNamesBefore = colnames(dfLossS)
        set.seed(1234) # still varying [very modestly] over different runs, to deep in package code 
        oMCS = MCSprocedure(Loss=dfLossS, alpha=dAlpha, B=iB, cl=NULL, statistic=sStatistic, k=iK)
        dPVal = min(oMCS@show[,iStatisticPval]) #oMCS@Info$mcs_pvalue
        lColNamesAfter = oMCS@Info$model.names
        
        if(oMCS@Info$n_elim>0){
          sElim = setdiff(lColNamesBefore, lColNamesAfter) # we have verified our code for more than one eliminated model
          dPvalMCS = dAlpha
        }else{
          # Eliminate model at pval, its MCS pval is max(current significance, pval), as we need to reach this MCS round
          sElim = oMCS@Info$model.names[oMCS@show[,iStatisticPval] == dPVal]
          dPvalMCS =  max(dPVal, dAlpha)
        }
        
        # Add MCS pval to table
        dfMCStable[sElim,lScoringRules[i+8*(h-1)]]  = dPvalMCS 
        
        # Eliminate model(s) [at pval or the ones eliminated in MCS round, which will then have MCS pval equal to current significance level]
        dfLossS[,sElim] = NULL # = dfMCStableTmax[,!c(sElim)]
        
        # Update alpha
        # next round alpha is mcs pvalue, unless we couldn't eliminate, then increase by 0.01 to keep going
        # we don't need keep going part, since we will move on to next pval if no elimination takes place
        dAlpha = dPvalMCS # max(dPVal, dAlpha) #max(max(dPVal, dAlpha), dAlpha+0.01) 
        
        print(oMCS)
        print(dfMCStable)
        print(sElim)
        print(dPVal)
        print(dAlpha)
        print(head(dfLossS))
        
      }
    }
  }
  
   return(dfMCStable)
}

MCSpvalCustomInflation <- function(mLossh1, mLossh5, iB, sStatistic, iK){
  #' MCS p-value table RiskMan Left-tail application
  #' 
  #' Given a matrix of losses construct matrix of MCS p-values
  #' Recall: MCS p-value of method j is smallest significance level for which
  #' method j is eliminated.
  #' The scores are: 'LogSsharp', 'LogSflat', 'QSsharp', 'QSflat', 'SphSsharp', 'SphSflat', 'CRPSsharp', 'CRPSflat'
  #' @param mLoss matrix of size iM x iS x iT with losses
  #' @param iB integer, number of bootstrap replications
  #' @param iCores integer, number of cores for parallel computations
  #' @param sStatistic statistic MCS procedure 'Tmax' or 'TR'
  #' @param iK block length bootstrap
  #'  description
  
  # Initialise MCS table 
  vIdx = c(2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 15, 14, 16, 17, 13) 
  lScoringRules=c('LogSflat', 'LogSsharp','LogSsharpslog','LogSsharpsbar','QSflat', 'QSsharp', 'QSsharpslog', 'QSsharpsbar', 'SphSflat', 'SphSsharp','SphSsharpslog','SphSsharpsbar', 'CRPSflat', 'CRPSsharp','CRPSslog','CRPSsbar','twCRPS')
  mMCSTable = matrix(data=NA, nrow=6, ncol=1*17)
  dfMCStable = data.frame(data=mMCSTable, row.names = c('rw','ar','bagging','csr','lasso','rf'))
  colnames(dfMCStable) = lScoringRules
  
  iStatisticPval = 3 + 3 * (sStatistic=='TR')
  vIdx = c(2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 15, 14, 16, 17, 13)
  vH = c(1)
  for(h in 1:length(vH)){
    # Select score matrix
    if(h==1){
      mLoss = mLossh1
    } else{
      mLoss = mLossh5#[3,,,]
    }
    for(i in 1:length(vIdx)){
      print(i)
      s = vIdx[i]
      mLossS = mLoss[,s,]
      dfLossS = data.frame(t(mLoss[,s,]))
      colnames(dfLossS) = c('rw','ar','bagging','csr','lasso','rf')
      dAlpha=-0.1 # initialisation
      while(dAlpha < 1 & ncol(dfLossS) > 1){
        lColNamesBefore = colnames(dfLossS)
        set.seed(1234) # still varying [very modestly] over different runs, to deep in package code 
        oMCS = MCSprocedure(Loss=dfLossS, alpha=dAlpha, B=iB, cl=NULL, statistic=sStatistic, k=iK)
        dPVal = min(oMCS@show[,iStatisticPval]) #oMCS@Info$mcs_pvalue
        lColNamesAfter = oMCS@Info$model.names
        
        if(oMCS@Info$n_elim>0){
          sElim = setdiff(lColNamesBefore, lColNamesAfter) # we have verified our code for more than one eliminated model
          dPvalMCS = dAlpha
        }else{
          # Eliminate model at pval, its MCS pval is max(current significance, pval), as we need to reach this MCS round
          sElim = oMCS@Info$model.names[oMCS@show[,iStatisticPval] == dPVal]
          dPvalMCS =  max(dPVal, dAlpha)
        }
        
        # Add MCS pval to table
        dfMCStable[sElim,lScoringRules[i+9*(h-1)]]  = dPvalMCS 
        
        # Eliminate model(s) [at pval or the ones eliminated in MCS round, which will then have MCS pval equal to current significance level]
        dfLossS[,sElim] = NULL # = dfMCStableTmax[,!c(sElim)]
        
        # Update alpha
        # next round alpha is mcs pvalue, unless we couldn't eliminate, then increase by 0.01 to keep going
        # we don't need keep going part, since we will move on to next pval if no elimination takes place
        dAlpha = dPvalMCS # max(dPVal, dAlpha) #max(max(dPVal, dAlpha), dAlpha+0.01) 
        
        print(oMCS)
        print(dfMCStable)
        print(sElim)
        print(dPVal)
        print(dAlpha)
        print(head(dfLossS))
        
      }
    }
  }
  
  dfMCStableOrder = data.frame(data=mMCSTable, row.names = c('rw','ar','bagging','csr','lasso','rf'))
  colnames(dfMCStableOrder) = lScoringRules
  for(s in c('rw','ar','bagging','csr','lasso','rf')){
    dfMCStableOrder[s,]=dfMCStable[s,]
  }
  return(dfMCStable)
}

MCSpvalCustomClimate <- function(mLossh1, mLossh5, iB, sStatistic, iK, lMethodNames, vIdx, lScoringRules=c(FALSE)){
  #' MCS p-value table RiskMan Left-tail application
  #' 
  #' Given a matrix of losses construct matrix of MCS p-values
  #' Recall: MCS p-value of method j is smallest significance level for which
  #' method j is eliminated.
  #' The scores are: 'LogSsharp', 'LogSflat', 'QSsharp', 'QSflat', 'SphSsharp', 'SphSflat', 'CRPSsharp', 'CRPSflat'
  #' @param mLoss matrix of size iM x iS x iT with losses
  #' @param iB integer, number of bootstrap replications
  #' @param iCores integer, number of cores for parallel computations
  #' @param sStatistic statistic MCS procedure 'Tmax' or 'TR'
  #' @param iK block length bootstrap
  #'  description
  
  # Initialise MCS table 
  if(lScoringRules[1]==FALSE){
    lScoringRules = c('LogSflat', 'LogSsharp',  'QSflat', 'QSsharp', 'SphSflat', 'SphSsharp', 'CRPSflat', 'CRPSsharp','twCRPS')
  }
  
  mMCSTable = matrix(data=NA, nrow=6, ncol=1*length(vIdx))
  dfMCStable = data.frame(data=mMCSTable, row.names = lMethodNames)
  colnames(dfMCStable) = lScoringRules
  
  iStatisticPval = 3 + 3 * (sStatistic=='TR')
  vH = c(1)
  for(h in 1:length(vH)){
    # Select score matrix
    if(h==1){
      mLoss = mLossh1
    } else{
      mLoss = mLossh5#[3,,,]
    }
    for(i in 1:length(vIdx)){
      print(i)
      s = vIdx[i]
      mLossS = mLoss[,s,]
      dfLossS = data.frame(t(mLoss[,s,]))
      colnames(dfLossS) = lMethodNames
      dAlpha=-0.1 # initialisation
      while(dAlpha < 1 & ncol(dfLossS) > 1){
        lColNamesBefore = colnames(dfLossS)
        set.seed(1234) 
        oMCS = MCSprocedure(Loss=dfLossS, alpha=dAlpha, B=iB, cl=NULL, statistic=sStatistic, k=iK)
        dPVal = min(oMCS@show[,iStatisticPval]) 
        lColNamesAfter = oMCS@Info$model.names
        
        if(oMCS@Info$n_elim>0){
          sElim = setdiff(lColNamesBefore, lColNamesAfter) 
          dPvalMCS = dAlpha
        }else{
          # Eliminate model at pval, its MCS pval is max(current significance, pval), as we need to reach this MCS round
          sElim = oMCS@Info$model.names[oMCS@show[,iStatisticPval] == dPVal]
          dPvalMCS =  max(dPVal, dAlpha)
        }
        
        # Add MCS pval to table
        dfMCStable[sElim,lScoringRules[i+9*(h-1)]]  = dPvalMCS 
        
        # Eliminate model(s) [at pval or the ones eliminated in MCS round, which will then have MCS pval equal to current significance level]
        dfLossS[,sElim] = NULL # = dfMCStableTmax[,!c(sElim)]
        
        # Update alpha
        # next round alpha is mcs pvalue, unless we couldn't eliminate, then increase by 0.01 to keep going
        # we don't need keep going part, since we will move on to next pval if no elimination takes place
        dAlpha = dPvalMCS # max(dPVal, dAlpha) #max(max(dPVal, dAlpha), dAlpha+0.01) 
        
        print(oMCS)
        print(dfMCStable)
        print(sElim)
        print(dPVal)
        print(dAlpha)
        print(head(dfLossS))
        
      }
    }
  }
  dfMCStableOrder = data.frame(data=mMCSTable, row.names = lMethodNames)
  colnames(dfMCStableOrder) = lScoringRules
  for(s in lMethodNames){
    dfMCStableOrder[s,]=dfMCStable[s,]
  }
  return(dfMCStable)
}

################################################################################
################################ RISKMANAGMENT #################################
################################################################################
if(bRiskMan){
  
  # Parameters
  sStatistic = 'TR' # statistic, 'Tmax' or 'TR'
  iK = 20
  iTest = 1000
  
  ## Data
  mLossh1 = -np$load(paste('mScores/mScores_h1_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  mLossh5 = -np$load(paste('mScores/mScores_h5_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  
  ## MCSpval for different values of q
  vQ = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25) # quantiles for which scores have been calculated
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
  
}

################################################################################
################################## INFLATION ###################################
################################################################################
if(bInflation){
  
  ## Parameters
  sStatistic = 'TR' # statistic, 'Tmax' or 'TR'
  iTest = 48
  iK = 5
  
  ## Scores
  if(bTails){
    mLossh6 = -np$load('mScores/mScores_h_6_ML-tpnorm_iT_180_iS_17_fWIndicatorTails_vFinal.npy')
    mLossh24 = -np$load('mScores/mScores_h_24_ML-tpnorm_iT_180_iS_17_fWIndicatorTails_vFinal.npy')
  }else{
    mLossh6 = -np$load('mScores/mScores_h_6_ML-tpnorm_iT_180_iS_17_fWIndicatorC_vFinal.npy')
    mLossh24 = -np$load('mScores/mScores_h_24_ML-tpnorm_iT_180_iS_17_fWIndicatorC_vFinal.npy')
  }
  lLoss = list(mLossh6, mLossh24)
  
  ## MCSpval for different values of q for inflation
  vQ = c(1,1.5,2) # quantiles for which scores have been calculated
  for(h in 1:length(lLoss)){
    mLoss = lLoss[[h]]
    for(q in 1:length(vQ)){
      dfMCSpval = MCSpvalCustomInflation(mLoss[q,,,], NA, iB,  sStatistic, iK)
      sFileName = paste('MCSTables/Inflation_MCSTable_','h',as.character(c(6,24)[h]),'_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'_tails',as.character(as.integer(bTails)),'.xlsx',sep = "")
      write_xlsx(dfMCSpval, sFileName)
    }
  }
}

################################################################################
################################### CLIMATE ####################################
################################################################################
if(bClimate){
  
  ## Parameters
  sStatistic = 'TR' # statistic, 'Tmax' or 'TR'
  iTest = 2922
  iK = 200
  
  ## Load data and settings
  sMCSSetting = paste(sStatistic, as.character(iK),sep = "")
  if(bTails){
    # Note: Right tail in climate application
    mLossh1 = -np$load('mScores/mScores_h1_TempGARCH-norm-t_iT_7305_iTest_2922_iS_16_fWIndicatorR_vFinal_bWarm_False.npy')
    mLossh3 = -np$load('mScores/mScores_h3_TempGARCH-norm-t_iT_7305_iTest_2922_iS_16_fWIndicatorR_vFinal_bWarm_False.npy')
    vQ=c(0.75, 0.80, 0.85, 0.90, 0.95, 0.975, 0.99) # quantiles for which scores have been calculated
    vIdx = c(2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 13, 14, 15, 16) 
    lScoringRules=c('LogSflat', 'LogSsharp','LogSsharpslog','LogSsharpsbar','QSflat', 'QSsharp', 'QSsharpslog', 'QSsharpsbar', 'SphSflat', 'SphSsharp','SphSsharpslog','SphSsharpsbar', 'CRPSflat', 'CRPSsharp','CRPSslog','CRPSsbar')
  }else{
    # Scores
    mLossh1 = -np$load('mScores/mScores_h1_TempGARCH-norm-t_iT_7305_iTest_2922_iS_17_fWIndicatorC_vFinal_bWarm_False.npy')
    mLossh3 = -np$load('mScores/mScores_h3_TempGARCH-norm-t_iT_7305_iTest_2922_iS_17_fWIndicatorC_vFinal_bWarm_False.npy')
    # Other settings
    vQ = c(1,2,4) # quantiles for which scores have been calculated
    vIdx = c(2, 1, 3, 4, 6, 5, 7, 8, 10, 9, 11, 12, 15, 14, 16, 17, 13) 
    lScoringRules=c('LogSflat', 'LogSsharp','LogSsharpslog','LogSsharpsbar','QSflat', 'QSsharp', 'QSsharpslog', 'QSsharpsbar', 'SphSflat', 'SphSsharp','SphSsharpslog','SphSsharpsbar', 'CRPSflat', 'CRPSsharp','CRPSslog','CRPSsbar','twCRPS')
  }
  lLoss = list(mLossh1, mLossh2, mLossh3)
  
  ## MCSpval for different values of q 
  lMethodNames = c('GARCH-N','GARCH-t','QGARCH-I-N','QGARCH-I-t','QGARCH-II-N','QGARCH-II-t')
  for(h in 1:length(lLoss)){
    mLoss = lLoss[[h]]
    for(q in 1:length(vQ)){
      dfMCSpval = MCSpvalCustomClimate(mLoss[q,,,], NA, iB, sStatistic, iK, lMethodNames,  vIdx, lScoringRules)
      sFileName = paste('MCSTables/Climate_MCSTableTest_','h',as.character(c(1,3)[h]),'_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'_tails',as.character(as.integer(bTails)),'.xlsx',sep = "")
      write_xlsx(dfMCSpval, sFileName)
    }
  }
}

################################################################################
#################### RISK MANAGEMENT ROBUSTNESS SCHECK #########################
################################################################################
if(bRiskManRobustCheck){
  
  ## Data
  mLossh1 = -np$load(paste('mScores/mScores_h1_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  mLossh5 = -np$load(paste('mScores/mScores_h5_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  
  ## Fixed parameters
  vQ = c(0.01, 0.05, 0.10, 0.15, 0.20, 0.25) # quantiles for which scores have been calculated
  
  # iTest = 1000
  iTest = 1000
  mLossh1 = -np$load(paste('mScores/mScores_h1_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  mLossh5 = -np$load(paste('mScores/mScores_h5_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  
  sStatistic = 'TR' # statistic, 'Tmax' or 'TR'
  iK = 20
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
  iK = 100
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
  sStatistic = 'Tmax' # statistic, 'Tmax' or 'TR'
  iK = 20
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
  iK = 100
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  } 
  # iTest = 750
  iTest = 750
  mLossh1 = -np$load(paste('mScores/mScores_h1_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  mLossh5 = -np$load(paste('mScores/mScores_h5_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  sStatistic = 'TR' # statistic, 'Tmax' or 'TR'
  iK = 20
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
  iK = 100
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
  #iTest = 1250
  iTest = 1250
  iK = 20
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
  iK = 100
  iTest = 1000
  mLossh1 = -np$load(paste('mScores/mScores_h1_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  mLossh5 = -np$load(paste('mScores/mScores_h5_TGARCH-RGARCH-norm-t_iT_6777_iTest_',as.character(iTest),'_vTGFinal_iP_0_p_0_qml_0_iTw_0_dRq_25_iS_16_vFinal.npy', sep=""))
  for(q in 1:length(vQ)){
    dfMCSpval = MCSpvalCustomRiskMan(mLossh1[q,,,], mLossh5[q,,,], iB,  sStatistic, iK)
    sFileName = paste('MCSTables/RiskManL_MCSTable_',as.character(sStatistic),'_iK',as.character(iK),'_iTest', as.character(iTest),'_q', as.character(vQ[q]*100),'.xlsx',sep = "")
    write_xlsx(dfMCSpval, sFileName)
  }
}