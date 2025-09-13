#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: Multivariate weight functions
"""

import numpy as np

###########################################################  
def fWIndicatorLBivProd(mY, vParamsW, mR):
    """
    Purpose
    ----------
    Indicator weight function, being one for y1 <= r1 and y1 <= r2

    Parameters
    ----------
    mY :           matrix, data
    vParamsW :     vector, other parameters weight function         
    mR :           matrix, threshold values
    
    Returns
    ----------
    iT x len(mR[0,:]) matrix with weights
    """

    if mY.ndim ==2 and mR.ndim ==2:
        vWbool = (mY[:,0].T <= mR[0,:]) * (mY[:,1].T <= mR[1,:])
    elif mY.ndim ==2 and mR.ndim == 1:
        vWbool = (mY[:,0].T <= mR[0]) * (mY[:,1].T <= mR[1])
    else:
        vWbool = (mY[0].T <= mR[0]) * (mY[1].T <= mR[1]) 
    
    return np.ones(vWbool.shape) * vWbool

###########################################################  
def fWLogisticLBivProd(mY, vParamsW, mR):
    """
    Purpose
    ----------
    Product of logistics weight functions, 

    PParameters
    ----------
    mY :           matrix, data
    vParamsW :     vector, other parameters weight function         
    mR :           matrix, threshold values
    
    Returns
    ----------
    iT x len(mR[0,:]) matrix with weights
    """
    
    dA = vParamsW
    
    if mY.ndim ==2 and mR.ndim ==2:
        vW = 1/(1+np.exp(dA * (mY[:,0].T - mR[0,:]))) * 1/(1+np.exp(dA*(mY[:,1].T - mR[1,:])))
    elif mY.ndim ==2 and mR.ndim == 1:
        vW = 1/(1+np.exp(dA * (mY[:,0].T - mR[0]))) * 1/(1+np.exp(dA*(mY[:,1].T - mR[1])))
    else:
        vW = 1/(1+np.exp(dA * (mY[0].T - mR[0]))) * 1/(1+np.exp(dA*(mY[1].T - mR[1])))
    
    return vW