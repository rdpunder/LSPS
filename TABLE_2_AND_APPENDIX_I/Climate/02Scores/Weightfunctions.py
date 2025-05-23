#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: File containing weight functions 
  
"""

# Fundamentals
import numpy as np  

###########################################################  
def fWIndicatorL(vY, vParamsW, vR):
    """
    Purpose
    ----------
    Indicator weight function, being one for y < r

    Parameters
    ----------
    mY :            matrix, data
    vParamsW :      vector, other parameters weight function         
    vR :            vector, threshold values
    
    Return
    ----------
    mY shape x len(vR) matrix with weights
    """
    
    vWbool = vY < vR 
    return np.ones(vWbool.shape) * vWbool

###########################################################  
def fWIndicatorR(vY, vParamsW, vR):
    """
    Purpose
    ----------
    Indicator weight function, being one for y > r

    Parameters
    ----------
    mY :            matrix, data
    vParamsW :      vector, other parameters weight function         
    vR :            vector, threshold values
    
    Return
    ----------
    mY shape x len(vR) matrix with weights
    """
    
    vWbool = vY > vR 
    return np.ones(vWbool.shape) * vWbool

###########################################################  
def fWIndicatorC(mY, vParamsW, dR):
    """
    Purpose
    ----------
    Indicator weight function, being one for c - r <= y <= c + r

    Parameters
    ----------
    mY :            matrix, data
    vParamsW :      vector, other parameters weight function         
    vR :            vector, threshold values
    
    Return
    ----------
    mY shape x len(vR) matrix with weights
    """
    
    dC = vParamsW
    mWbool = (mY > (dC-dR)) * (mY < (dC + dR))
    return np.ones(mWbool.shape) * mWbool

###########################################################  
def fWIndicatorTails(mY, vParamsW, dR):
    """
    Purpose
    ----------
    Indicator weight function, being zero for target - r <= y <= target + r

    Parameters
    ----------
    mY :            matrix, data
    vParamsW :      vector, other parameters weight function         
    vR :            vector, threshold values
    
    Return
    ----------
    mY shape x len(vR) matrix with weights
    """
    
    dC = vParamsW
    mWbool = 1 -  (mY > (dC-dR)) * (mY < (dC + dR))
    return np.ones(mWbool.shape) * mWbool