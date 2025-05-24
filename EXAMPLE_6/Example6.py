#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Purpose: MW 

"""

# Fundamentals
import numpy as np
from scipy.stats import norm
from scipy.integrate import quad
from scipy.optimize import root_scalar

# System
import os

# Plots
import matplotlib.pyplot as plt
from matplotlib import rc
os.environ["PATH"] += os.pathsep + "/Library/TeX/texbin"
rc('font', **{'family': 'serif', 'serif': ['Computer Modern'], 'size':'14'})
rc('text', usetex=True)

###############################################################################
def fnDensityP(vy, dSigmaP):
    """
    Purpose
    -------
    Compute the density of a standard normal distribution with standard deviation dSigmaP.
   
    Parameters
    ----------
    vy : array, vector of evaluation points
    dSigmaP : double, standard deviation of the distribution
   
    Returns
    -------
    array, density values at vy
    """
    return norm.pdf(vy, 0, dSigmaP)

###############################################################################
def fnDensityF(vy, dSigmaF):
    """
    Purpose
    -------
    Compute the density of a standard normal distribution with standard deviation dSigmaF.
   
    Parameters
    ----------
    vy : array, vector of evaluation points
    dSigmaP : double, standard deviation of the distribution
   
    Returns
    -------
    array, density values at vy
    """
    return norm.pdf(vy, 0, dSigmaF)

###############################################################################
def fnLogRatio(vy, dSigmaP, dSigmaF):
    """
    Purpose
    -------
    Compute the log-likelihood ratio between two normal densities with standard deviations dSigmaP and dSigmaF.
    
    Parameters
    ----------
    vy : array, vector of evaluation points
    dSigmaP : double, standard deviation of the numerator density
    dSigmaF : double, standard deviation of the denominator density
    
    Returns
    -------
    array, log-likelihood ratio values at vy
    """
    dP = fnDensityP(vy, dSigmaP)
    dF = fnDensityF(vy, dSigmaF)
    return np.log(dP / dF)

###############################################################################
def fnDifferenceInExpectations(dAlpha, dSigmaP, dSigmaF):
    """
    Purpose
    -------
    Compute the difference in expectations of the log-likelihood ratio over critical intervals defined by dAlpha.
    
    Parameters
    ----------
    dAlpha : double, significance level defining the critical intervals
    dSigmaP : double, standard deviation of the numerator density
    dSigmaF : double, standard deviation of the denominator density
    
    Returns
    -------
    double, difference in expectations over critical intervals
    """
    p = lambda vy: norm.pdf(vy, 0, dSigmaP)
    f = lambda vy: norm.pdf(vy, 0, dSigmaF)
    log_ratio = lambda vy: np.log(p(vy) / f(vy))

    vAF = [norm.ppf(dAlpha/2, 0, dSigmaF), norm.ppf(1 - dAlpha/2, 0, dSigmaF)]
    vAP = [norm.ppf(dAlpha/2, 0, dSigmaP), norm.ppf(1 - dAlpha/2, 0, dSigmaP)]
    vD = [vAP[0],vAF[0],vAF[1],vAP[1]] 

    dIntAF, _ = quad(lambda vy: log_ratio(vy) * p(vy), vAF[0], vAF[1])
    dIntD1, _ = quad(lambda vy: np.log(p(vy) / dAlpha) * p(vy), vD[0], vD[1])
    dIntD2, _ = quad(lambda vy: np.log(p(vy) / dAlpha) * p(vy), vD[2], vD[3])

    dIntD = dIntD1 + dIntD2

    return dIntAF + dIntD

###############################################################################
### main
def main():    
    
    ## Cleaning
    plt.close('all')
    
    # Parameters
    dSigmaF = np.sqrt(1/2)
    dSigmaP = 1

    # Extended range of alpha values
    vAlphaValuesExtended = np.linspace(0.001, 1, 100)
    vDifferencesExtended = [fnDifferenceInExpectations(dAlpha, dSigmaP, dSigmaF) for dAlpha in vAlphaValuesExtended]

    # Plot the difference in expectations
    plt.figure(figsize=(10, 6))
    plt.plot(vAlphaValuesExtended, vDifferencesExtended, label='Difference in Expectations', color='black')
    plt.axhline(0, color='red', linestyle='--', label='Zero Difference')
    plt.xlabel('$\\alpha$')
    plt.ylabel('Difference in Expectations')
    plt.title('Difference in Expectations vs. $\\alpha$')
    plt.legend()
    plt.grid(True)
    plt.savefig('Figures/DifferenceInExpectationsExtended.pdf')

    # Find the root of the function where the difference in expectations is zero
    result = root_scalar(fnDifferenceInExpectations, args=(dSigmaP, dSigmaF), bracket=[0.001, 0.99], method='brentq')
    dAlphaZeroIntersection = result.root

    sResultMessage = f"The value of alpha where the difference in expectations is zero: {dAlphaZeroIntersection}"
    print(sResultMessage)

###########################################################
### start main
if __name__ == "__main__":
    main()