#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy.integrate import dblquad
from scipy.stats import t,  multivariate_normal, chi2
from scipy.special import loggamma

class BivariateT:
    def __init__(self, vMean, mCov, dDf):
        """
        Initialize the bivariate t-distribution.
        :param vMean: Mean vector (length 2)
        :param mCov: 2x2 Covariance matrix
        :param dDf: Degrees of freedom
        """
        self.vMean = np.array(vMean)  # Mean vector
        self.mCov = np.array(mCov)    # Covariance matrix
        self.dDf = dDf                # Degrees of freedom
        self.iDim = 2                 # Fixed to bivariate case

    def pdfscalarout(self, vX):
        """
        Evaluate the PDF of the bivariate t-distribution at a given point.
        :param vX: Point to evaluate the PDF [x, y]
        :return: PDF value
        """
        vX = np.atleast_2d(vX) - self.vMean
        mCovInv = np.linalg.inv(self.mCov)
        dCovDet = np.linalg.det(self.mCov)
        vMahalanobis = np.einsum('ij,jk,ik->i', vX, mCovInv, vX)  # Mahalanobis distance
        dNumerator = np.exp(loggamma((self.dDf + self.iDim) / 2) - loggamma(self.dDf / 2))
        dDenominator = ((self.dDf * np.pi) ** (self.iDim / 2) *
                        np.sqrt(dCovDet) *
                        (1 + vMahalanobis / self.dDf) ** ((self.dDf + self.iDim) / 2))
        lPdf = dNumerator / dDenominator
        return lPdf[0]
    
    def pdf(self, mX):
        """
        Evaluate the PDF of the bivariate t-distribution at a given point or set of points.
        :param mX: Points to evaluate the PDF of shape (n, 2) or (2,)
        :return: PDF value(s). If (n, 2) is supplied, returns a length-n array.
        """
        # Ensure mX is (n, 2)
        mX = np.atleast_2d(mX)
        
        # Center mX by the mean
        mX_centered = mX - self.vMean
        
        # Precompute inverse and determinant of covariance
        mCovInv = np.linalg.inv(self.mCov)
        dCovDet = np.linalg.det(self.mCov)
    
        # Mahalanobis distance for each row
        # shape(mX_centered) = (n, 2)
        # shape(mCovInv)     = (2, 2)
        # -> shape(vMahalanobis) = (n,)
        vMahalanobis = np.einsum('ij,jk,ik->i', mX_centered, mCovInv, mX_centered)
    
        # Numerator
        # gamma((df + iDim)/2) / gamma(df/2)
        dNumerator = np.exp(
            loggamma((self.dDf + self.iDim) / 2.0) - loggamma(self.dDf / 2.0)
        )
        
        # Denominator term for each row
        # (1 + vMahalanobis / dDf)^((dDf + iDim)/2)
        # shape(...) = (n,)
        vDenominator = (
            (self.dDf * np.pi) ** (self.iDim / 2.0)
            * np.sqrt(dCovDet)
            * (1.0 + vMahalanobis / self.dDf) ** ((self.dDf + self.iDim) / 2.0)
        )
        
        # Finally, elementwise PDF values
        vPdf = dNumerator / vDenominator
        
        return vPdf  # shape (n,)

    def cdf(self, vX):
        """
        Compute the CDF of the bivariate t-distribution at a given point using numerical integration.
        :param vX: Upper bounds for the integration [x, y].
        :return: CDF value
        Note: We integrate in file
        """
        def pdf_bivariate_t(dX, dY):
            return self.pdf([dX, dY])
        
        # Integrate over the domain (-inf, vX[0]) x (-inf, vX[1])
        dCdfValue, dError = dblquad(
            func=pdf_bivariate_t,               # Function to integrate
            a=-np.inf,                          # Lower limit for x
            b=vX[0],                            # Upper limit for x
            gfun=lambda _: -np.inf,             # Lower limit for y
            hfun=lambda _: vX[1],               # Upper limit for y
            epsabs=1e-5, epsrel=1e-5            # Error tolerances
        )
        return dCdfValue
    
    def rvs(self, iNSamples):
        """
        Simulate random samples from the bivariate t-distribution.
        :param iNSamples: Number of samples to generate.
        :return: Array of shape (iNSamples, 2) with random samples.
        """
        # Generate samples from a standard multivariate normal distribution
        mMvnSamples = multivariate_normal.rvs(mean=np.zeros(self.iDim), cov=self.mCov, size=iNSamples)
        
        # Generate chi-squared random variables
        vChi2Samples = chi2.rvs(df=self.dDf, size=iNSamples)
        
        # Scale the multivariate normal samples
        mScaledSamples = mMvnSamples / np.sqrt(vChi2Samples[:, None] / self.dDf)
        
        # Shift by the mean vector
        return mScaledSamples + self.vMean


