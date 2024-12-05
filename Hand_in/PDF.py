from scipy.stats import crystalball, norm, expon
import numpy as np 



def g_s(X,beta,m,mu,sigma):
    """
    Returns the truncated crystalball PDF, normalised over range [0,5]
    """
    pdf=crystalball(beta,m,loc=mu,scale=sigma)
    norm_factor = pdf.cdf(5)-pdf.cdf(0)
    trunc_pdf= pdf.pdf(X)/norm_factor
    return trunc_pdf


def h_s(Y,lam):
    """
    Returns the truncated exponential PDF, normalised over range [0,10]
    """
    pdf = expon(scale=1/lam)
    norm_factor = pdf.cdf(10)-pdf.cdf(0)
    trunc_pdf = pdf.pdf(Y)/norm_factor
    return trunc_pdf

def g_b(X):
    """
    Returns the uniform PDF, normalised over range [0,5]
    """
    pdf=1/5*np.ones_like(X)
    return pdf

def h_b(Y,mu_b,sigma_b):
    """
    Returns the truncated normal PDF, normalised over range [0,10]
    """
    pdf = norm(loc=mu_b,scale=sigma_b)
    norm_factor=pdf.cdf(10)-pdf.cdf(0)
    trunc_pdf = pdf.pdf(Y)/norm_factor  
    return trunc_pdf

def f_tot(X,Y,f,beta,m,mu,sigma,lam,mu_b,sigma_b):
    """
    Returns the total 2D PDF 
    """
    pdf = f*g_s(X,beta,m,mu,sigma)*h_s(Y,lam)+(1-f)*g_b(X)*h_b(Y,mu_b,sigma_b)
    return pdf

