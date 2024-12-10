# now we need to fit in the discriminant 
from numba_stats import norm as fnorm, expon as fexpon
from iminuit import Minuit, cost
import multiprocessing
import numpy as np
import os
import PDF
from iminuit.cost import BinnedNLL
from sweights import SWeight
from scipy.stats import crystalball, norm, expon
from tqdm import tqdm  


def spdf(X, beta, m, mu, sigma):
    """
    Signal probability density function (PDF), normalized over the range [0, 5].
    Uses g_s from the PDF module.
    """
    return PDF.g_s(X, beta, m, mu, sigma)


def bpdf(X):
    """
    Background probability density function (PDF), normalized over the range [0, 5].
    Uses g_b from the PDF module.
    """
    return PDF.g_b(X)


def tmodel(x, Ns, Nb, beta,m, mu, sg, comps=['S','B']):
    res = np.zeros_like(x)
    if 'S' in comps:
        res = res + Ns*spdf(x, beta, m, mu, sg)
    if 'B' in comps:
        res = res + Nb*bpdf(x)
    return res 

def tdensity(x, Ns, Nb,beta,m, mu, sg):
    return Ns+Nb, tmodel(x, Ns, Nb, beta,m, mu, sg)

def fit(x_values,sample_size):
    n2ll = cost.ExtendedUnbinnedNLL( x_values, tdensity )  #is this correct
    f=0.6
    N=sample_size
    mi = Minuit( n2ll, Ns = f*N, Nb = (1-f)*N, beta = 1, m= 1.4,mu = 3, sg= 0.3)
    mi.migrad()
    return mi

def find_y_values(mi, x_values, y_values):
    sf = lambda x: spdf(x, mi.values['beta'], mi.values['m'], mi.values['mu'], mi.values['sg'])
    bf = lambda x: bpdf(x)
    sy = mi.values['Ns']
    by = mi.values['Nb']
    sweighter = SWeight( x_values, pdfs=[sf,bf], yields=[sy,by], discvarranges=((0,5),),verbose=False, checks = False)

    sw = sweighter.get_weight(0, x_values)
    bw = sweighter.get_weight(1, x_values)

    ysw, ye = np.histogram( y_values, bins=50, range=[0,10], weights=sw )
    ybw, ye = np.histogram( y_values, bins=50, range=[0,10], weights=bw )

    ysw2, ye = np.histogram( y_values, bins=50, range=[0,10], weights=sw**2 )
    ybw2, ye = np.histogram( y_values, bins=50, range=[0,10], weights=bw**2 )

    cy = 0.5*(ye[1:]+ye[:-1])

    return ysw, ye

def h_s_cdf(Y, lam): #Do we need Ns
    pdf = expon(scale=1/lam)
    norm_factor = pdf.cdf(10)-pdf.cdf(0)
    trunc_cdf = (pdf.cdf(Y)-pdf.cdf(0))/norm_factor
    return trunc_cdf

def fit_lamdba(ye, ysw, sample_size):
    nll = BinnedNLL(ysw, ye, h_s_cdf)
    f=0.6
    N = sample_size
    mi = Minuit(nll, lam=0.3)
    mi.migrad()
    lam = mi.values['lam']
    lam_error = mi.errors['lam']
    return lam, lam_error

def run_fit_toys_sweight(sample_size, toys_data):
    folder_name = "sWeighting_Data"
    if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")
    print('Running fit for sample size:', sample_size)

    lam_values = []
    lam_errors = []
    for i, data_points in enumerate(tqdm(toys_data[sample_size])):
        x_values = data_points[:,0]
        y_values = data_points[:,1]

        mi = fit(x_values,sample_size)
        ysw ,ye = find_y_values(mi, x_values, y_values)
        lam , lam_err = fit_lamdba(ye, ysw, sample_size)
        lam_values.append(lam)
        lam_errors.append(lam_err)

    values_filename = os.path.join(folder_name, f"{sample_size}_lambda_values.npy")
    errors_filename = os.path.join(folder_name, f"{sample_size}_lambda_errors.npy")

    np.save(values_filename, lam_values)
    np.save(errors_filename, lam_errors)

    print(f"Saved values to {values_filename} and errors to {errors_filename}")
    print(f"Finished fitting for sample size: {sample_size}")

    return lam_values, lam_errors
    
    
def main():
    sample_sizes = [500, 1000, 2500, 5000, 10000]
    toys = {}
    data_dir = "Parametric_Bootstrapping_Data"
    for sample_size in sample_sizes:
        toys[sample_size] = []
        for i in range(250):
            path = f"{data_dir}/{sample_size}_data_points"
            data_points = np.load(f"{path}/{sample_size}_data_points_{i}.npy")
            toys[sample_size].append(data_points)

    # Run the fits in parallel
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(run_fit_toys_sweight, [(size,toys) for size in sample_sizes])

    

if __name__ == '__main__':
    main()

