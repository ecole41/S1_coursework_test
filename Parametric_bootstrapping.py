from tqdm import tqdm 
from iminuit import Minuit
from iminuit.cost import ExtendedUnbinnedNLL
from scipy.stats import crystalball, norm, expon
import matplotlib.pyplot as plt
import numpy as np
import multiprocessing
import os

mu_true = 3
sigma_true = 0.3
beta_true= 1
m_true = 1.4
f_true =0.6
lam_true=0.3
mu_b_true =0
sigma_b_true = 2.5

def g_s(X,beta,m,mu,sigma):
    pdf=crystalball(beta,m,loc=mu,scale=sigma)
    norm_factor = pdf.cdf(5)-pdf.cdf(0)
    trunc_pdf= pdf.pdf(X)/norm_factor
    return trunc_pdf


def h_s(Y,lam):
    pdf = expon(scale=1/lam)
    norm_factor = pdf.cdf(10)-pdf.cdf(0)
    trunc_pdf = pdf.pdf(Y)/norm_factor
    return trunc_pdf

def g_b(X):
    pdf=1/5*np.ones_like(X)
    return pdf

def h_b(Y,mu_b,sigma_b):
    pdf = norm(loc=mu_b,scale=sigma_b)
    norm_factor=pdf.cdf(10)-pdf.cdf(0)
    trunc_pdf = pdf.pdf(Y)/norm_factor  
    return trunc_pdf

def f_tot(X,Y,f,beta,m,mu,sigma,lam,mu_b,sigma_b):
    pdf = f*g_s(X,beta,m,mu,sigma)*h_s(Y,lam)+(1-f)*g_b(X)*h_b(Y,mu_b,sigma_b)
    return pdf


def density(data,N,f,beta,m,mu,sigma,lam,mu_b,sigma_b):
    X,Y =data
    return N, N*f_tot(X,Y,f,beta,m,mu,sigma,lam,mu_b,sigma_b)

def generate(N,f,beta,m,mu,sigma,lam,mu_b,sigma_b):
    accepted_points = []
    N=int(N)
    X=np.linspace(0,5,100)
    Y=np.linspace(0,10,100)
    X, Y = np.meshgrid(X, Y)
    Z = f_tot(X, Y, f=f_true, beta=beta_true, m=m_true, mu=mu_true, sigma=sigma_true, lam=lam_true, mu_b=mu_b_true, sigma_b=sigma_b_true)
    max_value = np.max(Z)
    while len(accepted_points)<N:
        x_batch = np.random.uniform(0, 5, int(N/2))
        y_batch = np.random.uniform(0, 10, int(N/2))
        y_guess = np.random.uniform(0, max_value, int(N/2))
        y_val = f_tot(x_batch, y_batch, f, beta, m, mu, sigma, lam, mu_b, sigma_b)
        accepted_indices = y_guess <= y_val
        # Add accepted points to the list
        accepted_points.extend(zip(x_batch[accepted_indices], y_batch[accepted_indices]))
    accepted_points = np.array(accepted_points[:N])
    return accepted_points


def fit(dset,N,f,beta,m,mu,sigma,lam,mu_b,sigma_b):
    nll = ExtendedUnbinnedNLL(dset.T, density)
    mi = Minuit(nll , N, f ,beta ,m ,mu ,sigma ,lam ,mu_b ,sigma_b)
    mi.migrad ()
    mi.hesse ()
    return mi

def fit_toys(Ntoy,sample_size,f,beta,m,mu,sigma,lam,mu_b,sigma_b):
    values =[]
    errors = []
    Num=np.random.poisson(sample_size, Ntoy)
    toys = [ generate(Num[_],f,beta,m,mu,sigma,lam,mu_b,sigma_b) for _ in range(Ntoy) ]
    for toy in tqdm(toys):
        mi_t = fit(toy,sample_size,f,beta,m,mu,sigma,lam,mu_b,sigma_b)
        values.append( list(mi_t.values) )
        errors.append( list(mi_t.errors) )
    return values, errors


def run_fit_toys(sample_size,original_values,Ntoy):
        folder_name = "Parametric_Bootstrapping_Data"

        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            print(f"Created folder: {folder_name}")

        print(f"Running fit_toys for sample size: {sample_size}")
        f,beta, m, mu, sigma, lam, mu_b, sigma_b = original_values[1:] # Get the initial values
        values, errors = fit_toys(Ntoy, sample_size, f=f, beta=beta, m=m, mu=mu, sigma=sigma, lam=lam, mu_b=mu_b, sigma_b=sigma_b)
       
        filename = os.path.join(folder_name, f"{sample_size}_values_and_errors.txt")
    
    
        # Save values and errors to the file
        with open(filename, 'w') as file:
            file.write(f"Sample size: {sample_size}\n")
            file.write("Values:\n")
            file.write(f"{values}\n")  # Write the values
            file.write("Errors:\n")
            file.write(f"{errors}\n")  # Write the errors
        print(f"Results saved to {filename}")
        return sample_size



def main():
    sample_sizes = [500, 1000, 2500, 5000, 10000]
    original_sample = generate(100000, f_true, beta_true, m_true, mu_true, sigma_true, lam_true, mu_b_true, sigma_b_true)
    original_values = fit(original_sample, 100000, f_true, beta_true, m_true, mu_true, sigma_true, lam_true, mu_b_true, sigma_b_true)
    print(f"Original values: {original_values.values}")
    # for size in sample_sizes:
    #      run_fit_toys(size,original_values.values,250)

    # Use multiprocessing to run the function in parallel
    # Parallelizing the loop using multiprocessing
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.starmap(run_fit_toys, [(size, original_values.values, 250) for size in sample_sizes])

if __name__ == '__main__':
    main()




