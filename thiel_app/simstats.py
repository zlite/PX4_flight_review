# this is the module that does the sim-to-real comparison stats. Feel free to write your own!

import numpy as np
import math
from sklearn.decomposition import PCA


def tic(x_r, x_s):
    """
    Equation (1) (Thiel Inequality Coefficient) from paper
    
    :param x_r: reference output
    :param x_s: simulation output
    :return: the coefficient of TIC 
    """
    numerator = ((x_r - x_s)**2).sum()**(1/2)
    denominator = ((x_r**2).sum()**(1/2) + (x_s**2).sum()**(1/2))
    return numerator / denominator


def rate_of_change(x, t_Δ=1):
    """
    :param x: a series
    :param t_Δ: the intervals between each observation (series or constant)
    :return: rate of change for x
    """
    return np.diff(x) / t_Δ


def tic_improved(x_r, x_s):
    """
    Equation (2) from paper
    
    :param x_r: reference output
    :param x_s: simulation output
    :return: the IMPROVED coefficient of TIC 
    """
    xrss = (x_r**2).sum()**(1/2)
    xsss = (x_s**2).sum()**(1/2)
    if xrss == 0:
        return xsss
    else:
        return ((x_r - x_s)**2).sum()**(1/2) / xrss
    
    
def squashed_tic_improved(x_r, x_s, ξ):
    """
    Equation (3) from paper
    
    :param x_r: reference output
    :param x_s: simulation output
    :param ξ: squash control parameter
    :return: the squashed IMPROVED coefficient of TIC 
    """
    return np.exp(-ξ * tic_improved(x_r, x_s))



def make_matrix_A(x_r, simulations, ξ, Δ_t=1):
    """
    Equation (9) from paper
    
    :param x_r: reference output
    :param simulations: simulation outputs
    :param t_Δ: the intervals between each observation (series or constant)
    :param ξ: squash control parameter
    :return: the position/trend matrix 
    """
    A = []
    for x_s in simulations:
        a_i1 = squashed_tic_improved(x_r, x_s, ξ)
        a_i2 = squashed_tic_improved(rate_of_change(x_r, Δ_t), rate_of_change(x_s, Δ_t), ξ)
        A.append([a_i1, a_i2])
        
    return np.array(A)


def mean_centered(A):
    """
    Equation (10) from paper.
    
    :param A: a matrix
    :return: the matrix with columns mean-centered
    """
    return A / np.mean(A, axis=0)


def make_matrix_S(x_r, simulations, ξ, Δ_t=1):
    """
    Equation (11) from paper
    
    :param x_r: reference output
    :param simulations: simulation outputs
    :param ξ: squash control parameter
    :param t_Δ: the intervals between each observation (series or constant)
    :return: the position/trend matrix 
    """
    return mean_centered(make_matrix_A(x_r, simulations, ξ, Δ_t))


def compute_y(x_r, simulations, ξ, Δ_t=1):
    """
    Equation (18) from paper
    
    :param x_r: reference output
    :param simulations: simulation outputs
    :param ξ: squash control parameter
    :param t_Δ: the intervals between each observation (series or constant)
    :return: the y value for each simulation
    """
    S = make_matrix_S(x_r, simulations, ξ, Δ_t=1)
    k = PCA(1).fit(S).components_[0]
    return S @ k

def sim2real_stats(data):
    simdata = data['sim']
    realdata = data['real']
    simdata.dropna(inplace=True)  # delete empty rows
    realdata.dropna(inplace=True)  
    real = np.array(realdata)
    sim = np.array(simdata)
    stats = 0
    if (len(real) != 0) and (len(sim) !=0):  # avoid divide by zero errors
        sum1 = 0
        sum2 = 0
        sum3 = 0
        length = min(len(real), len(sim))  # only compare over the overlapping range of the two datasets, which is the smallest of the two
        for n in range(length):
            sum1 = sum1 + (real[int(n)]-sim[int(n)])**2
            sum2 = sum2 + real[int(n)]**2
            sum3 = sum3 + sim[int(n)]**2
        sum1 = 1/length * sum1
        sum2 = 1/length * sum2
        sum3 = 1/length * sum3
        sum1 = math.sqrt(sum1)
        sum2 = math.sqrt(sum2)
        sum3 = math.sqrt(sum3)
        stats = sum1/(sum2 + sum3)
        stats = round(stats,3)
    return stats

def sim2real_stats2(data):  # this is the Song variation of Thiel: https://drive.google.com/file/d/1XY8aZz89emFt-LAuUZ2pjC1GHwRARr9f/view
    numerator = 0
    denominator = 0
    simdata = data['sim']
    realdata = data['real']
    simdata.dropna(inplace=True)  # delete empty rows
    realdata.dropna(inplace=True)  
    real = np.array(realdata)
    sim = np.array(simdata)
    length = min(len(real), len(sim))  # only compare over the overlapping range of the two datasets, which is the smallest of the two
    for i in range(length-1):
        numerator = numerator + ((real[i+1]-real[i])-(sim[i+1]-sim[i]))**2
        denominator = denominator + (real[i+1] - real[i])**2
    
    numerator = math.sqrt(numerator)
    denominator = math.sqrt(denominator)
    total = round(numerator/denominator,3)
    return total