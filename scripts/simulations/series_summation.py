import argparse
import numpy as np
import scipy.special as special
import pickle


'''
This module contains tools to do fast summation of convergent series.
For details on methods used see Bender & Orszag, Ch. 8.
'''

def first_order_shanks(An):
    '''
    Simple Shanks transform based on partial sum A_n = A + a * q^n.
    Defines new partial sum series S_n = S_n(A_{n-1}, A_n, A_{n+1}) using Eq. 8.1.3 from
    Bender & Orszag.
    '''
    if len(An) < 2:
        print(f'Partial sum sequence {An} is too short! Exiting.')
        return None
    else:
        n = np.arange(1, len(An) - 1)
        Sn = np.zeros(len(An))
        Sn[n] = (An[n + 1]*An[n - 1] - An[n]**2) / (An[n - 1] - 2 * An[n] + An[n + 1])
        return Sn[:-1]

def richardson_extrapolation(An, N=1):
    '''
    Uses Richardson extrapolation to calculate series sum (Eq. 8.1.16 from
    Bender & Orszag). This is appropriate when the series converges 
    as a power law in n.
    '''
    if len(An) < 2:
        print(f'Partial sum sequence {An} is too short! Exiting.')
        return None
    else:
        n_array = np.arange(1, len(An) - N)
        k = np.arange(N + 1)
        Q0 = np.zeros(len(An))
        for n in n_array:
            Q0[n] = np.sum(((-1)**(k + N) * An[n + k] * (n + k)**N) / np.exp(special.gammaln(k + 1) + special.gammaln(N - k + 1)))
        return Q0

def slow_exponentially_convergent(z, n=0, exact=False):
    '''
    Generates nth partial sum from Taylor series of A(z) = 1/((z + 1)*(z + 2)). 
    See Eq. (8.1.1) from Bender & Orszag.
    '''

    A = 1 / ((z + 1)*(z + 2))
    if exact == False:
        A_1n = -(-z)**(n + 1) / (z + 1)
        A_2n = +(-z / 2)**(n + 1) / (z + 2)
        Az = A + A_1n + A_2n
    else:
        Az = A
    return Az

def slow_powerlaw_convergent(n=0, exact=False):
    '''
    Generates nth partial sum of the series 1 + 1/(2^2) + 1/(3^2) + ... = pi^2/6.
    '''

    if exact == False:
        k = np.arange(1, n + 1)
        A = np.sum(1 / (k**2))
    else:
        A = np.pi**2 / 6
    return A

def test_all():
    z = 0.99 # value of function to evaluate
    Az = slow_exponentially_convergent(z, exact=True)
    print(Az)

    N = 20 # max number of terms in series
    n = np.arange(N + 2)
    An = slow_exponentially_convergent(z, n)
    print(An[:-1])
    
    Sn = first_order_shanks(An)
    print(Sn)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--test_all', action='store_true')
    args = parser.parse_args()


    if args.test_all == True:
        test_all()
