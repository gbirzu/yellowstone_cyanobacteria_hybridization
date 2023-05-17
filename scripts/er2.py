import argparse
import numpy as np
import scipy.special as special
import simulations.series_summation as ssum
import pickle

'''
Implements algorithm from Song & Song (2007) to calculate <r^2>
for neutral model with recombination.

WARNING: This algorithm ignores finite sample sizes which leads
    to incorrect results that do not match simulations.
'''

def calculate_r2_fast(rho, theta, l_max=50, N=5):
    '''
    Note that convergence generally depends on rho and theta. For the values I've
    tested, l_max ~ 10 gives the first significant digit in almost all cases and 
    the default l_max = 50 gives at least two significant digits.
    '''
    Dpq = calculate_Dpq(rho, theta, l_max)
    al_matrix = 4 * Dpq[2, :, :][:, ::-1]
    r2_l = [np.trace(al_matrix, offset=l_max + 2)]
    for k in range(1, l_max + 3):
        r2_l.append(r2_l[-1] + np.trace(al_matrix, offset=l_max + 2 - k))
    Q5n = ssum.richardson_extrapolation(np.array(r2_l), N=N)
    return Q5n[l_max + 2 - 2 * N]

def calculate_r2(rho, theta, l_max=100):
    Dpq = calculate_Dpq(rho, theta, l_max)
    return 4 * np.sum(Dpq[2, :, :])

def calculate_Dpq(rho, theta, l_max):
    '''
    Calculates terms in E[r^2] up to level l_max using recursion formula Eq. (12).
    '''
    Dpq = initialize_Dpq(l_max, theta)
    for l in range(l_max + 1):
        if l % 2 == 0:
            n_max = l // 2
        else:
            n_max = (l - 1) // 2

        for n in range(n_max + 1):
            m = l - n
            # Set up system of n + 3 coupled linear eqs. using
            # f = D^k p^{m + 2 - k} q^{n + 2 - k} in master Eq. (2)
            # for 0 <= k <= n + 2

            # Set up equation in A * Dpq_mn = B form
            A, B = assign_Dpq_master_eqn_matrices(Dpq, rho, theta, m, n)
            Dpq_mn = np.linalg.solve(A, B)
            for k in range(n + 3):
                Dpq[k, m + 2 - k, n + 2 - k] = Dpq_mn[k]
                if n != m:
                    Dpq[k, n + 2 - k, m + 2 - k] = Dpq[k, m + 2 - k, n + 2 - k]
    return Dpq

def initialize_Dpq(l, theta):
    Dpq = np.zeros((l + 3, l + 3, l + 3))
    Dpq[0, 0, 0] = 1 # E[1] = 1
    Dpq[0, 1, 1] = 0.25 # Eq. 9
    n = np.arange(1, l + 3)
    Dpq[0, n, 0] = np.exp(special.gammaln(theta / 2 + n) - special.gammaln(theta / 2) - special.gammaln(theta + n) + special.gammaln(theta))
    Dpq[0, 0, n] = Dpq[0, n, 0]
    Dpq[0, n, 1] = Dpq[0, n, 0] / 2
    Dpq[0, 1, n] = Dpq[0, n, 1]
    return Dpq

def assign_Dpq_master_eqn_matrices(Dpq, rho, theta, m, n):
    Amn = np.zeros((n + 3, n + 3))
    Bmn = np.zeros(n + 3)

    for k in range(2, n + 2):
        i = m + 2 - k
        j = n + 2 - k
        Amn[k, k - 2] = -k * (k - 1)
        Amn[k, k - 1] = -4 * k * (k - 1)
        Amn[k, k] = k**2 + i * (i - 1 + theta) + j * (j - 1 + theta) + k * (1 + 4 * i + 4 * j + rho + 2 * theta)
        Amn[k, k + 1] = -2 * i * j
        Bmn[k] = (0.5 * (2 * i**2 + i * theta + 4 * k * i - 2 * i) * Dpq[k, i - 1, j] +
                0.5 * (2 * j**2 + j * theta + 4 * k * j - 2 * j) * Dpq[k, i, j - 1] +
                k * (k - 1) * (Dpq[k - 1, i, j] - 2 * Dpq[k - 1, i, j + 1] - 2 * Dpq[k - 1, i + 1, j]) +
                k * (k - 1) * (Dpq[k - 2, i + 1, j + 1] - Dpq[k - 2, i + 2, j + 1] - Dpq[k - 2, i + 1, j + 2]))

    # Apply BC
    k = 0
    i = m + 2
    j = n + 2
    Amn[k, k] = k**2 + i * (i - 1 + theta) + j * (j - 1 + theta) + k * (1 + 4 * i + 4 * j + rho + 2 * theta)
    Amn[k, k + 1] = -2 * i * j
    Bmn[k] = (0.5 * (2 * i**2 + i * theta + 4 * k * i - 2 * i) * Dpq[k, i - 1, j] +
            0.5 * (2 * j**2 + j * theta + 4 * k * j - 2 * j) * Dpq[k, i, j - 1])

    k = 1
    i = m + 1
    j = n + 1
    Amn[k, k - 1] = -4 * k * (k - 1)
    Amn[k, k] = k**2 + i * (i - 1 + theta) + j * (j - 1 + theta) + k * (1 + 4 * i + 4 * j + rho + 2 * theta)
    Amn[k, k + 1] = -2 * i * j
    Bmn[k] = (0.5 * (2 * i**2 + i * theta + 4 * k * i - 2 * i) * Dpq[k, i - 1, j] +
            0.5 * (2 * j**2 + j * theta + 4 * k * j - 2 * j) * Dpq[k, i, j - 1])

    k = n + 2
    i = m - n
    j = 0
    Amn[k, k - 2] = -k * (k - 1)
    Amn[k, k - 1] = -4 * k * (k - 1)
    Amn[k, k] = k**2 + i * (i - 1 + theta) + j * (j - 1 + theta) + k * (1 + 4 * i + 4 * j + rho + 2 * theta)
    Bmn[k] = (0.5 * (2 * i**2 + i * theta + 4 * k * i - 2 * i) * Dpq[k, i - 1, j] +
            k * (k - 1) * (Dpq[k - 1, i, j] - 2 * Dpq[k - 1, i, j + 1] - 2 * Dpq[k - 1, i + 1, j]) +
            k * (k - 1) * (Dpq[k - 2, i + 1, j + 1] - Dpq[k - 2, i + 2, j + 1] - Dpq[k - 2, i + 1, j + 2]))

    return Amn, Bmn

def calculate_sigma2_terms(rho, theta):
    Dpq = initialize_Dpq(0, theta)
    A, B = assign_Dpq_master_eqn_matrices(Dpq, rho, theta, 0, 0)

    Dpq_mn = np.linalg.solve(A, B)
    for k in range(3):
        Dpq[k, 2 - k, 2 - k] = Dpq_mn[k]

    # <D^2>
    D2 = Dpq[2, 0, 0]
    # <p(1 - p)q(1 - q)> = <pq> - <pq^2> - <p^2q> - <p^2q^2>
    denominator = Dpq[0, 1, 1] - Dpq[0, 1, 2] - Dpq[0, 2, 1] + Dpq[0, 2, 2]
    return (D2, denominator), Dpq

def sigma2_theory(rho, theta):
    '''
    Calculates sigma_d^2 using Eq. 17 from Song & Song (2007) (originally derived by Ohta & Kimura).
    '''
    numerator = 10 + rho + 4 * theta
    denominator = 22 + 13 * rho + rho**2 + 6 * theta * rho + 32 * theta + 8 * theta**2
    return numerator / denominator

def calculate_large_lmax_values(output_file):
    theta_list = [0.01, 0.1, 1]
    rho_list = [0, 1, 10, 100]

    results = []
    for theta in theta_list:
        for rho in rho_list:
            r2_avg = r2(rho, theta, l_max=700)
            print(f'rho = {rho}, theta = {theta}, r^2 = {r2_avg}')
            results.append((rho, theta, r2_avg))

    pickle.dump(results, open(output_file, 'wb'))

def test_all():
    # Test speed up from Richardson summation
    Dpq = calculate_Dpq(10, 0.2, 200)
    r2_l200 = 4 * np.sum(Dpq[2, :, :])
    print(r2_l200)

    #for rho in [0, 1, 2, 5, 10, 20]:
    for rho in [0, 10]:
        #for theta in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0]:
        for theta in [0.1, 1.0, 10.0]:
            r2_q5n = calculate_r2_fast(rho, theta, 300)
            print(rho, theta, r2_q5n)
            print('\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-r', '--rho', default=0, type=float, help='Rho.')
    parser.add_argument('-t', '--theta', default=1, type=float, help='Theta.')
    parser.add_argument('-l', '--l_max', default=100, type=int, help='Truncation level for recursion relation.')
    parser.add_argument('-m', '--measure', default='r2', help='LD measure [r2, sigma2].')
    parser.add_argument('-s', '--slow', action='store_true')
    parser.add_argument('-o', '--output_file', default=None, help='Calculate r^2 for preset rho and theta at l=500 and save to file.')
    parser.add_argument('--test_all', action='store_true')
    args = parser.parse_args()


    if args.test_all == True:
        test_all()
    elif args.output_file:
        calculate_large_lmax_values(args.output_file)
    else:
        if args.slow == True:
            print(f'Calculating r^2 for rho = {args.rho}, theta = {args.theta}...')
            Dpq = calculate_Dpq(args.rho, args.theta, args.l_max)
            print(f'\tr^2 = {4 * np.sum(Dpq[2, :, :])}')
        elif args.measure == 'r2':
            print(f'Calculating r^2 for rho = {args.rho}, theta = {args.theta} (with convergence acceleration)...')
            r2 = calculate_r2_fast(args.rho, args.theta, l_max=args.l_max)
            print(f'\tr^2 = {r2}')
        else:
            print(f'Calculating sigma_d^2 for rho = {args.rho}, theta = {args.theta}...')
            #sigma2_terms, Dpq = calculate_sigma2_terms(args.rho, args.theta)
            print(f'\t{sigma2_theory(args.rho, args.theta)}')
