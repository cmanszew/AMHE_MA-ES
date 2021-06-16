import numpy as np
import math


def identity_matrix(dim):
    return np.identity(dim)


def selection_angular_brackets(offspring_obj, index_l, mu):
    rows = offspring_obj.shape[0]
    w_l = np.zeros(rows)
    for l in range(rows):
        w_l[index_l[l]] = compute_w_l(l, mu)
    return angular_brackets(offspring_obj, w_l)


def angular_brackets(a, w):
    """
    :param d_l: matrix lambda rows, N columns
    :param w_l: column vector of weights
    :return:
    """
    acc = a[0] * w[0]
    for i in range(1, len(a)):
        acc = acc + a[i] * w[i]
    return acc


def compute_w_l(l, miu):
    if l < miu:
        return 1.0 / miu
    else:
        return 0.0


def maes(fitnessfct, dim, initial_pop=None, maxfun=1000, seed=44, midpoint=False, step=False, red=False):
    # Uer defined input parameters
    N = dim
    sigma = 0.3

    # Strategy parameter setting: Selection
    lam = int(4 + math.floor(3 * math.log(N)))
    if red is True:
        lam = lam - 1
    mu = lam / 2
    weights = np.array([0.0] * int(math.floor(mu)))
    for iw in range(len(weights)):
        weights[iw] = math.log(mu + 1 / 2) - math.log(iw + 1)
    mu = math.floor(mu)
    weights = weights / weights.sum()
    mueff = np.sum(weights) ** 2 / np.sum(weights ** 2)

    # Strategy parameter setting: Adaptation
    cs = (mueff + 2) / (N + mueff + 5)
    c1 = 2 / ((N + 1.3) ** 2 + mueff)
    cmu = min(1 - c1, 2 * (mueff - 2 * 1 / mueff) / ((N + 2) ** 2 + mueff))
    cw = cmu
    damps = 1 + 2 * max(0, math.sqrt((mueff - 1) / (N + 1)) - 1) + cs

    # Initialize dynamic (internal) strategy parameters and constants
    chiN = N ** 0.5 * (1 - 1 / (4 * N) + 1 / (21 * N ** 2))

    if initial_pop is None:
        y = np.ones((1, N))  # ones_matrix(population_size, dim)
    else:
        y = initial_pop

    g = 0
    s = np.zeros(dim)
    M = identity_matrix(dim)

    rng = np.random.default_rng(seed)
    last_midpoint_eval = 0
    while g < maxfun:
        # Generate offsprings and calculate their fitness
        z_l = np.zeros((lam, N))
        d_l = np.zeros((lam, N))
        y_l = np.zeros((lam, N))
        f_l = np.zeros(lam)
        for l in range(lam):
            z_l[l] = rng.standard_normal(N)
            d_l[l] = M @ z_l[l]
            y_l[l] = y + sigma * d_l[l]
            f_l[l] = fitnessfct(y_l[l])
            g = g + 1

        ranking = np.argsort(f_l)
        y = y + sigma * selection_angular_brackets(d_l, ranking, mu)
        if midpoint is True:
            if step is False or (step is True and (g > last_midpoint_eval + 20 * N)):
                last_midpoint_eval = g
                fitnessfct(y)
                g = g + 1
        s = (1 - cs) * s + math.sqrt(mueff * cs * (2 - cs)) * selection_angular_brackets(z_l, ranking, mu)
        z3dm = np.zeros((lam, N, N))
        for i, z in enumerate(z_l):
            z3dm[i] = np.outer(z, z)
        M = M @ (identity_matrix(N) + c1 / 2 * (np.outer(s, s) - identity_matrix(N))
                 + cw / 2 * (selection_angular_brackets(z3dm, ranking, mu) - identity_matrix(N)))
        sigma = sigma * math.exp(cs / damps * (np.linalg.norm(s) / chiN - 1))


def maes_midpoint(fitnessfct, dim, initial_pop=None, maxfun=1000):
    maes(fitnessfct, dim, initial_pop, maxfun, midpoint=True)


def maes_midpoint_step(fitnessfct, dim, initial_pop=None, maxfun=1000):
    maes(fitnessfct, dim, initial_pop, maxfun, midpoint=True, step=True)


def maes_midpoint_red(fitnessfct, dim, initial_pop=None, maxfun=1000):
    maes(fitnessfct, dim, initial_pop, maxfun, midpoint=True, red=True)


if __name__ == '__main__':
    maes(lambda x: np.sum(x ** 2), 2)
    # vl = np.array([[1.,2.],[3.,4.],[5.,6.]])
    # ml = [np.array([[1.,2.],[3.,4.]])]
    # a = angular_brackets(vl, np.array([0.5, 0.5, 0.0]))
# x_1**2 + x_2**2
# [
#   [1.0, 2.0]
#   [1.5, 2.2]
#   [1.6, 2.3]
# ]
