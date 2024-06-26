import numpy as np


def evaluate_base(weights, bias, n_values = 200):

    values = np.zeros(n_values)
    i = 0
    for x in np.linspace(-1,1,n_values):
        values[i] = 1/(1+np.abs(weights[0]*(x - bias[0])))
        i += 1
    return values


def evaluate_orts(base_values, betas, o):

    values = np.zeros(len(base_values[0]))
    i = 0
    for i in range(len(base_values[0])):
        for j in range(o+1):
            values[i] += base_values[j][i]*betas[o,j]
        i += 1
    return values
