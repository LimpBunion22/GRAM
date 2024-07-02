import numpy as np


def evaluate_base(weights, bias, in_values):

    n_data = in_values.shape[0]
    n_bases = weights.shape[0]
    out_values = np.zeros((n_data,n_bases))
    for d in range(n_data):
        for b in range(n_bases):
            out_values[d,b] = 1/np.prod(1+np.abs(weights[b,:]*(in_values[d,:] - bias[b,:])))
    return out_values


def evaluate_orts(base_values, betas):

    n_data = base_values.shape[0]
    n_bases = base_values.shape[1]
    out_values = np.zeros((n_bases,n_data))
    for d in range(n_data):
        for b in range(n_bases):
            out_values[b,d] = np.sum(base_values[d,:]*betas[b,:])
    return out_values
