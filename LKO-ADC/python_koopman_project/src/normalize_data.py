import numpy as np

def normalize_data(data, params=None):
    """
    Normalizes the data using mean and standard deviation (z-score).
    If params are not provided, it calculates them from the data.
    """
    if params is None:
        mu = np.mean(data, axis=1, keepdims=True)
        sigma = np.std(data, axis=1, keepdims=True)
        # Avoid division by zero for constant features
        sigma[sigma == 0] = 1
        params = {'mu': mu, 'sigma': sigma}

    normalized_data = (data - params['mu']) / params['sigma']
    return normalized_data, params

def denormalize_data(data, params):
    """
    Reverses the normalization using the provided mean and sigma.
    """
    return data * params['sigma'] + params['mu']