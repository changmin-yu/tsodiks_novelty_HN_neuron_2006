import numpy as np
import random


def generate_data(
    num_neurons: int, 
    num_morphs: int, 
    activation_prob: float = 0.5, 
    target_initial_correlation: float = 0.0, 
):
    x0 = (np.random.binomial(1, activation_prob, num_neurons) - 0.5) * 2 # observations are -1 and 1
    num_flips = int(round((1 - target_initial_correlation) * num_neurons / 2))
    x1 = x0.copy()
    flip_indices = np.random.choice(num_neurons, num_flips, replace=False)
    x1[flip_indices] *= -1
    
    X = np.zeros((num_morphs+2, num_neurons))
    X[0] = x0
    X[-1] = x1
    
    different_inds = np.where(x0 != x1)[0]
    
    random.shuffle(different_inds)
    morph_inds = np.split(different_inds, np.sort(np.random.choice(len(different_inds), num_morphs, replace=False)))[1:]
    
    for i in range(num_morphs):
        x_morph_temp = X[i].copy()
        x_morph_temp[morph_inds[i]] = -x_morph_temp[morph_inds[i]]
        X[i+1] = x_morph_temp
    
    return X


def hamming_distance(x1: np.ndarray, x2: np.ndarray):
    assert len(x1) == len(x2)
    return len(x1) - np.sum(np.equal(x1, x2))
