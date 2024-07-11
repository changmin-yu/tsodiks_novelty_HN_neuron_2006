import numpy as np
import random


def generate_data(
    num_neurons: int, 
    num_morphs: int, 
    activation_prob: float = 0.5, 
):
    x0 = (np.random.binomial(1, activation_prob, num_neurons) - 0.5) * 2 # observations are -1 and 1
    x1 = (np.random.binomial(1, activation_prob, num_neurons) - 0.5) * 2
    
    X = np.zeros((num_morphs+2, num_neurons))
    X[0] = x0
    X[-1] = x1
    
    identical_inds = np.where(x0 == x1)[0]
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
