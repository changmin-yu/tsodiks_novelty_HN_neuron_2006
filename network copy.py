from typing import Optional, Union
import numpy as np
import torch.nn as nn
import random
from typing import Optional, Union

class TsodyksHopfieldNetwork(nn.Module):
    def __init__(self, N_neurons, num_iter, eta, threshold):
        super(TsodyksHopfieldNetwork, self).__init__()
        self.num_neurons = N_neurons
        self.W = np.zeros((N_neurons, N_neurons))
        self.scaled_W = np.zeros((N_neurons, N_neurons))
        self.num_iter = num_iter
        self.threshold = threshold
        self.N = 0
        self.running_rho = 0.0
        self.eta = eta
        
    def train_weights(self, x: np.ndarray, w: Optional[np.ndarray] = None):
        N, D = x.shape
        assert self.num_neurons == D
        
        if w is None:
            w = np.ones((N, ))
        rho = np.mean(x)
        
        for i in range(N): #N_pattern
            t = x[i] - rho
            self.W += w[i] * np.outer(t, t)

        np.fill_diagonal(self.W,0)

        self.N += N
        self.scaled_W = self.W / self.N
        
    def one_step_dynamics(self, x: np.ndarray):
        return np.sign(self.scaled_W.dot(x) - self.threshold)
        
    def converge(self, x: np.ndarray):
        x_old = x # input pattern
        for i in range(self.num_iter):
            x = np.sign(self.scaled_W.dot(x) - self.threshold)
            if np.all(x == x_old):
                break
            x_old = x
        return x, i
    

def hamming_distance(x1: np.ndarray, x2: np.ndarray):
    assert len(x1) == len(x2)
    return (len(x1) - np.sum(np.equal(x1, x2)))/len(x1)

def calculate_correlation(x, y):
    # Ensure x and y are numpy arrays
    x = np.array(x)
    y = np.array(y)
    
    # Step 1: Normalize x and y to have zero mean and unit variance
    x_normalized = (x - np.mean(x)) / np.std(x)
    y_normalized = (y - np.mean(y)) / np.std(y)
    
    # Step 2: Compute the dot product of the normalized vectors
    dot_product = np.dot(x_normalized, y_normalized)
    
    # Step 3: Since vectors are normalized, the dot product equals the correlation coefficient
    correlation_coefficient = dot_product / len(x)
    
    return correlation_coefficient


def create_target_source(num_neurons, t_s_corr): 
    x0 = np.array(np.random.binomial(1, 0.5, num_neurons) - 0.5) * 2
    prob_same = (1 + t_s_corr) / 2
    x1 = np.array([xi if np.random.random() < prob_same else -xi for xi in x0])
    return x0, x1, calculate_correlation(x0,x1)


def create_pattern(x0, x1, num_morphs) :
    permutation = np.random.permutation(len(x0))
    x0 = x0[permutation]
    x1 = x1[permutation]
    different_inds = np.where(x0 != x1)[0]
    random.shuffle(different_inds)
    
    morph_inds = np.split(different_inds, np.sort(np.random.choice(len(different_inds), num_morphs, replace=False)))[1:]

    x_morph = [x0]
    for i in range(num_morphs):
        x_morph_temp = x_morph[-1].copy()
        x_morph_temp[morph_inds[i]] = -x_morph_temp[morph_inds[i]]
        x_morph.append(x_morph_temp)

    x_morph = np.array(x_morph[1:])

    return x0, x1, x_morph

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