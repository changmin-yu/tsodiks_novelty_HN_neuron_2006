from typing import Optional, Union
import numpy as np
import torch.nn as nn


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