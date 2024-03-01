from typing import Optional, Union
import numpy as np


class HopfieldNetwork:
    def __init__(
        self, 
        num_neurons: int, 
        synchronous: bool, 
        num_iter: int,
        threshold: float = 0.0
    ):
        self.num_neurons = num_neurons
        self.W = np.zeros((num_neurons, num_neurons))
        self.scaled_W = np.zeros((num_neurons, num_neurons))
        
        self.synchronous = synchronous
        self.num_iter = num_iter
        self.threshold = threshold
        
        self.N = 0
        
    def train_weights(self, x: np.ndarray):
        N, D = x.shape
        assert D == self.num_neurons
        
        self.N += N
        
        rho = np.mean(x)
        
        for i in range(N):
            t = x[i] - rho
            self.W += np.outer(t, t)
        
        self.W[np.arange(self.num_neurons), np.arange(self.num_neurons)] = 0.0
        self.scaled_W = self.W / self.N
        
    def predict(self, x: np.ndarray):
        predicted = np.zeros_like(x)
        for i in range(len(x)):
            predicted[i] = self.forward(x[i])
        
        return predicted
    
    def forward(self, x: np.ndarray):
        if self.synchronous:
            e = self.energy(x)
            
            for i in range(self.num_iter):
                x = np.sign(self.scaled_W.dot(x) - self.threshold)
                e_new = self.energy(x)
                
                if e == e_new:
                    break
                e = e_new

        else:
            e = self.energy(x)
            
            for i in range(self.num_iter):
                for j in range(100):
                    ind = np.random.randint(0, self.num_neurons)
                    x[ind] = np.sign(np.dot(self.scaled_W[ind], x) - self.threshold)
                
                e_new = self.energy(x)
                
                if e == e_new:
                    break
                e = e_new
        
        return x
            
    def energy(self, x: np.ndarray):
        return -0.5 * np.dot(x, np.dot(self.scaled_W, x)) + np.sum(x * self.threshold)


class MishaHopfieldNetwork(HopfieldNetwork):
    def __init__(
        self, 
        num_neurons: int, 
        synchronous: bool, 
        num_iter: int,
        threshold: float = 0.0, 
        eta: float = 0.0, 
    ):
        super().__init__(num_neurons, synchronous, num_iter, threshold)
        
        self.running_rho = 0.0
        self.eta = eta
        
    def train_weights(self, x: np.ndarray, w: Optional[np.ndarray] = None):
        N, D = x.shape
        assert self.num_neurons == D
        
        if w is None:
            w = np.ones((N, ))
        
        rho = np.mean(x)
        
        self.running_rho = (self.N * self.running_rho + N * rho) / (self.N + N)
        
        self.N += N
        
        for i in range(N):
            t = x[i] - self.running_rho
            self.W += w[i] * np.outer(t, t)
        
        self.W[np.arange(self.num_neurons), np.arange(self.num_neurons)] = 0.0
        self.scaled_W = self.W / self.N
        
    def one_step_dynamics(self, x: np.ndarray):
        return np.sign(self.scaled_W.dot(x) - self.threshold)
        
    def forward(self, x: np.ndarray):
        x_old = x
        if self.synchronous:
            
            for i in range(self.num_iter):
                x = np.sign(self.scaled_W.dot(x) - self.threshold)
                
                if np.all(x == x_old):
                    break
                x_old = x

        else:
            
            for i in range(self.num_iter):
                for j in range(100):
                    ind = np.random.randint(0, self.num_neurons)
                    x[ind] = np.sign(np.dot(self.scaled_W[ind], x) - self.threshold)
                
                if np.all(x == x_old):
                    break
                x_old = x
        
        return x, i
    

class MishaHopfieldNetwork_v2(MishaHopfieldNetwork):
    def __init__(
        self, 
        num_neurons: int, 
        synchronous: bool, 
        num_iter: int,
        threshold: float = 0.0, 
        eta: float = 0.0, 
    ):
        super().__init__(num_neurons, synchronous, num_iter, threshold, eta)
        
    def train_weights(self, x: np.ndarray, w: Optional[np.ndarray] = None):
        N, D = x.shape
        assert self.num_neurons == D
        
        if w is None:
            w = np.ones((N, ))
            
        rho = np.mean(x)
        
        for i in range(N):
            t = x[i] - rho
            self.W += w[i] * np.outer(t, t)

        self.W[np.arange(self.num_neurons), np.arange(self.num_neurons)] = 0.0
        self.scaled_W = self.W / N
    

def hamming_distance(x1: np.ndarray, x2: np.ndarray):
    assert len(x1) == len(x2)
    return len(x1) - np.sum(np.equal(x1, x2))