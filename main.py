import numpy as np
import random
from hopfield_network import HopfieldNetwork, MishaHopfieldNetwork, hamming_distance


def test_classical_HN():
    x = np.array([
        [-1, -1, -1, -1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1], 
        [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1, -1,], 
    ])
    
    network = HopfieldNetwork(16, synchronous=False, num_iter=20)
    
    network.train_weights(x)
    
    test_x = np.array([
        [-1, -1, -1, 1, 1, 1, 1, 1, -1, -1, -1, -1, 1, 1, 1, 1], 
        [-1, -1, -1, -1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -1, -1, -1,], 
    ])
    
    predict_x = network.predict(test_x)
    
    return network


def test_Tsodyks():
    num_neurons = 80
    num_morphs = 30
    
    num_iterations = 10

    x0 = (np.random.binomial(1, 0.5, num_neurons) - 0.5) * 2
    x1 = (np.random.binomial(1, 0.5, num_neurons) - 0.5) * 2
    x_init = np.array([x0, x1])
    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)

    different_inds = np.where(x0 != x1)[0]

    random.shuffle(different_inds)
    morph_inds = np.split(different_inds, np.sort(np.random.choice(len(different_inds), num_morphs, replace=False)))[1:]

    x_morph = [x0]
    for i in range(num_morphs):
        x_morph_temp = x_morph[-1].copy()
        x_morph_temp[morph_inds[i]] = -x_morph_temp[morph_inds[i]]
        x_morph.append(x_morph_temp)

    x_morph = np.array(x_morph[1:])
    w_morph = np.zeros((num_morphs, ))
    
    eta = 0.5
    
    network = MishaHopfieldNetwork(num_neurons, synchronous=True, num_iter=20)
    
    network.train_weights(x_init, w=w_init) # pre-training of two contexts
    
    correlations_gradual = np.zeros((num_iterations, num_morphs))
    correlations_random = np.zeros((num_iterations, num_morphs))
    
    # gradual morphing
    for i in range(num_iterations):
        for j in range(num_morphs):
            x_one_step = network.one_step_dynamics(x_morph[j])
            hamming_distance_temp = hamming_distance(x_one_step, x_morph[j]) / hamming_distance_init
            w_morph[j] += eta * hamming_distance_temp
            network.train_weights(x_morph[j][None], w=w_morph[[j]])
            
        for j in range(num_morphs):
            converged_state = network.forward(x_morph[j])
            correlations_gradual[i, j] = np.sum(converged_state * x_init[0]) / num_neurons
            
    # random morphing
    for i in range(num_iterations):
        random_perms = np.random.permutation(np.arange(num_morphs))
        for j in range(num_morphs):
            x_one_step = network.one_step_dynamics(x_morph[random_perms[j]])
            hamming_distance_temp = np.array([hamming_distance(x_one_step, x_morph[random_perms[j]])]) / hamming_distance_init
            w_morph[random_perms[j]] += eta * hamming_distance_temp
            network.train_weights(x_morph[random_perms[j]][None], w=w_morph[[random_perms[j]]])
            
        for j in range(num_morphs):
            converged_state = network.forward(x_morph[j])
            correlations_random[i, j] = np.sum(converged_state * x_init[0]) / num_neurons
    
    return correlations_gradual, correlations_random


if __name__=="__main__":
    # network = test_classical_HN()
    correlations_gradual, correlations_random = test_Tsodyks()