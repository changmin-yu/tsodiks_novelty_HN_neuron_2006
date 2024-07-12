import numpy as np
import random

import matplotlib.pyplot as plt

from networks import HopfieldNetwork, NoveltyHebbianLearningHopfieldNetwork
from utils import (
    generate_data, 
    hamming_distance, 
)


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


def test_novelty_HN(target_initial_correlation: float=0.0):
    num_seeds = 10
    
    num_neurons = 80
    num_morphs = 30
    
    X = generate_data(num_neurons, num_morphs, target_initial_correlation=target_initial_correlation)
    x0 = X[0]
    x1 = X[-1]
    x_morph = X[1:-1]
    
    x_init = np.array([x0, x1])
    init_correlations = np.sum(x0 * x1) / num_neurons

    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    
    eta = 0.5
    
    correlations_gradual = np.zeros((num_seeds, num_morphs))
    correlations_random = np.zeros((num_seeds, num_morphs))
    
    # gradual morphing
    for i in range(num_seeds):
        
        network = NoveltyHebbianLearningHopfieldNetwork(num_neurons, synchronous=True, num_iter=20)
        network.train_weights(x_init, w=w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))
        
        for j in range(num_morphs):
            x_one_step = network.one_step_dynamics(x_morph[j])
            hamming_distance_temp = hamming_distance(x_one_step, x_morph[j]) / hamming_distance_init
            w_morph[j] += eta * hamming_distance_temp
            network.train_weights(x_morph[j][None], w=w_morph[[j]])
            
            converged_state, _ = network.forward(x_morph[j])
            correlations_gradual[i, j] = np.sum(converged_state * x_init[0]) / num_neurons
            
    # random morphing
    for i in range(num_seeds):
        
        network = NoveltyHebbianLearningHopfieldNetwork(num_neurons, synchronous=True, num_iter=20)
        network.train_weights(x_init, w=w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))
        
        random_perms = np.random.permutation(np.arange(num_morphs))
        for j in range(num_morphs):
            x_one_step = network.one_step_dynamics(x_morph[random_perms[j]])
            hamming_distance_temp = hamming_distance(x_one_step, x_morph[random_perms[j]]) / hamming_distance_init
            w_morph[random_perms[j]] += eta * hamming_distance_temp
            network.train_weights(x_morph[random_perms[j]][None], w=w_morph[[random_perms[j]]])
            
            converged_state, _ = network.forward(x_morph[random_perms[j]])
            correlations_random[i, random_perms[j]] = np.sum(converged_state * x_init[0]) / num_neurons
            
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    ax.plot(np.arange(1, num_morphs+1)/num_morphs, correlations_gradual.mean(0), "x--", color="blue", label="gradual morphing")
    ax.fill_between(
        np.arange(1, num_morphs+1)/num_morphs, 
        correlations_gradual.mean(0)-correlations_gradual.std(0), 
        correlations_gradual.mean(0)+correlations_gradual.std(0), 
        color="blue", 
        alpha=0.3, 
    )
    
    ax.plot(np.arange(1, num_morphs+1)/num_morphs, correlations_random.mean(0), "x--", color="green", label="random morphing")
    ax.fill_between(
        np.arange(1, num_morphs+1)/num_morphs, 
        correlations_random.mean(0)-correlations_random.std(0), 
        correlations_random.mean(0)+correlations_random.std(0), 
        color="green", 
        alpha=0.3, 
    )
    
    ax.legend()
    ax.set_title(f"Initial correlation: {init_correlations:.2f}")
    ax.set_xlabel("Morphing index")
    ax.set_ylabel("Correlation")
    
    fig.savefig(f"figures/novelty_HN_corr{init_correlations:.2f}.png")
    
    return init_correlations, correlations_gradual, correlations_random


def test_novelty_HN_wills(target_initial_correlation: float=0.0):
    num_seeds = 10
    
    num_neurons = 80
    num_morphs = 30
    
    morphs_inds = np.array([12, 24, 6, 18], dtype=int) - 1
    num_repetitions = 1
    
    X = generate_data(num_neurons, num_morphs, target_initial_correlation=target_initial_correlation)
    x0 = X[0]
    x1 = X[-1]
    x_morph = X[1:-1]
    
    x_init = np.array([x0, x1])
    init_correlations = np.sum(x0 * x1) / num_neurons

    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    
    eta = 0.5
    
    correlations = np.zeros((num_seeds, num_morphs))
    
    for i in range(num_seeds):
        network = NoveltyHebbianLearningHopfieldNetwork(num_neurons, synchronous=True, num_iter=20)
        network.train_weights(x_init, w=w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))
        
        for j in range(len(morphs_inds)):
            for k in range(num_repetitions):
                x_one_step = network.one_step_dynamics(x_morph[morphs_inds[j]])
                hamming_distance_temp = hamming_distance(x_one_step, x_morph[morphs_inds[j]]) / hamming_distance_init
                w_morph[morphs_inds[j]] += eta * hamming_distance_temp
                network.train_weights(x_morph[morphs_inds[j]][None], w=w_morph[[morphs_inds[j]]])
                
            converged_state, _ = network.forward(x_morph[morphs_inds[j]])
            correlations[i, morphs_inds[j]] = np.sum(converged_state * x_init[0]) / num_neurons
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    correlations = correlations[:, [5, 11, 17, 23]]
    
    ax.plot(np.arange(1, len(morphs_inds)+1)/(len(morphs_inds)), correlations.mean(0), "x--", color="green", label="random morphing")
    ax.fill_between(
        np.arange(1, len(morphs_inds)+1)/(len(morphs_inds)), 
        correlations.mean(0)-correlations.std(0), 
        correlations.mean(0)+correlations.std(0), 
        color="green", 
        alpha=0.3, 
    )
    
    ax.legend()
    ax.set_title(f"Wills Experiments (Initial correlation: {init_correlations:.2f})")
    ax.set_xlabel("Morphing index")
    ax.set_ylabel("Correlation")
    
    fig.savefig(f"figures/novelty_HN_corr{init_correlations:.2f}_wills.png")


def test_novelty_HN_leutgeb(target_initial_correlation: float=0.0):
    num_seeds = 10
    
    num_neurons = 80
    num_morphs = 30
    
    morphs_inds = np.array([6, 12, 18, 24], dtype=int) - 1
    num_repetitions = 2
    
    X = generate_data(num_neurons, num_morphs, target_initial_correlation=target_initial_correlation)
    x0 = X[0]
    x1 = X[-1]
    x_morph = X[1:-1]
    
    x_init = np.array([x0, x1])
    init_correlations = np.sum(x0 * x1) / num_neurons

    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    
    eta = 0.5
    
    correlations = np.zeros((num_seeds, num_morphs))
    
    for i in range(num_seeds):
        network = NoveltyHebbianLearningHopfieldNetwork(num_neurons, synchronous=True, num_iter=20)
        network.train_weights(x_init, w=w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))
        
        for j in range(len(morphs_inds)):
            for k in range(num_repetitions):
                x_one_step = network.one_step_dynamics(x_morph[morphs_inds[j]])
                hamming_distance_temp = hamming_distance(x_one_step, x_morph[morphs_inds[j]]) / hamming_distance_init
                w_morph[morphs_inds[j]] += eta * hamming_distance_temp
                network.train_weights(x_morph[morphs_inds[j]][None], w=w_morph[[morphs_inds[j]]])
                
            converged_state, _ = network.forward(x_morph[morphs_inds[j]])
            correlations[i, morphs_inds[j]] = np.sum(converged_state * x_init[0]) / num_neurons
    
    fig, ax = plt.subplots(1, 1, figsize=(5, 5))
    
    correlations = correlations[:, [5, 11, 17, 23]]
    
    ax.plot(np.arange(1, len(morphs_inds)+1)/(len(morphs_inds)), correlations.mean(0), "x--", color="green", label="random morphing")
    ax.fill_between(
        np.arange(1, len(morphs_inds)+1)/(len(morphs_inds)), 
        correlations.mean(0)-correlations.std(0), 
        correlations.mean(0)+correlations.std(0), 
        color="green", 
        alpha=0.3, 
    )
    
    ax.legend()
    ax.set_title(f"Leutgeb Experiments (Initial correlation: {init_correlations:.2f})")
    ax.set_xlabel("Morphing index")
    ax.set_ylabel("Correlation")
    
    fig.savefig(f"figures/novelty_HN_corr{init_correlations:.2f}_leutgeb.png")

if __name__=="__main__":
    # network = test_classical_HN()
    # for corr in np.arange(0, 0.25, 0.05):
    #     init_correlations, correlations_gradual, correlations_random = test_novelty_HN(corr)
    # test_novelty_HN_wills()
    test_novelty_HN_leutgeb()