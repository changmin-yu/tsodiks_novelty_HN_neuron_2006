import numpy as np
import random
from network import TsodyksHopfieldNetwork 
from utils import graph, graph_Wills_Leutgeb, hamming_distance, generate_data
import matplotlib.pyplot as plt
from scipy.spatial import distance
from typing import Optional, Union

def test_FullMorph(num_neurons, num_morphs, pattern, num_repetition):
    '''
    30-morph sequence with increasing correlation 

    Input: 
    num_neurons: number of neurons 
    num_morphs: number of morphing patterns 
    pattern: pattern sequence with the first element source pattern; last element target pattern; in-between morphing pattern 
    num_repetition: how many times the system is given the pattern 

    Return 
    correlations_gradual: correlation for gradual increasing order  
    correlations_random: correlation for mixed order 
    num_morphs: number of morphs 

    '''
    x0 = pattern[0]
    x1 = pattern[-1]
    x_morph = pattern[1:-1]
    num_seed = 10 
    eta = 0.5

    x_init = np.array([x0, x1])
    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    
    correlations_gradual = np.zeros((num_seed, num_morphs))
    correlations_random = np.zeros((num_seed, num_morphs))
    
    ########################## gradual morphing ##########################
    for i in range(num_seed): 
        network = TsodyksHopfieldNetwork(num_neurons, num_iter=10, eta=eta, threshold=0.0)
        network.train_weights(x=x_init, w= w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))

        for j in range(num_morphs):
            for k in range (num_repetition):
                # if j == len(x_morph)-1: #double check if the last 
                #     print(np.sum(np.dot(x0,x_morph[j]))/len(x0))
                x_one_step = network.one_step_dynamics(x_morph[j])
                hamming_distance_temp = hamming_distance(x_one_step, x_morph[j])/hamming_distance_init
                w_morph[j] += eta * hamming_distance_temp
                network.train_weights(x_morph[j][None], w=w_morph[[j]])
            
            # correlation
            converged_state = network.converge(x_morph[j])
            correlations_gradual[i, j] = np.sum(converged_state[0] * x_init[0]) / num_neurons


    ########################### random morphing ##########################
    for i in range(num_seed):
        network = TsodyksHopfieldNetwork(num_neurons, num_iter=10, eta=eta, threshold=0.0)
        network.train_weights(x=x_init, w=w_init)     
        w_morph = np.zeros((num_morphs, ))

        random_perms = np.random.permutation(np.arange(num_morphs))
        invert_permute = {i: j for i, j in enumerate(random_perms)}
        for j in range(num_morphs):
            for k in range (num_repetition):
                x_one_step = network.one_step_dynamics(x_morph[random_perms[j]])
                hamming_distance_temp = hamming_distance(x_one_step, x_morph[random_perms[j]]) / hamming_distance_init
                w_morph[random_perms[j]] += eta * hamming_distance_temp
                network.train_weights(x_morph[random_perms[j]][None], w=w_morph[[random_perms[j]]])

            converged_state = network.converge(x_morph[random_perms[j]])
            correlations_random[i, invert_permute[j]] = np.sum(converged_state[0] * x_init[0]) / num_neurons 
        
    
    return correlations_gradual, correlations_random, num_morphs

def test_novelty_HN_wills(X, num_neurons, num_morphs, num_repetitions, memory=False): # Random Morphing 
    '''
    Reproduces Wills paper (random morphing) with 6 moprhing patterns

    Input: 
    X: pattern sequence with the first element source pattern; last element target pattern; in-between morphing pattern
    num_neurons: number of neurons 
    num_morphs: number of morphing patterns 
    num_repetitions:  how many times the system is given the pattern
    memory: if true: append the source pattern at the end of the morphing sequence 

    Return: 
    correlations: morphing sequences' correlations with the source pattern  
    morphs_inds: morphing indices (used for graphing later)
    '''
    num_seeds = 10
    if memory == True:
        morphs_inds = np.array([20, 12, 24, 6, 18, 9, 1], dtype=int) - 1
    else: 
        morphs_inds = np.array([20, 12, 24, 6, 18, 9], dtype=int) - 1

    x0 = X[0]
    x1 = X[-1]
    x_morph = X[1:-1]
    
    x_init = np.array([x0, x1])
    # init_correlations = np.sum(x0 * x1) / num_neurons

    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    
    eta = 0.5
    
    correlations = np.zeros((num_seeds, num_morphs))

    for i in range(num_seeds):
        network = TsodyksHopfieldNetwork(num_neurons, num_iter=10, eta=eta, threshold=0.0)
        network.train_weights(x_init, w=w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))
        
        for j in range(len(morphs_inds)):
            for k in range(num_repetitions): # different sessions: how many times it give the system to learn  
                x_one_step = network.one_step_dynamics(x_morph[morphs_inds[j]])
                hamming_distance_temp = hamming_distance(x_one_step, x_morph[morphs_inds[j]]) / hamming_distance_init
                w_morph[morphs_inds[j]] += eta * hamming_distance_temp
                network.train_weights(x_morph[morphs_inds[j]][None], w=w_morph[[morphs_inds[j]]])
                
            converged_state, _ = network.converge(x_morph[morphs_inds[j]])
            correlations[i, morphs_inds[j]] = np.sum(converged_state * x_init[0]) / num_neurons
    
    return correlations, morphs_inds

def test_novelty_HN_leutgeb(X, num_neurons, num_morphs, num_repetitions, memory=False): # Gradual Morphing 
    '''
    Reproduces Leutgeb paper (gradual morphing) with 6 moprhing patterns

    Input: 
    X: pattern sequence with the first element source pattern; last element target pattern; in-between morphing pattern
    num_neurons: number of neurons 
    num_morphs: number of morphing patterns 
    num_repetitions:  how many times the system is given the pattern
    memory: if true: append the source pattern at the end of the morphing sequence 

    Return: 
    correlations: morphing sequences' correlations with the source pattern  
    morphs_inds: morphing indices (used for graphing later)
    '''
    
    num_seeds = 10
    if memory == True:
        morphs_inds = np.array([6, 9, 12, 18, 20, 24, 1], dtype=int) - 1
    else: 
        morphs_inds = np.array([6, 9, 12, 18, 20, 24], dtype=int) - 1

    x0 = X[0]
    x1 = X[-1]
    x_morph = X[1:-1]
    
    x_init = np.array([x0, x1])
    # init_correlations = np.sum(x0 * x1) / num_neurons

    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    
    eta = 0.5
    
    correlations = np.zeros((num_seeds, num_morphs))

    for i in range(num_seeds):
        network = TsodyksHopfieldNetwork(num_neurons, num_iter=10, eta=eta, threshold=0.0)
        network.train_weights(x_init, w=w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))
        
        for j in range(len(morphs_inds)):
            for k in range(num_repetitions): #different sessions: how many times it give the system to learn 
                x_one_step = network.one_step_dynamics(x_morph[morphs_inds[j]])
                hamming_distance_temp = hamming_distance(x_one_step, x_morph[morphs_inds[j]]) / hamming_distance_init
                w_morph[morphs_inds[j]] += eta * hamming_distance_temp
                network.train_weights(x_morph[morphs_inds[j]][None], w=w_morph[[morphs_inds[j]]])
                
            converged_state, _ = network.converge(x_morph[morphs_inds[j]])
            correlations[i, morphs_inds[j]] = np.sum(converged_state * x_init[0]) / num_neurons
    return correlations, morphs_inds

def test_memory_interference_exp(num_neurons, num_morphs, pattern, num_repetition): # last pattern is the first pattern again 
    '''
    function to produce memory interference for Wills and Leutgeb experiment 

    '''

    x0 = np.expand_dims(pattern[0],0)
    x1 = np.expand_dims(pattern[-1],0)
    x_morph = np.concatenate((pattern[1:-1], x0), axis=0)
    pattern = np.concatenate((x0, x_morph, x1), axis=0)
    num_morphs = int(len(x_morph))
    correlations_random, morph_idx = test_novelty_HN_wills(pattern, num_neurons, num_morphs, num_repetition, memory = True)
    correlations_gradual, morph_idx = test_novelty_HN_leutgeb(pattern, num_neurons, num_morphs, num_repetition, memory = True)
        
    return correlations_gradual, correlations_random, morph_idx

def test_memory_interference_morph30(num_neurons, num_morphs, pattern, num_repetition):
    '''
    produce memory interference for 30 morphing sequence 

    '''
    x0 = np.expand_dims(pattern[0],0)
    x1 = np.expand_dims(pattern[-1],0)
    x_morph = np.concatenate((pattern[1:-1], x0), axis=0) # append the source pattern at the end of x_morph
    pattern = np.concatenate((x0, x_morph, x1), axis=0)
    num_morphs = int(len(x_morph))
    correlations_gradual, correlations_random, num_morph = test_FullMorph(num_neurons, num_morphs, pattern, num_repetition)
    return correlations_gradual, correlations_random, num_morph

'''
Master function to produce all results 

Input: 
method: options: 'test_FullMorph'// 'test_memory_interference_morph30' // 'test_novelty_HN_wills'//  'test_novelty_HN_leutgeb'// 'test_memory_interference_exp'
num_neurons: number of neurons 
num_morphs: number of morphing 
t_s_corrr_list: target-source correlation list (a list from 0.0 to 1.0)
num_repetitions: number of repetitions

Return: 
'test_FullMorph' and 'test_memory_interference_morph30' ------ None 
'test_novelty_HN_wills' and 'test_novelty_HN_leutgeb' ------ correlation for each morphing pattern + morph_idx 
'test_memory_interference_exp' ------ gradual_increasing correlation; mixed_increasing correlation; morph_idx
'''
def main (method, num_neurons, num_morphs, t_s_corr_list, num_repetitions: Optional[int]=None):     
    if method.__name__ == 'test_FullMorph' or method.__name__ == 'test_memory_interference_morph30':
            average_corr_gradual = []
            average_corr_random = []
            for t_s_corr in t_s_corr_list: # Different Correlation 
                print('correlation between the source and the target is: ', t_s_corr)
                for _ in range (1): # Same Correlation, Different Pattern
                    pattern = generate_data(num_neurons, num_morphs, target_initial_correlation=t_s_corr)
                    correlations_gradual, correlations_random, morph_idx = method(num_neurons, num_morphs, pattern, num_repetitions) # [num_seed, num_morph]
                    average_corr_gradual.append(correlations_gradual)
                    average_corr_random.append(correlations_random)
                graph(method=method, average_corr_gradual = np.stack(average_corr_gradual,0).mean(0), average_corr_random= np.stack(average_corr_random,0).mean(0), num_morphs= morph_idx, t_s_corr = t_s_corr)

    elif method.__name__ == 'test_novelty_HN_wills' or method.__name__ == 'test_novelty_HN_leutgeb':
        pattern_corr = []
        for t_s_corr in t_s_corr_list: # Different Correlation
            average_corr = [] # if Same Correlation, Different Pattern is needed
            print('correlation between the source and the target is: ', t_s_corr)
            for _ in range (1): # Same Correlation, Different Pattern
                X = generate_data(num_neurons, num_morphs, target_initial_correlation=t_s_corr)
                correlation, morph_idx = method(X, num_neurons, num_morphs, num_repetitions)
                average_corr.append(correlation) #[num_seed, num_morph]
            pattern_corr.append(np.stack(average_corr,0).mean(0))
        return pattern_corr, morph_idx
    
    elif method.__name__ == 'test_memory_interference_exp':
        average_corr_gradual = []
        average_corr_random = []
        for t_s_corr in t_s_corr_list: # Different Correlation 
            print('correlation between the source and the target is: ', t_s_corr)
            for _ in range (1): # Same Correlation, Different Pattern
                pattern = generate_data(num_neurons, num_morphs, target_initial_correlation=t_s_corr)
                correlations_gradual, correlations_random, morph_idx = method(num_neurons, num_morphs, pattern, num_repetitions) # [num_seed, num_morph]
                average_corr_gradual.append(correlations_gradual)
                average_corr_random.append(correlations_random)
        return  np.stack(average_corr_gradual,0), np.stack(average_corr_random,0), morph_idx


