import numpy as np
import random
from network import TsodyksHopfieldNetwork, create_pattern, hamming_distance, create_target_source, generate_data
from utils import graph, graph_Wills_Leutgeb
import matplotlib.pyplot as plt
from scipy.spatial import distance
from typing import Optional, Union

def test_FullMorph(num_neurons, num_morphs, pattern, num_repetition):
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

def test_memory_inference_exp(num_neurons, num_morphs, pattern, num_repetition): # last pattern is the first pattern again 

    x0 = np.expand_dims(pattern[0],0)
    x1 = np.expand_dims(pattern[-1],0)
    x_morph = np.concatenate((pattern[1:-1], x0), axis=0)
    pattern = np.concatenate((x0, x_morph, x1), axis=0)
    num_morphs = int(len(x_morph))
    correlations_random, morph_idx = test_novelty_HN_wills(pattern, num_neurons, num_morphs, num_repetition, memory = True)
    correlations_gradual, morph_idx = test_novelty_HN_leutgeb(pattern, num_neurons, num_morphs, num_repetition, memory = True)
        
    return correlations_gradual, correlations_random, morph_idx

def test_memory_inference_morph30(num_neurons, num_morphs, pattern, num_repetition):
    x0 = np.expand_dims(pattern[0],0)
    x1 = np.expand_dims(pattern[-1],0)
    x_morph = np.concatenate((pattern[1:-1], x0), axis=0)
    pattern = np.concatenate((x0, x_morph, x1), axis=0)
    num_morphs = int(len(x_morph))
    correlations_gradual, correlations_random, num_morph = test_FullMorph(num_neurons, num_morphs, pattern, num_repetition)
    return correlations_gradual, correlations_random, num_morph

# def test_ABAB_inference (num_neurons, num_morphs, pattern, num_repetition): #ABA pattern memory inference 
    x0 = np.expand_dims(pattern[0],0)
    x1 = np.expand_dims(pattern[-1],0)
    x2 = pattern[1:-1]
    x_morph = np.concatenate((x2, x0),axis=0)
    num_morphs = len(x_morph)
    num_seeds = 10 
    eta = 0.5

    x_init = np.array([np.squeeze(x0,0), np.squeeze(x1,0)])
    w_init = np.array([1., 1.])
    hamming_distance_init = hamming_distance(x0, x1)
    correlations_A = np.zeros((num_seeds, num_morphs))
    correlations_B = np.zeros((num_seeds, num_morphs))

    for i in range(num_seeds): 
        network = TsodyksHopfieldNetwork(num_neurons, num_iter=10, eta=eta, threshold=0.0)
        network.train_weights(x=x_init, w= w_init) # pre-training of two contexts
        w_morph = np.zeros((num_morphs, ))

        for j in range(num_morphs):
            for k in range (num_repetition): 
                x_one_step = network.one_step_dynamics(x_morph[j])
                hamming_distance_temp = hamming_distance(x_one_step, x_morph[j])/hamming_distance_init
                w_morph[j] += eta * hamming_distance_temp
                network.train_weights(x_morph[j][None], w=w_morph[[j]])
            
            # correlation
            converged_state = network.converge(x_morph[j])
            correlations_A[i, j] = np.sum(converged_state[0] * x_init[0]) / num_neurons
            correlations_B[i, j] = np.sum(converged_state[0] * x_init[1]) / num_neurons
    
    return correlations_A, correlations_B, num_morphs

def main (method, num_neurons, num_morphs, t_s_corr_list, num_repetitions: Optional[int]=None):
    
    if method.__name__ == 'test_FullMorph' or method.__name__ == 'test_memory_inference_morph30':
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
            average_corr = []
            print('correlation between the source and the target is: ', t_s_corr)
            for _ in range (1): # Same Correlation, Different Pattern
                X = generate_data(num_neurons, num_morphs, target_initial_correlation=t_s_corr)
                correlation, morph_idx = method(X, num_neurons, num_morphs, num_repetitions)
                average_corr.append(correlation) #[num_seed, num_morph]
            pattern_corr.append(np.stack(average_corr,0).mean(0))
        return pattern_corr, morph_idx
    
    elif method.__name__ == 'test_memory_inference_exp':
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


################# 30-morph ##############
# num_neurons = 400
# num_morphs = 30
# t_s_corr_list = np.linspace(0,0.7,10)
# num_repetition = 1
# main(test_FullMorph, num_neurons, num_morphs, t_s_corr_list, num_repetitions=num_repetition)

################# 30-morph-memory-interference ##############
num_neurons = 400
num_morphs = 30
t_s_corr_list = np.linspace(0,0.7,10)
num_repetition = 1
corr_grd, corr_rnd, morph_idx = main(method = test_memory_inference_morph30, num_neurons=num_neurons, num_morphs=num_morphs, 
                                     t_s_corr_list=t_s_corr_list, num_repetitions=num_repetition)


################# 4-morph-experiment-memory-interference ##############
# num_neurons = 120
# num_morphs = 30
# t_s_corr_list = np.linspace(0,0.50,10)
# num_repetition = 1
# corr_grd, corr_rnd, morph_idx = main(method=test_memory_inference_exp, num_neurons=num_neurons, num_morphs=num_morphs, 
#                                      t_s_corr_list=t_s_corr_list, num_repetitions=num_repetition)
# for i in range(len(t_s_corr_list)):
#     graph_Wills_Leutgeb(corr_rnd[i], corr_grd[i], morph_idx, t_s_corr_list[i], True)

################# 4-morph-experiment ##############
# num_neurons = 120
# num_morphs = 30
# t_s_corr_list = np.linspace(0,0.50,10)
# num_repetition = 1
# corr_rnd, morph_idx = main(method=test_novelty_HN_wills, num_neurons=num_neurons, num_morphs=num_morphs, 
#                                   t_s_corr_list=t_s_corr_list, num_repetitions=num_repetition) 
   
# corr_grd, morph_idx = main(method=test_novelty_HN_leutgeb, num_neurons=num_neurons, num_morphs=num_morphs, 
#                                   t_s_corr_list=t_s_corr_list, num_repetitions=num_repetition)
# for i in range(len(t_s_corr_list)):
#     graph_Wills_Leutgeb(corr_rnd[i], corr_grd[i], morph_idx, t_s_corr_list[i], False)
