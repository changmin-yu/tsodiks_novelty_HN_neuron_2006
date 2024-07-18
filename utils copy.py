import numpy as np 
import matplotlib.pyplot as plt 
from typing import Optional, Union
import matplotlib.colors as mcolors
import random
from typing import Optional, Union

def hamming_distance(x1: np.ndarray, x2: np.ndarray):
    assert len(x1) == len(x2)
    return (len(x1) - np.sum(np.equal(x1, x2)))/len(x1)

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


####################### graphing utils #######################
css4_colors = mcolors.CSS4_COLORS
color1 = '#507b9c' #grad
color2 = '#feb308' #rnd

def graph(method, 
          num_morphs: Optional[int]=None, t_s_corr: Optional[list]=None, average_corr_gradual:Optional[list]=None, average_corr_random:Optional[list]=None,
          correlations: Optional[list]=None, morphs_inds: Optional[list]=None):
    
    if method.__name__ == 'test_FullMorph' or method.__name__ == 'test_memory_interference_morph30':
        if method.__name__ == 'test_FullMorph': 
            path = '/Users/weilinran/Desktop/CCSNN/fig/Morph30/' + 'Morph30_{}.jpg'.format(np.round(t_s_corr,3))
        else: 
            path = '/Users/weilinran/Desktop/CCSNN/fig/memory_inference/Morph30/' +'memory30_{}.jpg'.format(np.round(t_s_corr,3))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.plot(average_corr_gradual.mean(0),'-.r', c=color1, label='Gradual Morphing', linewidth=3)
        ax.fill_between(
            np.arange(1, num_morphs+1), 
            average_corr_gradual.mean(0) - average_corr_gradual.std(0), 
            average_corr_gradual.mean(0) + average_corr_gradual.std(0), 
            color=color1, 
            alpha=0.3, 
        )
        ax.plot(average_corr_random.mean(0), color=color2, label='Random Morphing', linewidth=3)
        ax.fill_between(
            np.arange(1, num_morphs+1), 
            average_corr_random.mean(0)- average_corr_random.std(0), 
            average_corr_random.mean(0)+ average_corr_random.std(0), 
            color=color2, 
            alpha=0.3, 
        )

        ax.set_ylim(-0.2,1.2)
        ax.set_ylabel('Correlation with The First Pattern')
        ax.set_xlabel('Morphing Index')
        fig.suptitle('Morphing Patterns with Correlation: {}'.format(np.round(t_s_corr,3)))
        fig.legend()
        plt.show()
        fig.savefig(path)

    elif method.__name__ == 'test_novelty_HN_wills' or method.__name__ == 'test_novelty_HN_leutgeb':
        fig, ax = plt.subplots(1, 1, figsize=(5, 5))
        correlations = np.stack(correlations, 0).mean(0)
        correlations = correlations[[5, 8, 11, 17, 19, 23]]
        if method.__name__== 'test_novelty_HN_wills':
            name = 'Wills'
            label = 'Random Morphing'
        else: 
            name = 'Leutgeb'
            label = 'Gradual Morphing'

        ax.plot(np.arange(1, len(morphs_inds)+1), correlations, color=color1, label=label, linewidth=3)
        ax.set_ylim(-0.2,1.2)

        ax.legend()
        ax.set_title('{} Experiments (Initial correlation_{})'.format(name, np.round(t_s_corr,3)))
        ax.set_xlabel("Morphing index")
        ax.set_ylabel("Correlation with The First Pattern")
        
        fig.savefig('/Users/weilinran/Desktop/CCSNN/fig/'+ name + '/novelty_HN_corr{}.jpg'.format(np.round(t_s_corr,3)))
        plt.show()

def graph_Wills_Leutgeb(corr_wills, corr_leut, morph_idx, t_s_corr, memory=False):
    if memory == False:
        path = '/Users/weilinran/Desktop/CCSNN/fig/Will_Leutgeb/' + 'Will_Leut_memory_{}.jpg'.format(np.round(t_s_corr,3))
    else: 
        path = '/Users/weilinran/Desktop/CCSNN/fig/memory_inference/Wills_Leutgeb/' + 'Will_Leut_memory_{}.jpg'.format(np.round(t_s_corr,3))

    fig, ax = plt.subplots(1, 1, figsize=(8, 5))

    # Wills experiment
    if memory == True:
        corr_wills = corr_wills[:,[5, 8, 11, 17, 19, 23, 0]]
    else: 
        corr_wills = corr_wills[:,[5, 8, 11, 17, 19, 23]]

    # Leutgeb experiment 
    if memory == True:
        corr_leut = corr_leut[:,[5, 8, 11, 17, 19, 23, 0]]
    else: 
        corr_leut = corr_leut[:,[5, 8, 11, 17, 19, 23]]

    ax.plot(np.arange(1, len(morph_idx)+1), corr_leut.mean(0),'-.r', color=color1, label='Gradual Morphing', linewidth=3)
    ax.plot(np.arange(1, len(morph_idx)+1), corr_wills.mean(0),color=color2, label='Random Morphing', linewidth=3)
    
    ax.fill_between(
            np.arange(1, len(morph_idx)+1), 
            corr_wills.mean(0)-corr_wills.std(0), 
            corr_wills.mean(0)+corr_wills.std(0), 
            color=color1, 
            alpha=0.3, )
    
    ax.fill_between(
            np.arange(1, len(morph_idx)+1), 
            corr_leut.mean(0)-corr_leut.std(0), 
            corr_leut.mean(0)+corr_leut.std(0), 
            color=color2, 
            alpha=0.3, )

    ax.set_ylim(-0.2,1.2)
    ax.set_ylabel('Correlation with The First Pattern')
    ax.set_xlabel('Morphing Index')
    fig.suptitle('Morphing Patterns with Correlation: {}'.format(np.round(t_s_corr,3)))
    fig.legend()
    plt.show()
    fig.savefig(path)