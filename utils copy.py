import numpy as np 
import matplotlib.pyplot as plt 
from typing import Optional, Union
import matplotlib.colors as mcolors


css4_colors = mcolors.CSS4_COLORS

# color1 = '#5a86ad' #grad
# color2 = '#825f87' #rnd
color1 = '#507b9c' #grad
color2 = '#feb308' #rnd
 
def graph(method, 
          num_morphs: Optional[int]=None, t_s_corr: Optional[list]=None, average_corr_gradual:Optional[list]=None, average_corr_random:Optional[list]=None,
          correlations: Optional[list]=None, morphs_inds: Optional[list]=None):
    
    if method.__name__ == 'test_FullMorph' or method.__name__ == 'test_memory_inference_morph30':
        if method.__name__ == 'test_FullMorph': 
            path = '/Users/weilinran/Desktop/CCSNN/fig/Morph30/' + 'Morph30_{}.jpg'.format(np.round(t_s_corr,3))
        else: 
            path = '/Users/weilinran/Desktop/CCSNN/fig/memory_inference/Morph30/' +'memory30_{}.jpg'.format(np.round(t_s_corr,3))

        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
        # plt.gcf().set_facecolor('#53596F')
        # plt.gca().set_facecolor('#53596F')
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
        # ax.spines['top'].set_color('#53596F')
        # ax.spines['right'].set_color('#53596F')
        # ax.spines['bottom'].set_color('white')
        # ax.spines['left'].set_color('white')

        ax.set_ylim(-0.2,1.2)
        ax.set_ylabel('Correlation with The First Pattern')
        ax.set_xlabel('Morphing Index')
        fig.suptitle('Morphing Patterns with Correlation: {}'.format(np.round(t_s_corr,3)))
        # ax.tick_params(axis='x', colors='white')
        # ax.tick_params(axis='y', colors='white')
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
        # ax.fill_between(
        #     np.arange(1, len(morphs_inds)+1)/(len(morphs_inds)), 
        #     correlations.mean(0)-correlations.std(0), 
        #     correlations.mean(0)+correlations.std(0), 
        #     color="green", 
        #     alpha=0.3, 
        # )      

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
    # plt.gcf().set_facecolor('#53596F')
    # plt.gca().set_facecolor('#53596F')

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

    # ax.spines['top'].set_color('#53596F')
    # ax.spines['right'].set_color('#53596F')
    # ax.spines['bottom'].set_color('white')
    # ax.spines['left'].set_color('white')

    ax.set_ylim(-0.2,1.2)
    ax.set_ylabel('Correlation with The First Pattern')
    ax.set_xlabel('Morphing Index')
    fig.suptitle('Morphing Patterns with Correlation: {}'.format(np.round(t_s_corr,3)))
    # ax.tick_params(axis='x', colors='white')
    # ax.tick_params(axis='y', colors='white')
    fig.legend()
    plt.show()
    fig.savefig(path)