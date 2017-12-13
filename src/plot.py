#!/usr/bin/env python -O

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

NUMBER_OF_EPISODES = 25
NUMBER_OF_SEEDS = 1000

folders = [{
    'name': 'random_exploration',
    'label': 'Random'
}, {
    'name': 'k_distance_based_exploration',
    'label': 'State Based k-distance'
}, {
    'name': 'lof_based_exploration',
    'label': 'State Based LOF'
}, {
    'name': 'transition_k_distance_based_exploration',
    'label': 'Transition Based k-distance'
}, {
    'name': 'transition_lof_based_exploration',
    'label': 'Transition Based LOF'
}, {
    'name': 'policy_iteration',
    'label': 'Policy Iteration'
}]

sns.set_style("whitegrid")
plt.gcf().set_size_inches(5, 5)
plt.gcf().set_dpi(600)
episodes = np.arange(NUMBER_OF_EPISODES) + 1

for folder in folders:
    data = np.ones((NUMBER_OF_SEEDS, NUMBER_OF_EPISODES, 5, 5)) * np.nan
    std = np.ones((NUMBER_OF_SEEDS, NUMBER_OF_EPISODES)) * np.nan
    count = np.zeros((NUMBER_OF_SEEDS, NUMBER_OF_EPISODES)) * np.nan
    for i in range(NUMBER_OF_SEEDS):
        try:
            with open('results/{}/{}.npy'.format(folder['name'], i),
                      'rb') as infile:
                data[i, ...] = np.load(infile)
        except FileNotFoundError:
            continue
        for j in range(NUMBER_OF_EPISODES):
            std[i, j] = np.std(data[i, j, :, :])
            count[i, j] = np.sum(data[i, j, :, :])
    for i in reversed(range(1, NUMBER_OF_EPISODES)):
        std[:, i] -= std[:, i - 1]
        count[:, i] -= count[:, i - 1]

    plt.errorbar(
        episodes,
        np.nanmean(std / count, axis=0),
        yerr=st.sem(std / count, axis=0, nan_policy='omit'),
        label=folder['label'])

plt.xlabel('Episode')
plt.ylabel('Mean Error of a Timestep')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('results.png')
