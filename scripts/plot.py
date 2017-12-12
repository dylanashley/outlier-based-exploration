#!/usr/bin/env python -O

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

NUMBER_OF_EPISODES = 25

# load k-distance based exploration results
k_distance_based_exploration = np.zeros((200, NUMBER_OF_EPISODES, 5,
                                         5)) * np.nan
for i in range(200):
    try:
        with open('results/k_distance_based_exploration/{}.npy'.format(i),
                  'rb') as infile:
            k_distance_based_exploration[i, ...] = np.load(infile)
    except FileNotFoundError:
        pass
k_distance_based_exploration_std = np.zeros((200, NUMBER_OF_EPISODES)) * np.nan
for i in range(200):
    for j in range(NUMBER_OF_EPISODES):
        k_distance_based_exploration_std[i, j] = np.std(
            k_distance_based_exploration[i, j, :, :])
k_distance_based_exploration_count = np.zeros((200,
                                               NUMBER_OF_EPISODES)) * np.nan
for i in range(200):
    for j in range(NUMBER_OF_EPISODES):
        k_distance_based_exploration_count[i, j] = np.sum(
            k_distance_based_exploration[i, j, :, :])
for i in reversed(range(1, NUMBER_OF_EPISODES)):
    k_distance_based_exploration_std[:,
                                     i] -= k_distance_based_exploration_std[:,
                                                                            i -
                                                                            1]
    k_distance_based_exploration_count[:,
                                       i] -= k_distance_based_exploration_count[:,
                                                                                i
                                                                                -
                                                                                1]

# load LOF based exploration results
lof_based_exploration = np.zeros((200, NUMBER_OF_EPISODES, 5, 5)) * np.nan
for i in range(200):
    try:
        with open('results/lof_based_exploration/{}.npy'.format(i),
                  'rb') as infile:
            lof_based_exploration[i, ...] = np.load(infile)
    except FileNotFoundError:
        pass
lof_based_exploration_std = np.zeros((200, NUMBER_OF_EPISODES)) * np.nan
for i in range(200):
    for j in range(NUMBER_OF_EPISODES):
        lof_based_exploration_std[i, j] = np.std(
            lof_based_exploration[i, j, :, :])
lof_based_exploration_count = np.zeros((200, NUMBER_OF_EPISODES)) * np.nan
for i in range(200):
    for j in range(NUMBER_OF_EPISODES):
        lof_based_exploration_count[i, j] = np.sum(
            lof_based_exploration[i, j, :, :])
for i in reversed(range(1, NUMBER_OF_EPISODES)):
    lof_based_exploration_std[:, i] -= lof_based_exploration_std[:, i - 1]
    lof_based_exploration_count[:, i] -= lof_based_exploration_count[:, i - 1]

# load policy iteration results
policy_iteration = np.zeros((1000, NUMBER_OF_EPISODES, 5, 5)) * np.nan
for i in range(1000):
    try:
        with open('results/policy_iteration/{}.npy'.format(i), 'rb') as infile:
            policy_iteration[i, ...] = np.load(infile)
    except FileNotFoundError:
        pass
policy_iteration_std = np.zeros((1000, NUMBER_OF_EPISODES)) * np.nan
for i in range(1000):
    for j in range(NUMBER_OF_EPISODES):
        policy_iteration_std[i, j] = np.std(policy_iteration[i, j, :, :])
policy_iteration_count = np.zeros((1000, NUMBER_OF_EPISODES)) * np.nan
for i in range(1000):
    for j in range(NUMBER_OF_EPISODES):
        policy_iteration_count[i, j] = np.sum(policy_iteration[i, j, :, :])
for i in reversed(range(1, NUMBER_OF_EPISODES)):
    policy_iteration_std[:, i] -= policy_iteration_std[:, i - 1]
    policy_iteration_count[:, i] -= policy_iteration_count[:, i - 1]

# load random exploration results
random_exploration = np.zeros((1000, NUMBER_OF_EPISODES, 5, 5)) * np.nan
for i in range(1000):
    try:
        with open('results/random_exploration/{}.npy'.format(i),
                  'rb') as infile:
            random_exploration[i, ...] = np.load(infile)
    except FileNotFoundError:
        pass
random_exploration_std = np.zeros((1000, NUMBER_OF_EPISODES)) * np.nan
for i in range(1000):
    for j in range(NUMBER_OF_EPISODES):
        random_exploration_std[i, j] = np.std(random_exploration[i, j, :, :])
random_exploration_count = np.zeros((1000, NUMBER_OF_EPISODES)) * np.nan
for i in range(1000):
    for j in range(NUMBER_OF_EPISODES):
        random_exploration_count[i, j] = np.sum(random_exploration[i, j, :, :])
for i in reversed(range(1, NUMBER_OF_EPISODES)):
    random_exploration_std[:, i] -= random_exploration_std[:, i - 1]
    random_exploration_count[:, i] -= random_exploration_count[:, i - 1]

# setup plot
sns.set_style("whitegrid")
plt.gcf().set_size_inches(5, 5)
plt.gcf().set_dpi(600)

# plot error over episodes
episodes = np.arange(NUMBER_OF_EPISODES) + 1
plt.errorbar(
    episodes,
    np.nanmean(
        k_distance_based_exploration_std / k_distance_based_exploration_count,
        axis=0),
    yerr=st.sem(
        k_distance_based_exploration_std / k_distance_based_exploration_count,
        axis=0,
        nan_policy='omit'),
    label='k-distance')
plt.errorbar(
    episodes,
    np.nanmean(
        lof_based_exploration_std / lof_based_exploration_count, axis=0),
    yerr=st.sem(
        lof_based_exploration_std / lof_based_exploration_count,
        axis=0,
        nan_policy='omit'),
    label='LOF')
plt.errorbar(
    episodes,
    np.nanmean(policy_iteration_std / policy_iteration_count, axis=0),
    yerr=st.sem(
        policy_iteration_std / policy_iteration_count,
        axis=0,
        nan_policy='omit'),
    label='Policy Iteration')
plt.errorbar(
    episodes,
    np.nanmean(random_exploration_std / random_exploration_count, axis=0),
    yerr=st.sem(
        random_exploration_std / random_exploration_count,
        axis=0,
        nan_policy='omit'),
    label='Random')
plt.xlabel('Episode')
plt.ylabel('Mean Error of a Timestep')
plt.legend(loc='best')

plt.tight_layout()
plt.savefig('results.png')
