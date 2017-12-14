#!/usr/bin/env python -O
# -*- coding: ascii -*-

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
import seaborn as sns

EXPLORATION_NUMBER_OF_EPISODES = 25
GOAL_STATE = (2, 2)
LEARNING_NUMBER_OF_EPISODES = 10000
MAX_NUMBER_OF_SEEDS = 1000

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

# create figures
figures = dict()
for name in [
        'cumulative_episodic_error', 'cumulative_normalized_episodic_error',
        'episodic_length', 'cumulative_percentage_optimal_actions'
]:
    fig = plt.figure(figsize=(5, 5), dpi=600)
    ax = fig.add_subplot(111)
    figures[name] = {'fig': fig, 'ax': ax}
sns.set_style('whitegrid')

# build std performance figures
for folder in folders:
    count = np.ones((MAX_NUMBER_OF_SEEDS,
                     EXPLORATION_NUMBER_OF_EPISODES)) * np.nan
    cumulative_count = np.ones((MAX_NUMBER_OF_SEEDS,
                                EXPLORATION_NUMBER_OF_EPISODES)) * np.nan
    cumulative_std = np.ones((MAX_NUMBER_OF_SEEDS,
                              EXPLORATION_NUMBER_OF_EPISODES)) * np.nan
    data = np.ones((MAX_NUMBER_OF_SEEDS, EXPLORATION_NUMBER_OF_EPISODES, 5,
                    5)) * np.nan
    for i in range(MAX_NUMBER_OF_SEEDS):
        try:
            with open('results/{}/{}.npy'.format(folder['name'], i),
                      'rb') as infile:
                visitations = np.load(infile)
        except FileNotFoundError as e:
            continue
        assert (len(visitations) > 0)

        # compress visitations to episodes
        data[i, :, :, :].fill(0)
        episode = 0
        for x, y in visitations:
            data[i, episode, x, y] += 1
            if (x, y) == GOAL_STATE:
                episode += 1
        assert (episode == EXPLORATION_NUMBER_OF_EPISODES)

        # calculate metrics over data
        for j in range(EXPLORATION_NUMBER_OF_EPISODES):
            count[i, j] = np.sum(data[i, j, :, :])
        for j in range(1, EXPLORATION_NUMBER_OF_EPISODES):
            data[i, j, :, :] += data[i, j - 1, :, :]
        for j in range(EXPLORATION_NUMBER_OF_EPISODES):
            cumulative_count[i, j] = np.sum(data[i, j, :, :])
            cumulative_std[i, j] = np.std(data[i, j, :, :])

    # plot figures for folder
    figures['cumulative_episodic_error']['ax'].errorbar(
        np.arange(EXPLORATION_NUMBER_OF_EPISODES) + 1,
        np.nanmean(cumulative_std, axis=0),
        yerr=st.sem(cumulative_std, axis=0, nan_policy='omit'),
        label=folder['label'])
    figures['episodic_length']['ax'].errorbar(
        np.arange(EXPLORATION_NUMBER_OF_EPISODES) + 1,
        np.nanmean(count, axis=0),
        yerr=st.sem(count, axis=0, nan_policy='omit'),
        label=folder['label'])
    figures['cumulative_normalized_episodic_error']['ax'].errorbar(
        np.arange(EXPLORATION_NUMBER_OF_EPISODES) + 1,
        np.nanmean(cumulative_std / cumulative_count, axis=0),
        yerr=st.sem(
            cumulative_std / cumulative_count, axis=0, nan_policy='omit'),
        label=folder['label'])

# build learning performance figure
data = np.ones((MAX_NUMBER_OF_SEEDS, LEARNING_NUMBER_OF_EPISODES, 2)) * np.nan
for i in range(MAX_NUMBER_OF_SEEDS):
    try:
        with open(
                'results/policy_iteration_for_optimal_policy/{}.npy'.format(i),
                'rb') as infile:
            data[i, :, :] = np.load(infile)
    except FileNotFoundError:
        continue
    for j in range(1, LEARNING_NUMBER_OF_EPISODES):
        data[i, j, :] += data[i, j - 1, :]

# plot learning performance figure
figures['cumulative_percentage_optimal_actions']['ax'].errorbar(
    np.arange(LEARNING_NUMBER_OF_EPISODES) + 1,
    np.nanmean(data[:, :, 0] / data[:, :, 1], axis=0),
    yerr=st.sem(data[:, :, 0] / data[:, :, 1], axis=0, nan_policy='omit'))

# set axes labels for figures and display legend
figures['cumulative_episodic_error']['ax'].set_ylabel('Cumulative Error')
figures['cumulative_normalized_episodic_error']['ax'].set_ylabel(
    'Cumulative Normalized Error')
figures['cumulative_percentage_optimal_actions']['ax'].set_ylabel(
    'Cumulative Percentage of Timesteps Optimal Action Was Taken')
figures['episodic_length']['ax'].set_ylabel('Timesteps in Episode')
for name in [
        'cumulative_episodic_error', 'cumulative_normalized_episodic_error',
        'cumulative_percentage_optimal_actions', 'episodic_length'
]:
    figures[name]['ax'].set_xlabel('Episode')
for name in [
        'cumulative_episodic_error', 'cumulative_normalized_episodic_error',
        'episodic_length'
]:
    figures[name]['ax'].legend(loc='best')

# save figures
for name, v in figures.items():
    v['fig'].tight_layout()
    v['fig'].savefig('results/{}.png'.format(name))
