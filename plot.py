#!/usr/bin/env python -O

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

k_distance_based_exploration = np.array(
    [[2.7370, 0.1483], [4.6262, 0.2010], [5.9213, 0.2144], [7.2166, 0.2621],
     [8.2862, 0.2802], [9.2388, 0.2893], [10.2220, 0.3081], [11.0836, 0.3335],
     [11.8401, 0.3591], [13.0235, 0.4145]])
lof_based_exploration = np.array(
    [[2.4168, 0.1035], [3.8896, 0.1278], [4.7700, 0.1276], [5.6858, 0.1348],
     [6.2699, 0.1332], [6.9734, 0.1330], [7.5895, 0.1353], [8.2384, 0.1442],
     [8.8775, 0.1550], [9.6382, 0.1920]])
policy_iteration = np.array(
    [[1.7125, 0.0526], [2.7717, 0.0659], [3.6604, 0.0728], [4.5657, 0.0757],
     [5.4610, 0.0753], [6.3102, 0.0833], [7.1571, 0.0866], [7.9695, 0.0900],
     [8.8090, 0.0977], [9.6188, 0.1024]])
random_exploration = np.array(
    [[1.9699, 0.0696], [3.2939, 0.0888], [4.4300, 0.1015], [5.5297, 0.1115],
     [6.7355, 0.1231], [7.8270, 0.1307], [8.8285, 0.1414], [9.8323, 0.1520],
     [10.7728, 0.1628], [11.7773, 0.1735]])

sns.set_style("whitegrid")
plt.gcf().set_size_inches(5, 5)
plt.gcf().set_dpi(900)


# MSE plot
episodes = np.arange(1, 11)
plt.errorbar(episodes, k_distance_based_exploration[:, 0], yerr=k_distance_based_exploration[:, 1], label='k-distance')
plt.errorbar(episodes, lof_based_exploration[:, 0], yerr=lof_based_exploration[:, 1], label='LOF')
plt.errorbar(episodes, policy_iteration[:, 0], yerr=policy_iteration[:, 1], label='Policy Iteration')
plt.errorbar(episodes, random_exploration[:, 0], yerr=random_exploration[:, 1], label='Random')
plt.legend(loc='best')
plt.xlabel('Episodes')
plt.ylabel('Standard Deviation')
plt.tight_layout()
plt.savefig('results.png')


# plt.plot(steps, all_rmse_mtd - all_rmse_td)
# ax.plot(steps, np.ones(NUMBER_OF_STEPS) * np.mean(all_rmse_mtd - all_rmse_td), 'r:')
# ax.set_xlabel('Timestep', fontsize=12)
# # ax.set_xlabel('Episode', fontsize=12)
# ax.set_ylabel('RMSE (negative value indicates improvement)', labelpad=10, fontsize=12)
# ax.set_title('Change in RMSE using MTD rather than TD', fontsize=12)

# # Beta plot
# ax = plt.subplot(ROWS, COLUMNS, 2)
# ax.plot(steps, np.mean(all_alpha, axis=0))
# ax.plot(steps, np.ones(NUMBER_OF_STEPS) * ALPHA, 'r:')
# ax.set_xlabel('Timestep', fontsize=12)
# # ax.set_xlabel('Episode', fontsize=12)
# ax.set_ylabel(r'$\alpha$', labelpad=10, fontsize=12)
# ax.set_title(r'MTD Mean Value With $\theta$ = {} versus TD'.format(THETA), fontsize=12)

# # Lambda plot
# ax = plt.subplot(ROWS, COLUMNS, 3)
# ax.plot(steps, all_lambda)
# ax.plot(steps, np.ones(NUMBER_OF_STEPS) * LAMBDA, 'r:')
# ax.set_xlabel('Timestep', fontsize=12)
# # ax.set_xlabel('Episode', fontsize=12)
# ax.set_yticks(np.linspace(0, 1, 5))
# ax.set_ylabel(r'$\lambda$', labelpad=10, fontsize=12)
# ax.set_title(r'MTD with $D$ = {} versus TD'.format(D), fontsize=12)


# plt.tight_layout()
# if FILENAME:
#     plt.savefig(FILENAME + '.png')
