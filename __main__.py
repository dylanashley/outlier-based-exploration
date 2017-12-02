#!/usr/bin/env python
# -*- coding: ascii -*-

import numpy as np
import signal

from GQ import GQ
from grid_world import GridWorld
from lof_grid import LOFGrid
from tile_coder import TileCoder

NUMBER_OF_ACTIONS = 4
NUMBER_OF_EPISODES = 1000
SEED = 595750214  # generated using random.org


def main():
    global siginfo_message

    # build domain
    x_card = y_card = 5
    x_start = y_start = 1
    x_goal = y_goal = 3
    domain = GridWorld(
        x_card, y_card, x_start, y_start, x_goal, y_goal, gamma=0.9)

    # build tile coder
    tiling_card = 10
    tiling_count = 10
    tile_coder = TileCoder(tiling_card, tiling_count)

    # build vectors to store values
    indices = np.zeros((tile_coder.n, NUMBER_OF_ACTIONS), dtype=int)
    phi = np.zeros(tile_coder.tile_count * NUMBER_OF_ACTIONS, dtype=int)
    phi_bar = np.copy(phi)
    action_values = np.zeros(NUMBER_OF_ACTIONS)

    # build grid
    grid_card = (10, 10)
    grid_k = 20
    init_points = list()
    for _ in range(grid_k + 1):
        init_points.append(domain.new_episode())
    lof_grid = LOFGrid(grid_card, grid_k, init_points)

    # build learner
    driver = GQ(
        tile_coder.tile_count * NUMBER_OF_ACTIONS, random_init_range=(0, 0.01))

    # build structure to track lof
    lof_mean, lof_n = 0, 0

    for episode in range(NUMBER_OF_EPISODES):

        # reset learner and domain
        state = domain.new_episode()

        # tile state
        indices.fill(0)
        tile_coder.discretize(*state, indices[:, 0])
        for action in range(1, NUMBER_OF_ACTIONS):
            indices[:,
                    action] += indices[:, action - 1] + tile_coder.tile_count

        # get action values
        for action in range(NUMBER_OF_ACTIONS):
            action_values[action] = np.sum(driver.w[indices[:, action]])

        # reset learner
        phi.fill(0)
        phi[indices[:, action]] += 1
        driver.reset(phi)

        # add new state to grid
        lof_grid.insert(state)

        # run episode
        done = False
        while not done:

            # set message for siginfo
            siginfo_message = '[{0:3.2f}%] EPISODE: {1} of {2}, POS: ({3}, {4}), GOAL: ({5}, {6})'.format(
                100 * episode / NUMBER_OF_EPISODES, episode + 1,
                NUMBER_OF_EPISODES, domain.x, domain.y, domain.x_goal,
                domain.y_goal)

            # take the best action
            action = np.argmax(action_values)
            # BEGIN TEST CODE ##########################################
            optimal_probs = np.zeros(NUMBER_OF_ACTIONS)
            if domain.x > domain.x_goal:
                optimal_probs[domain.WEST] += 1
            if domain.x < domain.x_goal:
                optimal_probs[domain.EAST] += 1
            if domain.y > domain.y_goal:
                optimal_probs[domain.NORTH] += 1
            if domain.y < domain.y_goal:
                optimal_probs[domain.SOUTH] += 1
            optimal_probs /= np.sum(optimal_probs)
            action = np.random.randint(NUMBER_OF_ACTIONS)
            rho = optimal_probs[action] / 0.25
            # END TEST CODE ############################################
            (last_state, reward, gamma, state), done = domain.step(action)
            phi.fill(0)
            phi[indices[:, action]] += 1

            # tile state
            indices.fill(0)
            tile_coder.discretize(*state, indices[:, 0])
            for action in range(1, NUMBER_OF_ACTIONS):
                indices[:, action] += indices[:, action -
                                              1] + tile_coder.tile_count

            # get action values
            action_values.fill(0)
            for action in range(NUMBER_OF_ACTIONS):
                action_values[action] = np.sum(driver.w[indices[:, action]])

            # add new state to grid
            # lof = lof_grid.insert(state)
            # lof_n += 1
            # lof_mean += (lof - lof_mean) / lof_n

            # update estimates for the
            phi_bar.fill(0)
            phi_bar[indices[:, np.argmax(action_values)]] += 1
            # driver.update(np.exp(lof - 1), 0.9, phi, phi_bar, 0.1, 0.9)

            # BEGIN TEST CODE ##########################################
            driver.update(reward, 1.0, phi, phi_bar, 0.01, 0.9, rho=rho)
            # END TEST CODE ############################################

    for y in range(domain.y_card):
        for x in range(domain.x_card):
            state = domain._coord_to_normalized(x, y)
            indices.fill(0)
            tile_coder.discretize(*state, indices[:, 0])
            for action in range(1, NUMBER_OF_ACTIONS):
                indices[:, action] += indices[:, action -
                                              1] + tile_coder.tile_count
            action_values.fill(0)
            for action in range(NUMBER_OF_ACTIONS):
                action_values[action] = np.sum(driver.w[indices[:, action]])
            optimal_probs = np.zeros(NUMBER_OF_ACTIONS)
            if x > domain.x_goal:
                optimal_probs[domain.WEST] += 1
            if x < domain.x_goal:
                optimal_probs[domain.EAST] += 1
            if y > domain.y_goal:
                optimal_probs[domain.NORTH] += 1
            if y < domain.y_goal:
                optimal_probs[domain.SOUTH] += 1
            print(bool(optimal_probs[np.argmax(action_values)]), end=' ')
        print()

    print(lof_mean)


if __name__ == '__main__':
    # setup numpy
    np.random.seed(SEED)
    np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')

    # setup siginfo response system
    global siginfo_message
    siginfo_message = None
    if hasattr(signal, 'SIGINFO'):
        signal.signal(signal.SIGINFO,
                      lambda signum, frame: print(siginfo_message))

    main()
