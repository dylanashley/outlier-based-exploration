#!/usr/bin/env python -O
# -*- coding: ascii -*-

import argparse
import numpy as np
import signal
import sys

from grid_world import GridWorld
from lof_calculator import *
from tile_coder import TileCoder

LAMBDA = 0.9
MU = 0.1
NUMBER_OF_ACTIONS = 4
NUMBER_OF_EPISODES = 25
PI = 0.0
GAMMA = 0.9

NUMBER_OF_TILINGS = 1
TILING_CARDINALITY = 10

DOMAIN_X_CARD = 5
DOMAIN_Y_CARD = 5
DOMAIN_X_START = 0
DOMAIN_Y_START = 0
DOMAIN_X_GOAL = 2
DOMAIN_Y_GOAL = 2

ALPHA = 0.5 / NUMBER_OF_TILINGS
EPSILON_ACTION_SELECTION_PROBABILITY = PI / NUMBER_OF_ACTIONS
GREEDY_ACTION_SELECTION_PROBABILITY = (
    1 - PI) + EPSILON_ACTION_SELECTION_PROBABILITY


def main(args):
    global siginfo_message

    # build domain
    domain = GridWorld(DOMAIN_X_CARD, DOMAIN_Y_CARD, DOMAIN_X_START,
                       DOMAIN_Y_START, DOMAIN_X_GOAL, DOMAIN_Y_GOAL)

    # build tile coder
    tile_coder = TileCoder(TILING_CARDINALITY, NUMBER_OF_TILINGS)

    # build lof calculator
    grid_card = (10, 10)
    k = 20
    init_points = list()
    for _ in range(k + 1):
        (last_state, _, _, state), done = domain.step(
            np.random.randint(NUMBER_OF_ACTIONS))
        init_points.append(last_state)
        if done:
            domain.new_episode()
    lof_calculator = GridLOFCalculator(grid_card, k, init_points)

    # make arrays to save reallocation later
    e = np.zeros(tile_coder.tile_count * NUMBER_OF_ACTIONS)
    F = np.zeros((NUMBER_OF_TILINGS, NUMBER_OF_ACTIONS), dtype=int)
    Q = np.zeros(NUMBER_OF_ACTIONS)
    theta = np.random.rand(tile_coder.tile_count * NUMBER_OF_ACTIONS) * 0.01

    # make function to populate f and q
    def populate(state, F, Q):
        tile_coder.discretize(*state, F[:, 0])
        for action in range(1, NUMBER_OF_ACTIONS):
            F[:, action] = F[:, action - 1] + tile_coder.tile_count
        Q.fill(0)
        for i in range(NUMBER_OF_TILINGS):
            for j in range(NUMBER_OF_ACTIONS):
                Q[j] += theta[F[i, j]]

    # make list to track state visitations
    visitations = list()

    for episode in range(NUMBER_OF_EPISODES):

        # reset learner and domain
        state = domain.new_episode()
        e.fill(0)

        # update state visitations
        visitations.append((domain.x, domain.y))

        # add new state to grid
        lof_calculator.insert(state)

        # run episode
        step = 0
        while True:
            step += 1

            # set message for siginfo
            siginfo_message = '[{0:3.2f}%] EPISODE: {1} of {2}, POS: ({3}, {4}), GOAL: ({5}, {6}), STEPS: {7}'.format(
                100 * episode / NUMBER_OF_EPISODES, episode + 1,
                NUMBER_OF_EPISODES, domain.x, domain.y, domain.x_goal,
                domain.y_goal, step)

            # populate f and q
            populate(state, F, Q)

            # choose an action
            greedy_action = np.argmax(Q)
            if (np.random.rand() > MU):
                action = greedy_action
            else:
                action = np.random.randint(NUMBER_OF_ACTIONS)
            if (action != greedy_action):
                e.fill(0)

            # take the action
            (last_state, _, _, state), done = domain.step(action)

            # update state visitations
            visitations.append((domain.x, domain.y))

            # add new state to grid
            k_distance = lof_calculator.insert(state, rv='k-distance')

            # get delta
            delta = k_distance - Q[action]

            # update traces for visited features
            for i in F[:, action]:
                e[i] = 1

            # if terminal then update theta and end episode
            if done:
                theta += ALPHA * delta * e
                break

            # populate f and q
            populate(state, F, Q)

            # update everything that still needs updating
            for i in range(NUMBER_OF_ACTIONS):
                if i == action:
                    delta += GAMMA * GREEDY_ACTION_SELECTION_PROBABILITY * Q[i]
                else:
                    delta += GAMMA * EPSILON_ACTION_SELECTION_PROBABILITY * Q[
                        i]
            theta += ALPHA * delta * e
            e *= GAMMA * LAMBDA

    with open(args['filename'], 'wb') as outfile:
        np.save(outfile, np.array(visitations))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('filename')
    parser.add_argument('-s', '--seed', type=int)
    return vars(parser.parse_args())


if __name__ == '__main__':
    # get command line arguments
    args = parse_args()

    # setup numpy
    if args['seed'] is not None:
        np.random.seed(args['seed'])
    np.seterr(divide='raise', over='raise', under='ignore', invalid='raise')

    # setup siginfo response system
    global siginfo_message
    siginfo_message = None
    if hasattr(signal, 'SIGINFO'):
        signal.signal(
            signal.SIGINFO,
            lambda signum, frame: sys.stderr.write('{}\n'.format(siginfo_message))
        )

    # parse args and run
    main(args)
