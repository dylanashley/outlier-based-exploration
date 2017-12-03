#!/usr/bin/env python -O
# -*- coding: ascii -*-

import argparse
import numpy as np
import signal
import sys

from grid_world import GridWorld

NUMBER_OF_ACTIONS = 4
NUMBER_OF_EPISODES = 10

DOMAIN_X_CARD = 5
DOMAIN_Y_CARD = 5
DOMAIN_X_START = 0
DOMAIN_Y_START = 0
DOMAIN_X_GOAL = 2
DOMAIN_Y_GOAL = 2


def main(args):
    global siginfo_message

    # build domain
    domain = GridWorld(DOMAIN_X_CARD, DOMAIN_Y_CARD, DOMAIN_X_START,
                       DOMAIN_Y_START, DOMAIN_X_GOAL, DOMAIN_Y_GOAL)

    # make table to track state visitations
    visitations = np.zeros((domain.x_card, domain.y_card), dtype=int)

    print('episode,performance')

    for episode in range(NUMBER_OF_EPISODES):

        # reset learner and domain
        state = domain.new_episode()

        # update state visitations
        visitations[domain.x, domain.y] += 1

        # run episode
        step = 0
        while True:
            step += 1

            # set message for siginfo
            siginfo_message = '[{0:3.2f}%] EPISODE: {1} of {2}, POS: ({3}, {4}), GOAL: ({5}, {6}), STEPS: {7}'.format(
                100 * episode / NUMBER_OF_EPISODES, episode + 1,
                NUMBER_OF_EPISODES, domain.x, domain.y, domain.x_goal,
                domain.y_goal, step)

            # take the action
            _, done = domain.step(np.random.randint(NUMBER_OF_ACTIONS - 1))

            # update state visitations
            visitations[domain.x, domain.y] += 1

            # if terminal then end episode
            if done:
                break

        print('{},{}'.format(
            episode + 1, (max(visitations.flat) - min(visitations.flat)) / sum(
                visitations.flat)))


def parse_args():
    parser = argparse.ArgumentParser()
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
            lambda signum, frame: sys.stderr.write(siginfo_message + '\n'))

    # parse args and run
    main(args)
