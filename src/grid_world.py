# -*- coding: ascii -*-

import collections
import itertools
import numpy as np

from scipy.stats import truncnorm

Transition = collections.namedtuple('Transition',
                                    ['last_state', 'reward', 'gamma', 'state'])


class GridWorld:

    NORTH = 0
    SOUTH = 1
    EAST = 2
    WEST = 3
    STAY = 4

    def __init__(self,
                 x_card,
                 y_card,
                 x_start=None,
                 y_start=None,
                 x_goal=None,
                 y_goal=None,
                 gamma=1.0,
                 random_generator=np.random):
        assert (x_card > 1)
        assert (y_card > 1)
        if x_start is not None:
            assert (0 <= x_start < x_card)
        if y_start is not None:
            assert (0 <= y_start < y_card)
        if x_goal is not None:
            assert (0 <= x_goal < x_card)
        if y_goal is not None:
            assert (0 <= y_goal < y_card)
        assert (0 <= gamma <= 1)

        self.x_card = x_card
        self.y_card = y_card
        self.x_start = x_start
        self.y_start = y_start
        self.x_goal = x_goal
        self.y_goal = y_goal
        self.gamma = gamma
        self.random_generator = random_generator
        if self.x_goal is None:
            self.x_goal = self._x_normalized_to_coord(
                self.random_generator.rand())
        if self.y_goal is None:
            self.y_goal = self._y_normalized_to_coord(
                self.random_generator.rand())
        self._build_grid()
        self._build_true_value_table()
        self.new_episode()

    def _build_grid(self):
        self.grid = np.empty((self.x_card, self.y_card), dtype=object)
        for x, y in itertools.product(range(self.x_card), range(self.y_card)):
            # unutbu; https://stackoverflow.com/questions/18441779/how-to-specify-upper-and-lower-limits-when-using-numpy-random-normal; 2017-11-30
            x_mu, y_mu = self._coord_to_normalized(x, y)
            x_sigma = 1 / self.x_card
            y_sigma = 1 / self.y_card
            x_gen = truncnorm(
                (0 - x_mu) / x_sigma, (1 - x_mu) / x_sigma,
                loc=x_mu,
                scale=x_sigma)
            y_gen = truncnorm(
                (0 - y_mu) / y_sigma, (1 - y_mu) / y_sigma,
                loc=y_mu,
                scale=y_sigma)
            self.grid[x, y] = (x_gen, y_gen)

    def _build_true_value_table(self):
        self.true_value = np.zeros(self.grid.shape)
        for x, y in itertools.product(range(self.x_card), range(self.y_card)):
            distance = abs(x - self.x_goal) + abs(y - self.y_goal)
            if self.gamma == 1:
                self.true_value[x, y] = -distance
            else:
                self.true_value[x, y] = -self.gamma * (
                    self.gamma**distance - 1) / (self.gamma - 1)

    def _coord_to_normalized(self, x, y):
        return (self._x_coord_to_normalized(x), self._y_coord_to_normalized(y))

    def _x_coord_to_normalized(self, x):
        assert (0 <= x < self.x_card)
        return (x + 0.5) / self.x_card

    def _y_coord_to_normalized(self, y):
        assert (0 <= y < self.y_card)
        return (y + 0.5) / self.y_card

    def _normalized_to_coord(self, x, y):
        return (self._x_normalized_to_coord(x), self._y_normalized_to_coord(y))

    def _x_normalized_to_coord(self, x):
        assert (0 <= x <= 1)
        return min(int(x * (self.x_card - 1)), self.x_card - 1)

    def _y_normalized_to_coord(self, y):
        assert (0 <= y <= 1)
        return min(int(y * (self.y_card - 1)), self.y_card - 1)

    def _is_done(self):
        return (self.x == self.x_goal) and (self.y == self.y_goal)

    def _take_action(self, action):
        assert (action in {
            self.NORTH, self.SOUTH, self.EAST, self.WEST, self.STAY
        })
        if action == self.NORTH:
            self.y = max(0, self.y - 1)
        elif action == self.SOUTH:
            self.y = min(self.y_card - 1, self.y + 1)
        elif action == self.EAST:
            self.x = min(self.x_card - 1, self.x + 1)
        elif action == self.WEST:
            self.x = max(0, self.x - 1)
        else:
            pass
        self._make_samples()

    def _make_samples(self):
        self.sample_x = self.grid[self.x, self.y][0].rvs(
            random_state=self.random_generator)
        self.sample_y = self.grid[self.x, self.y][1].rvs(
            random_state=self.random_generator)

    def new_episode(self):
        if self.x_start is None:
            self.x = self._x_normalized_to_coord(self.random_generator.rand())
        else:
            self.x = self.x_start
        if self.y_start is None:
            self.y = self._y_normalized_to_coord(self.random_generator.rand())
        else:
            self.y = self.y_start
        if self._is_done():
            return self.new_episode()
        self._make_samples()
        return (self.sample_x, self.sample_y)

    def optimal_actions(self, x=None, y=None):
        if x is None:
            x = self.x
        if y is None:
            y = self.y
        rv = list()
        if x > self.x_goal:
            rv.append(self.WEST)
        if x < self.x_goal:
            rv.append(self.EAST)
        if y > self.y_goal:
            rv.append(self.NORTH)
        if y < self.y_goal:
            rv.append(self.SOUTH)
        return rv

    def step(self, action):
        last_state = (self.sample_x, self.sample_y)
        self._take_action(action)
        if self._is_done():
            reward = 0
            done = True
        else:
            reward = -1
            done = False
        gamma = self.gamma
        state = (self.sample_x, self.sample_y)
        return Transition(last_state, reward, gamma, state), done
