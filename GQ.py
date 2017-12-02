# -*- coding: ascii -*-

import numpy as np

from tools import scale


class GQ:
    def __init__(self, n, random_init_range=None):
        self.e = np.zeros(n)
        self.theta = np.zeros(n)
        self.w = np.zeros(n)
        if random_init_range is not None:
            assert (len(random_init_range) == 2)
            assert (max(random_init_range) - min(random_init_range) > 0)
            for i in range(n):
                self.w[i] += scale(np.random.rand(), 0, 1,
                                   min(random_init_range),
                                   max(random_init_range))

    def predict(self, phi):
        return np.dot(self.w, phi)

    def reset(self, phi):
        np.copyto(self.e, phi)

    def update(self,
               reward,
               gamma,
               phi,
               phi_bar,
               alpha,
               lambda_,
               rho=1,
               eta=1,
               I=1):
        delta = reward + gamma * np.dot(self.theta, phi_bar) - np.dot(
            self.theta, phi)
        self.e *= rho
        self.e += I * phi
        self.theta += alpha * (
            delta * self.e - gamma *
            (1 - lambda_) * np.dot(self.w, self.e) * phi_bar)
        self.w += alpha * eta * (delta * self.e - np.dot(self.w, phi) * phi)
        self.e *= gamma * lambda_
