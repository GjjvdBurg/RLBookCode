#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Bandit Algorithms

This script follows Chapter 2 of Sutton and Barto (2nd) and simply reproduces 
figures 2.2 to 2.5.

Author: Gertjan van den Burg
License: MIT
Copyright: (c) 2020, The Alan Turing Institute

"""

import abc
import math
import matplotlib.pyplot as plt
import numpy as np
import random
import tqdm

from matplotlib import ticker
from scipy.special import logsumexp


class TestBed:
    """ k-Armed Test Bed """

    def __init__(self, k=10, baseline=0):
        self.k = k
        self.baseline = baseline
        self._opt_action = None

    @property
    def opt_action(self):
        if self._opt_action is None:
            raise ValueError("Not initialised properly!")
        return self._opt_action

    def step(self, action):
        mean = self._qstar[action]
        return random.gauss(mean, 1)

    def reset(self):
        self._qstar = []
        for _ in range(self.k):
            self._qstar.append(random.gauss(self.baseline, 1))
        self._opt_action = argmax(lambda a: self._qstar[a], range(self.k))
        return self


class Bandit(metaclass=abc.ABCMeta):
    def __init__(self, k=10, initial_value=0, stepsize="avg"):
        self.k = k
        self.initial_value = initial_value
        self.stepsize = stepsize

    def reset(self):
        # Reset the state of the bandit.
        self.Q = {a: self.initial_value for a in range(self.k)}
        self.N = {a: 0 for a in range(self.k)}
        if self.stepsize == "avg":
            self.alpha = lambda a: 1 if self.N[a] == 0 else 1.0 / self.N[a]
        else:
            self.alpha = lambda a: self.stepsize

    @abc.abstractmethod
    def get_action(self):
        """ Choose an action to take """

    def record(self, action, reward):
        """ Record the reward of the action taken """
        # Follows algorithm on page 32
        A, R = action, reward
        self.N[A] += 1
        self.Q[A] += self.alpha(A) * (R - self.Q[A])


class EpsilonGreedy(Bandit):
    def __init__(self, k=10, epsilon=0.1, initial_value=0, stepsize="avg"):
        super().__init__(k=k, initial_value=initial_value, stepsize=stepsize)
        self.epsilon = epsilon

    def get_action(self):
        if random.random() <= self.epsilon:
            return random.randint(0, self.k - 1)
        return argmax(lambda a: self.Q[a], range(self.k))

    def label(self):
        return (
            r"$\varepsilon$-greedy ($\varepsilon = %g$, $Q_1 = %g$, $\alpha = %s$)"
            % (self.epsilon, self.initial_value, self.stepsize)
        )


class UpperConfidence(Bandit):
    def __init__(self, k=10, c=2.0):
        super().__init__(k=k)
        self.c = c

    def reset(self):
        super().reset()
        self.t = 0

    def get_action(self):
        self.t += 1
        func = lambda a: self.Q[a] + self.c * math.sqrt(
            math.log(self.t) / self.N[a]
        )
        for a in range(self.k):
            # first pick all actions at least once
            if self.N[a] == 0:
                return a
        return argmax(func, range(self.k))

    def label(self):
        return r"UCB ($c = %g$)" % self.c


class GradientBandit(Bandit):
    def __init__(self, k=10, stepsize="avg", use_baseline=True):
        super().__init__(k=k, stepsize=stepsize)
        self.use_baseline = use_baseline

    def reset(self):
        super().reset()
        self.H = {a: 0 for a in range(self.k)}
        self.probs, self.Rtbar, self.t = None, 0, 0

    def get_action(self):
        self.t += 1
        lse = logsumexp(list(self.H.values()))
        self.probs = [math.exp(self.H[a] - lse) for a in range(self.k)]
        a = random.choices(list(range(self.k)), weights=self.probs, k=1)
        return a[0]

    def record(self, action, reward):
        At, Rt = action, reward
        for a in range(self.k):
            self.H[a] += (
                self.alpha(a) * (Rt - self.Rtbar) * ((At == a) - self.probs[a])
            )
        # Note that the choice of baseline is somewhat arbitrary, but the
        # average reward works well in practice. See discussion on page 40 of
        # Sutton & Barto.
        if self.use_baseline:
            self.Rtbar += 1 / self.t * (Rt - self.Rtbar)

    def label(self):
        bsln = "with" if self.use_baseline else "without"
        return r"Gradient ($\alpha = %s$, %s baseline)" % (self.stepsize, bsln)


def argmax(func, args):
    """Simple argmax function """
    m, inc = -float("inf"), None
    for a in args:
        if (v := func(a)) > m:
            m, inc = v, a
    return inc


def plot_common(axis, data, bandits):
    axis.plot(data.T)
    axis.legend([b.label() for b in bandits])
    axis.set_xlabel("Steps")


def make_reward_plot(axis, avg_rewards, bandits):
    plot_common(axis, avg_rewards, bandits)
    axis.set_ylabel("Average\nreward", rotation="horizontal", ha="center")


def make_optact_plot(axis, avg_optact, bandits):
    plot_common(axis, avg_optact, bandits)
    axis.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axis.set_ylim(0, 1)
    axis.yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
    axis.set_ylabel("%\nOptimal\naction", rotation="horizontal", ha="center")


def run_experiment(env, bandits, repeats, steps):
    B = len(bandits)
    rewards = np.zeros((B, repeats, steps))
    optact = np.zeros((B, repeats, steps))
    for r in tqdm.trange(repeats):
        # reset the bandits and the environment
        [bandit.reset() for bandit in bandits]
        env.reset()
        for i in range(steps):
            for b in range(B):
                bandit = bandits[b]
                action = bandit.get_action()
                reward = env.step(action)
                bandit.record(action, reward)
                rewards[b, r, i] = reward
                optact[b, r, i] = action == env.opt_action

    avg_rewards = rewards.mean(axis=1)
    avg_optact = optact.mean(axis=1)
    return avg_rewards, avg_optact


def figure_2_2(k=10, repeats=2000, steps=1000, epsilons=None):
    env = TestBed(k=k)
    epsilons = epsilons or [0.1, 0.01, 0]
    bandits = [EpsilonGreedy(k=k, epsilon=e) for e in epsilons]
    avg_rewards, avg_optact = run_experiment(env, bandits, repeats, steps)

    fig, axes = plt.subplots(2, 1)
    make_reward_plot(axes[0], avg_rewards, bandits)
    make_optact_plot(axes[1], avg_optact, bandits)
    plt.show()


def figure_2_3(k=10, repeats=2000, steps=1000):
    env = TestBed(k=k)
    bandits = [
        EpsilonGreedy(k=k, epsilon=0.1, initial_value=0, stepsize=0.1),
        EpsilonGreedy(k=k, epsilon=0, initial_value=5, stepsize=0.1),
    ]
    _, avg_optact = run_experiment(env, bandits, repeats, steps)

    fig, axis = plt.subplots(1, 1)
    make_optact_plot(axis, avg_optact, bandits)
    plt.show()


def figure_2_4(k=10, repeats=2000, steps=1000, c=2):
    env = TestBed(k=k)
    bandits = [EpsilonGreedy(k=k, epsilon=0.1), UpperConfidence(k=k, c=c)]
    avg_rewards, _ = run_experiment(env, bandits, repeats, steps)

    fig, axis = plt.subplots(1, 1)
    make_reward_plot(axis, avg_rewards, bandits)
    plt.show()


def figure_2_5(k=10, repeats=1000, steps=1000):
    env = TestBed(k=k, baseline=4)
    bandits = [
        GradientBandit(k=k, stepsize=0.1),
        GradientBandit(k=k, stepsize=0.4),
        GradientBandit(k=k, stepsize=0.1, use_baseline=False),
        GradientBandit(k=k, stepsize=0.4, use_baseline=False),
    ]
    _, avg_optact = run_experiment(env, bandits, repeats, steps)

    fig, axis = plt.subplots(1, 1)
    make_optact_plot(axis, avg_optact, bandits)
    plt.show()


def playground(k=10, repeats=2000, steps=1000):
    """ Function for if you want to play around with bandits"""
    env = TestBed(k=k)
    bandits = [
        EpsilonGreedy(k=k, epsilon=0.01),
        EpsilonGreedy(k=k, initial_value=5, epsilon=0.1),
        UpperConfidence(k=k, c=2),
        GradientBandit(k=k, stepsize=0.1),
    ]
    avg_reward, avg_optact = run_experiment(env, bandits, repeats, steps)

    fig, axes = plt.subplots(2, 1)
    make_reward_plot(axes[0], avg_reward, bandits)
    make_optact_plot(axes[1], avg_optact, bandits)
    plt.show()


def main():
    # enable or disable plots you want to see
    figure_2_2()
    figure_2_3()
    figure_2_4()
    figure_2_5()
    # playground(repeats=1000, steps=5000)


if __name__ == "__main__":
    main()
