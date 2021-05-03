import gym
import gym_wumpusworld
import numpy as np
import util
import random
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

env = gym.make('Wumpus-v0')
Q = np.zeros([10, 6])
alpha = 1
gamma = 1
episode = 200
epsilon = 1
qval = util.Counter()
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = episode // 2
epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)
alpha_decay_value = 0.005


def get_q_value(state, action):
    if not qval[str(state), action]:
        qval[str(state), action] = 0.0

    return qval[str(state), action]


def compute_action_from_q_values(state):
    val = -1000000
    act = None
    # print("epsilon is {}".format(epsilon))
    if util.flipCoin(epsilon):
        act = random.choice(env.action_space)
    for action in env.action_space:
        qval = get_q_value(state, action)
        if qval > val:
            val = qval
            act = action
    # print("the action is", act)
    return act


def compute_value_from_q_value(state):
    l = []
    for action in env.action_space:
        l.append(qval[str(state), action])
    return max(l)


legend_elements = [Line2D([0], [0], marker='o', color='b', label='episodes = {}'.format(episode)),
                   Line2D([0], [0], marker='o', color='g', label='gamma = {}'.format(gamma)),
                   Line2D([0], [0], marker='o', color='r', label='alpha = {} and decaying'.format(alpha))]

plt.title("QLearning")
# plt.xlabel("Number of episodes")
# plt.ylabel("Reward per episode")
# plt.legend(handles=legend_elements, loc='upper left')
for i in range(episode):
    observation = env.reset()
    d = False
    j = 0
    repisode = 0
    print("episode num ", i + 1)
    while not d:
        j += 1
        # print("j is", j)
        env.render()
        # print("alpha is", alpha)
        # print(qval["{'x': 0, 'y': 0, 'gold': False, 'direction': <Direction.EAST: 1>, 'arrow': True, 'stench': False, 'breeze': False, 'glitter': False, 'bump': False, 'scream': False}", 2])
        a = compute_action_from_q_values(observation)
        observation1, reward, d, info = env.step(a)
        repisode += reward
        sample = reward + gamma * compute_value_from_q_value(observation1)
        qval[str(observation), a] = (1 - alpha) * qval[str(observation), a] + alpha * sample
        observation = observation1
        print("run through ", j)
        if j > 5999:
            plt.plot(i + 1, repisode, 'go', linewidth=2, markersize=2)
            plt.draw()
            plt.pause(0.01)
            if END_EPSILON_DECAYING >= i + 1 >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
            if i + 1 > episode // 2:
                epsilon = 0
            if alpha > .1:
                alpha -= alpha_decay_value
            break
        if d:
            plt.plot(i + 1, repisode, 'go', linewidth=2, markersize=2)
            plt.draw()
            plt.pause(0.01)
            if END_EPSILON_DECAYING >= i + 1 >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value
            if i + 1 > episode // 2:
                epsilon = 0
            if alpha > .1:
                alpha -= alpha_decay_value
            break

plt.pause(50)