import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gym.spaces import Discrete

# ================================================================
# Policies
# ================================================================

np.random.seed(8)

class DiscreteLinearPolicy(object):

    def __init__(self, theta):
        k = dim_obs * dim_act
        self.W = theta[:k].reshape(dim_obs, dim_act)
        self.b = theta[k:].reshape(1, dim_act)

    def act(self, ob):
        y = ob.dot(self.W) + self.b
        a = y.argmax()
        return a


class ContinuousLinearPolicy(object):

    def __init__(self, theta):
        k = dim_obs * dim_act
        self.W = theta[:k].reshape(dim_obs, dim_act)
        self.b = theta[k:]

    def act(self, ob):
        a = np.clip(ob.dot(self.W) + self.b, act_space.low, act_space.high)
        return a


class DiscreteNeuralPolicy(object):

    def __init__(self, theta):
        k1 = dim_obs * dim_h1
        k2 = k1 + dim_h1
        k3 = k2 + dim_h1 * dim_act

        self.W1 = theta[:k1].reshape(dim_obs, dim_h1)
        self.b1 = theta[k1:k2].reshape(1, dim_h1)
        self.W2 = theta[k2:k3].reshape(dim_h1, dim_act)
        self.b2 = theta[k3:].reshape(1, dim_act)

    def act(self, ob):
        z1 = ob.dot(self.W1) + self.b1
        h1 = z1 * (z1 > 0)
        y = h1.dot(self.W2) + self.b2
        return y.argmax()


class ContinuousNeuralPolicy(object):

    def __init__(self, theta):
        k1 = dim_obs * dim_h1
        k2 = k1 + dim_h1
        k3 = k2 + dim_h1 * dim_act

        self.W1 = theta[:k1].reshape(dim_obs, dim_h1)
        self.b1 = theta[k1:k2].reshape(1, dim_h1)
        self.W2 = theta[k2:k3].reshape(dim_h1, dim_act)
        self.b2 = theta[k3:].reshape(1, dim_act)

    def act(self, ob):
        z1 = ob.dot(self.W1) + self.b1
        h1 = z1 * (z1 > 0)
        a = h1.dot(self.W2) + self.b2
        return np.clip(a, act_space.low, act_space.high)

# =============================================================================

def discount_rollout(policy, env, num_steps=100, discount=0.9, render=False):
    o = env.reset()
    rewards = []
    steps = 0
    done = False
    for t in xrange(num_steps):
        steps += 1
        a = policy.act(o)
        o, r, done, _ = env.step(a)
        rewards.append(r)
        if done: 
            break

    Gt = 0
    discounted_rewards = []
    threshold = 2 * int(1 / (1 - discount))

    for (i,r) in enumerate(reversed(rewards)):
        Gt = Gt * discount + r
        if i >= threshold:
            discounted_rewards.append(Gt)

    if steps <= threshold:
        return Gt
    else:
        return np.mean(np.array(discounted_rewards))


def rollout(policy, env, num_steps, render=False):
    o = env.reset()
    done = False
    rewards = 0
    while not done:
        a = policy.act(o)
        o, r, done, _ = env.step(a)
        rewards += r

    return rewards


def noisy_evaluation(theta, discount=0.9):
    policy = get_policy(theta)
    return rollout(policy, env, num_steps)


def get_policy(theta):
    if is_discrete:
        return DiscreteNeuralPolicy(theta) if is_neural else DiscreteLinearPolicy(theta)
    else:
        return ContinuousNeuralPolicy(theta) if is_neural else ContinuousLinearPolicy(theta)


# =============================================================================

env_name = 'HalfCheetah-v1'
env = gym.make(env_name)
env.seed(0)
obs_space = env.observation_space
act_space = env.action_space

num_steps = 40
n_iter = 20

batch_size = 20
elite_frac = 0.2

n_elite = int(batch_size * elite_frac)
n_buffer = n_elite // 3
assert n_buffer < n_elite

extra_std = 2.0
extra_decay_time = n_iter // 2

is_neural = True
is_discrete = isinstance(act_space, Discrete)

dim_obs = obs_space.shape[0]
dim_act = act_space.n if is_discrete else act_space.shape[0]
dim_h1 = int(np.sqrt(dim_obs * dim_act))

weight_coeff = 1e-3

if is_neural:
    dim_theta = (dim_obs + 1) * dim_h1 + (dim_h1 + 1) * dim_act
else:
    dim_theta = (dim_obs + 1) * dim_act


def cem(use_buffer=False, use_weight=False):
    theta_buffer = np.zeros((n_buffer, dim_theta))
    reward_buffer = np.zeros(n_buffer) - 10

    theta_mean = np.zeros(dim_theta)
    theta_std = np.ones(dim_theta)
    mean_rewards = []
    for i in xrange(n_iter):
        extra_cov = max(1.0 - i / extra_decay_time, 0) * extra_std**2
        thetas = np.random.multivariate_normal(
            mean=theta_mean, 
            cov=np.diag(np.array(theta_std**2) + extra_cov), 
            size=batch_size)

        rewards = np.array(map(noisy_evaluation, thetas))

        if use_buffer:
            thetas = np.concatenate((thetas, theta_buffer))
            rewards = np.concatenate((rewards, reward_buffer))

        elite_inds = rewards.argsort()[-n_elite:]
        elite_thetas = thetas[elite_inds]

        if use_weight:
            elite_rewards = rewards[elite_inds]
            elite_weights = np.exp(elite_rewards * weight_coeff)
            elite_weights /= sum(elite_weights)

        if use_buffer:
            reward_buffer = rewards[elite_inds[-n_buffer:]]
            theta_buffer = thetas[elite_inds[-n_buffer:]]

        # Update the parameters (minimize CE loss)
        if use_weight:
            theta_mean = np.squeeze((elite_weights.T).dot(elite_thetas))
            theta_std = (elite_thetas - theta_mean).std(axis=0)
        else:
            theta_mean = elite_thetas.mean(axis=0)
            theta_std = elite_thetas.std(axis=0)

        mean_reward = rollout(get_policy(theta_mean), env, num_steps)
        mean_rewards.append(mean_reward)
        #print "Iteration %i. mean_th R: %5.3g mean R: %5.3g Var: %5.3g" % (i+1, mean_reward, np.mean(rewards), np.mean(theta_std))
    return np.array(mean_rewards)


mean_rewards = np.zeros(n_iter)
mean_rewards_use_buffer = np.zeros(n_iter)
mean_rewards_use_weight = np.zeros(n_iter)

n_seed = 3
for i in xrange(n_seed):
    np.random.seed(i)
    mean_rewards += cem()
    mean_rewards_use_buffer += cem(True, False)
    mean_rewards_use_weight += cem(False, True)
    print "iteration ", i

mean_rewards /= n_seed
mean_rewards_use_buffer /= n_seed
mean_rewards_use_weight /= n_seed

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('iteration')
ax.set_ylabel('rewards')
x = range(n_iter)
cem, = plt.plot(x, mean_rewards, label='cem')
wb, = plt.plot(x, mean_rewards_use_buffer, label='with buffer')
ww, = plt.plot(x, mean_rewards_use_weight, label='with weight')
plt.legend(handles=[cem, wb, ww])
plt.xticks(x)
plt.title(env_name)
plt.show()