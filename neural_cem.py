import gym
import matplotlib.pyplot as plt
import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim

from torch.autograd import Variable

class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(dim_theta, 100)
        self.fc21 = nn.Linear(100, 10)
        self.fc22 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 100)
        self.fc4 = nn.Linear(100, dim_theta)

        self.relu = nn.ReLU()

    def encode(self, th):
        h1 = self.relu(self.fc1(th))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = Variable(std.data.new(std.size()).normal_())
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, th):
        mu, logvar = self.encode(th)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class LB(nn.Module):
    def __init__(self):
        super(LB, self).__init__()

        self.fc1 = nn.Linear(dim_theta, 100)
        self.fc2 = nn.Linear(100, 10)
        self.fc3 = nn.Linear(10, 100)
        self.fc4 = nn.Linear(100, dim_theta)

        self.relu = nn.ReLU()

    def encode(self, th):
        h1 = self.relu(self.fc1(th))
        return self.fc2(h1)

    def decode(self, z):
        h3 = self.relu(self.fc3(z))
        return self.fc4(h3)

    def forward(self, th):
        z = self.encode(th)
        return self.decode(z), z


def loss_function2(recon_th, th, z, r):
    MSE = torch.sum((recon_th - th) ** 2) / float(batch_size)
    r_exp = torch.exp(r / 500)
    weights = (r_exp / torch.sum(r_exp)).float()
    k = torch.sum(z.pow(2), dim=1)
    KLD = torch.sum(weights * k)
    #print MSE.data[0], KLD.data[0]
    return MSE + KLD

def loss_function(recon_th, th, mu, logvar):
    MSE = torch.sum((recon_th - th) ** 2) / float(batch_size)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD


class PolicyNet(object):
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


def rollout(policy, env, num_steps=100):
    o = env.reset()
    rewards = 0
    done = False
    step = 0
    while not done:
        step += 1
        a = policy.act(o)
        o, r, done, _ = env.step(a)
        rewards += r

    return rewards


def noisy_evaluation(theta):
    policy = PolicyNet(theta)
    return rollout(policy, env)


torch.manual_seed(0)

env_name = 'Walker2d-v1'
env = gym.make(env_name)
env.seed(0)
obs_space = env.observation_space
act_space = env.action_space
dim_obs = obs_space.shape[0]
dim_act = act_space.shape[0]

dim_h1 = int(np.sqrt(dim_obs * dim_act))
dim_theta = (dim_obs + 1) * dim_h1 + (dim_h1 + 1) * dim_act

n_iter = 20
batch_size = 20
elite_frac = 0.2
n_elite = int(batch_size * elite_frac)
n_means = 2

extra_std = 2.0
extra_decay_time = n_iter // 2

def vae_train():
    model = VAE()
    t = 0
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    R = []

    for i in xrange(n_iter):
        model.train()

        thetas = model.decode(Variable(torch.randn(batch_size, 10))).data.numpy()
        rewards = np.array(map(noisy_evaluation, thetas))
        elite_inds = rewards.argsort()[-n_elite:]
        elite_thetas = Variable(torch.from_numpy(thetas[elite_inds]), requires_grad=False)
        elite_rewards = rewards[elite_inds]

        optimizer.zero_grad()
        recon_batch, mu, logvar = model(elite_thetas)
        loss = loss_function(recon_batch, elite_thetas, mu, logvar)
        loss.backward()
        optimizer.step()

        R.append(np.mean(elite_rewards))
        print "iteration {} mean rewards {} loss {}".format(i, np.mean(elite_rewards), loss.data[0])
    return R

def lb_train():
    model = LB()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    R = []
    for i in xrange(n_iter):
        model.train()

        thetas = model.decode(Variable(torch.randn(batch_size, 10))).data.numpy()
        rewards = np.array(map(noisy_evaluation, thetas))
        elite_inds = rewards.argsort()[-n_elite:]
        elite_thetas = Variable(torch.from_numpy(thetas[elite_inds]), requires_grad=False)
        elite_rewards = rewards[elite_inds]

        optimizer.zero_grad()
        recon_batch, z = model(elite_thetas)
        loss = loss_function2(recon_batch, elite_thetas, z, Variable(torch.from_numpy(elite_rewards), requires_grad=False))
        loss.backward()
        optimizer.step()

        R.append(np.mean(elite_rewards))
        print "iteration {} mean rewards {} loss {}".format(i, np.mean(elite_rewards), loss.data[0])
    return R

def cem():
    theta_mean = np.zeros(dim_theta)
    theta_std = np.ones(dim_theta)
    mean_rewards = []

    for i in xrange(n_iter):
        t0 = time.time()
        extra_cov = max(1.0 - i / extra_decay_time, 0) * extra_std**2
        thetas = np.random.multivariate_normal(
            mean=theta_mean, 
            cov=np.diag(np.array(theta_std**2) + extra_cov), 
            size=batch_size)

        rewards = np.array(map(noisy_evaluation, thetas))
        elite_inds = rewards.argsort()[-n_elite:]
        elite_thetas = thetas[elite_inds]
        elite_rewards = rewards[elite_inds]
        # Update the parameters (minimize CE loss)
        theta_mean = elite_thetas.mean(axis=0)
        theta_std = elite_thetas.std(axis=0)

        mean_reward = noisy_evaluation(theta_mean)
        mean_rewards.append(mean_reward)

        print "Iteration %i. mean R: %5.3g" % (i+1, np.mean(elite_rewards))
    return np.array(mean_rewards)

def k_mean(x):
    means1 = np.array(x[-1], copy=True)
    means2 = np.array(x[-2], copy=True)
    list1 = None
    list2 = None
    old_list1 = [1]
    while not old_list1 == list1:
        old_list1 = list1
        list1 = []
        list2 = []
        for i in xrange(n_elite):
            if np.sum((means1 - x[i])**2) < np.sum((means2 - x[i])**2):
                list1.append(i)
            else:
                list2.append(i)
        means1 = (x[np.array(list1)]).mean(axis=0)
        means2 = (x[np.array(list2)]).mean(axis=0)

    std = np.concatenate(( 
        (x[np.array(list1)]).std(axis=0).reshape(1, dim_theta), 
        (x[np.array(list2)]).std(axis=0).reshape(1, dim_theta) ), axis=0)
    means = np.concatenate((
        means1.reshape(1, dim_theta), 
        means2.reshape(1, dim_theta) ), axis=0)
    #print std
    return means, std, [len(list1), len(list2)]



def beam_cem():
    kmeans = np.zeros((n_means, dim_theta))
    std = np.ones((n_means, dim_theta))
    max_mean_rewards = []
    counter = np.array([batch_size//n_means] * n_means)
    for i in xrange(n_iter):
        extra_cov = max(1.0 - i / extra_decay_time, 0.1) * extra_std**2

        thetas = np.random.multivariate_normal(
            mean=kmeans[-1],
            cov=np.diag(np.array(std[-1]**2) + extra_cov), 
            size=counter[-1] * batch_size // n_means)
        for k in xrange(n_means-1):
            thetas = np.concatenate((thetas, np.random.multivariate_normal(
            mean=kmeans[k],
            cov=np.diag(np.array(std[k]**2) + extra_cov), 
            size=counter[k] * batch_size // n_means)))

        rewards = np.array(map(noisy_evaluation, thetas))
        elite_inds = rewards.argsort()[-n_elite:]
        elite_samples = thetas[elite_inds]
        elite_rewards = rewards[elite_inds]
        # Update the parameters (minimize CE loss)
        kmeans, std, counter = k_mean(elite_samples)

        r1 = noisy_evaluation(kmeans[0,:])
        r2 = noisy_evaluation(kmeans[1,:])
        max_mean_rewards.append(max(r1, r2))
        print "Iteration %i. R1:%5.3g R2:%5.3g" % (i+1, r1, r2)
    return np.array(max_mean_rewards)

n_seed = 3

bc = np.zeros(n_iter)
ce = np.zeros(n_iter)

for i in xrange(n_seed):
    np.random.seed(i)
    bc += beam_cem()
    ce += cem()
    print "iteration ", i

bc /= n_seed
ce /= n_seed

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('iteration')
ax.set_ylabel('rewards')
x = range(n_iter)
a, = plt.plot(x, ce, label='cem')
b, = plt.plot(x, bc, label='beam cem')
plt.legend(handles=[a, b])

plt.xticks(np.arange(0, 20, 1))
plt.title(env_name)
plt.show()
'''
n_seed = 5

mu_vae = np.zeros(n_iter)
mu_lb = np.zeros(n_iter)


for i in xrange(n_seed):
    np.random.seed(i)
    mu_vae += vae_train()
    mu_lb += lb_train()
    print "iteration ", i

mu_lb /= n_seed
mu_vae /= n_seed

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('iteration')
ax.set_ylabel('rewards')
x = range(n_iter)
a, = plt.plot(x, mu_vae, label='vae')
b, = plt.plot(x, mu_lb, label='vae w/mean')
plt.legend(handles=[a, b])

plt.xticks(np.arange(0, 100, 10))
plt.title(env_name)
plt.show()
'''
