import matplotlib.pyplot as plt
import numpy as np
import time

np.random.seed(0)

n_clauses = 100
n_variables = 20

sat_obj = np.random.randint(2, size=n_clauses*3)
assign = np.random.randint(n_variables, size=n_clauses*3)

def max_sat(x):
    obj = x[assign]
    res = abs(obj - sat_obj).reshape(n_clauses, 3)
    return sum(np.sum(res, axis=1) > 0)

n_iter = 6
lmda = 1 / float(n_clauses)

batch_size = 100
elite_frac = 0.04
n_elite = int(batch_size * elite_frac)
n_buffer = max(n_elite // 3, 1)

'''
v_best = 0
for i in xrange(2**n_variables):
	tmp = [int(x) for x in list('{0:0b}'.format(i))]
	rest = [0] * (n_variables - len(tmp))
	x_best = np.array(rest + tmp)
	res = max_sat(x_best)
	print x_best, v_best
	if res > v_best:
		v_best = res
		if res == n_variables:
			break

print "best possible is ", v_best
# best possible is 98
'''

def sat_run(use_buffer=False, use_weight=False):
	n_maxsat = []
	x_buffer = np.zeros((n_buffer, n_variables))
	reward_buffer = np.zeros(n_buffer)
	theta_sat = np.ones(n_variables) / 2.0

	for i in xrange(n_iter):
		#t1 = time.time()
		x_s = np.array([np.random.binomial(1, p=theta_sat) for k in xrange(batch_size)])
		#t2 = time.time()
		rewards = np.array(map(max_sat, x_s))

		if use_buffer:
			x_s = np.concatenate((x_s, x_buffer))
			rewards = np.concatenate((rewards, reward_buffer))

		#t3 = time.time()
		elite_inds = rewards.argsort()[-n_elite:]
		if use_buffer:
			x_buffer = x_s[elite_inds[-n_buffer],:].reshape(n_buffer, n_variables)
			reward_buffer = rewards[elite_inds[-n_buffer]].reshape(n_buffer)

		if use_weight:
			elite_rewards = rewards[elite_inds]
			elite_rewards -= np.min(elite_rewards)
			elite_weights = np.exp(elite_rewards * lmda)
			elite_weights /= sum(elite_weights)

		#t4 = time.time()
		if use_weight:
			theta_sat = np.squeeze((elite_weights.T).dot(x_s[elite_inds]))
		else:
			theta_sat = np.sum(x_s[elite_inds], axis=0) / float(n_elite)

		theta_sat = np.clip(theta_sat, 0, 1)
		#t5 = time.time()
		n_maxsat.append(max_sat(np.ones(n_variables) * (theta_sat > 0.5)))
		#print "Iteration %i. R: %5.3g" % (i+1, max_sat(np.ones(n_variables) * (theta_sat > 0.5)))

	return np.array(n_maxsat)

cem_ms = sat_run()
buffer_ms = sat_run(True)
weight_ms = sat_run(False, True)

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_xlabel('iteration')
ax.set_ylabel('# of satisfied clauses')
x = range(n_iter)
y = np.ones(n_iter) * 98

cem, = plt.plot(x, cem_ms, label='cem')
wb, = plt.plot(x, buffer_ms, label='with buffer')
ww, = plt.plot(x, weight_ms, label='with weight')
yy, = plt.plot(x, y, label='maximum')

plt.legend(handles=[cem, wb, ww, yy])
plt.xticks(x)
plt.title('Max Satisfiability')
plt.show()


