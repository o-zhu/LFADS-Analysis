import numpy as np
import pickle
from itertools import product

import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt


def pev_analysis(a, b):

	b = b[:,np.newaxis]

	weights = np.linalg.lstsq(a, b, rcond=None)
	error   = b - a @ weights[0]

	error = error.reshape(b.shape)
	mse   = np.mean(error**2)
	rvar  = np.var(b)
	pev   = 1 - mse/(rvar+1e-9) if rvar > 1e-9 else 0

	return pev, weights[0]


def direction_to_d(direction, num_directions):

	angles = 2*np.pi*direction/num_directions
	d = np.stack([np.cos(angles), np.sin(angles), np.ones_like(angles)], axis=-1)
	return d

##############################################################################

data = pickle.load(open('./datadir/DMS_v0.pkl', 'rb'))
par = data['parameters']
neurals = data['h'].astype(np.float32).transpose([1,2,0])

savefn = './savedir/factors8_iters120000_KL300_factorization_data.pkl'
print('Analyzing: ' + savefn)

factors = pickle.load(open(savefn, 'rb'))['factorization']
reconst = pickle.load(open(savefn, 'rb'))['reconstruction']
weights = pickle.load(open(savefn, 'rb'))['model_weights']['W_fac_rates']

n_factors = factors.shape[-1]

d_sample = direction_to_d(data['sample'], par['num_motion_dirs'])
d_test   = direction_to_d(data['test'], par['num_motion_dirs'])

d_match = np.stack([data['match'], 1-data['match'], np.ones_like(data['match'])], axis=-1)

factor_to_groups = np.zeros([4, n_factors])
exc_fac = np.arange(0,80,2)
inh_fac = np.arange(80,100,2)
exc_dep = np.arange(1,80,2)
inh_dep = np.arange(81,100,2)

factor_to_groups[0,:] = np.mean(weights[:,exc_fac], axis=1)
factor_to_groups[1,:] = np.mean(weights[:,inh_fac], axis=1)
factor_to_groups[2,:] = np.mean(weights[:,exc_dep], axis=1)
factor_to_groups[3,:] = np.mean(weights[:,inh_dep], axis=1)

if True:
	plt.imshow(factor_to_groups, aspect='auto')
	plt.title(savefn)
	plt.xlabel('Factors')
	plt.ylabel('Weight Groups')
	plt.yticks([0,1,2,3])
	plt.colorbar()
	plt.show()
	quit()


test_tuning_factors = np.zeros([par['num_time_steps'], n_factors])
test_tuning_neurals = np.zeros([par['num_time_steps'], par['n_hidden']])
sample_tuning_factors = np.zeros([par['num_time_steps'], n_factors])
sample_tuning_neurals = np.zeros([par['num_time_steps'], par['n_hidden']])
match_tuning_factors = np.zeros([par['num_time_steps'], n_factors])
match_tuning_neurals = np.zeros([par['num_time_steps'], par['n_hidden']])


for n, t in product(range(n_factors), range(par['num_time_steps'])):
	pev, weights = pev_analysis(d_sample, factors[t,:,n])
	sample_tuning_factors[t,n] = pev

	pev, weights = pev_analysis(d_test, factors[t,:,n])
	test_tuning_factors[t,n] = pev

	pev, weights = pev_analysis(d_match, factors[t,:,n])
	match_tuning_factors[t,n] = pev


for n, t in product(range(par['n_hidden']), range(par['num_time_steps'])):
	pev, weights = pev_analysis(d_sample, neurals[t,:,n])
	sample_tuning_neurals[t,n] = pev

	pev, weights = pev_analysis(d_test, neurals[t,:,n])
	test_tuning_neurals[t,n] = pev

	pev, weights = pev_analysis(d_match, neurals[t,:,n])
	match_tuning_neurals[t,n] = pev


fig, ax = plt.subplots(3,2, figsize=(12,8))
ax[0,0].set_title('Sample Tuning for Neurals')
ax[1,0].set_title('Test Tuning for Neurals')
ax[2,0].set_title('Match Tuning for Neurals')
ax[0,1].set_title('Sample Tuning for Factors')
ax[1,1].set_title('Test Tuning for Factors')
ax[2,1].set_title('Match Tuning for Factors')

ax[0,0].imshow(sample_tuning_neurals.T, aspect='auto', clim=(0,1))
ax[1,0].imshow(test_tuning_neurals.T, aspect='auto', clim=(0,1))
ax[2,0].imshow(match_tuning_neurals.T, aspect='auto', clim=(0,1))
ax[0,1].imshow(sample_tuning_factors.T, aspect='auto', clim=(0,1))
ax[1,1].imshow(test_tuning_factors.T, aspect='auto', clim=(0,1))
ax[2,1].imshow(match_tuning_factors.T, aspect='auto', clim=(0,1))

for a, b in product([0,1,2], [0,1]):
	if a == 2:
		ax[a,b].set_xlabel('Time')
	
	if b == 0:
		ax[a,b].set_yticks(np.arange(0,par['n_hidden'],10))
		ax[a,b].set_ylabel('Neurons')
	elif b == 1:
		ax[a,b].set_yticks(np.arange(n_factors))
		ax[a,b].set_ylabel('Factors')

plt.show()
quit()




























quit()




# Project factors to neurals
neural_pev, neural_weights = pev_analysis(factors, neurals)

# Project neurals to factors
factor_pev, factor_weights = pev_analysis(neurals, factors)

neural_weights_groups = np.zeros([neural_weights.shape[0],4])
factor_weights_groups = np.zeros([factor_weights.shape[1],4])

neural_weights_groups[:,0] = np.mean(neural_weights[:,exc_fac], axis=1)
neural_weights_groups[:,1] = np.mean(neural_weights[:,inh_fac], axis=1)
neural_weights_groups[:,2] = np.mean(neural_weights[:,exc_dep], axis=1)
neural_weights_groups[:,3] = np.mean(neural_weights[:,inh_dep], axis=1)

factor_weights_groups[:,0] = np.mean(factor_weights[exc_fac,:], axis=0)
factor_weights_groups[:,1] = np.mean(factor_weights[inh_fac,:], axis=0)
factor_weights_groups[:,2] = np.mean(factor_weights[exc_dep,:], axis=0)
factor_weights_groups[:,3] = np.mean(factor_weights[inh_dep,:], axis=0)


fig, ax = plt.subplots(2,3, figsize=(12,8))

ax[0,0].imshow(neural_weights, aspect='auto')
ax[0,0].set_xlabel('Neurons')
ax[0,0].set_ylabel('Factors')
ax[0,0].set_title('Weights (Project Factors $\\rightarrow$ Neural)')

ax[1,0].imshow(neural_pev.T, aspect='auto', clim=(0,1))
ax[1,0].set_title('Neural PEV')
ax[1,0].set_xlabel('Time')
ax[1,0].set_ylabel('Neurons')

ax[0,1].imshow(factor_weights, aspect='auto')
ax[0,1].set_title('Weights (Project Neural $\\rightarrow$ Factors)')
ax[0,1].set_xlabel('Factors')
ax[0,1].set_ylabel('Neurons')

ax[1,1].imshow(factor_pev.T, aspect='auto', clim=(0,1))
ax[1,1].set_title('Factors PEV')
ax[1,1].set_xlabel('Time')
ax[1,1].set_ylabel('Factors')

ax[0,2].imshow(neural_weights_groups, aspect='auto')
ax[0,2].set_title('Weight Groups (Project Factors $\\rightarrow$ Neurals)')
ax[0,2].set_xlabel('Groups')
ax[0,2].set_ylabel('Factors')
ax[1,2].imshow(factor_weights_groups, aspect='auto')
ax[1,2].set_title('Weight Groups (Project Neurals $\\rightarrow$ Factors)')
ax[1,2].set_xlabel('Groups')
ax[1,2].set_ylabel('Factors')

#plt.savefig('./plotdir/tuning.png', bbox_inches='tight')
plt.show()