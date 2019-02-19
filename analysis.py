import numpy as np
from scipy.stats import pearsonr, spearmanr
import pickle
from itertools import product
import os, sys

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


def weights_groups(weights):

	# Designate groups
	exc_fac = np.arange(0,80,2)
	inh_fac = np.arange(80,100,2)
	exc_dep = np.arange(1,80,2)
	inh_dep = np.arange(81,100,2)

	# Take absolute value of weights
	weights = np.abs(weights)
	
	# Make groups array and aggregate weights
	factor_to_groups = np.zeros([4, n_factors])
	factor_to_groups[0,:] = np.mean(weights[:,exc_fac], axis=1)
	factor_to_groups[1,:] = np.mean(weights[:,inh_fac], axis=1)
	factor_to_groups[2,:] = np.mean(weights[:,exc_dep], axis=1)
	factor_to_groups[3,:] = np.mean(weights[:,inh_dep], axis=1)

	# Return aggregated weights
	return factor_to_groups


def direction_to_d(direction, num_directions):

	angles = 2*np.pi*direction/num_directions
	d = np.stack([np.cos(angles), np.sin(angles), np.ones_like(angles)], axis=-1)
	return d


def mouse_orientation_pevs(factor_data, orientations):

	n_factors = factor_data.shape[2]
	n_time_steps = factor_data.shape[0]

	orientations = np.squeeze(orientations) * (2*np.pi/360)
	orientations = np.stack([np.cos(orientations), np.sin(orientations), np.ones_like(orientations)], axis=1)

	orientation_pevs = np.zeros([n_factors, n_time_steps])
	for n, t in product(range(n_factors), range(n_time_steps)):
		pev, weights = pev_analysis(orientations, factor_data[t,:,n])
		orientation_pevs[n,t] = pev

	return orientation_pevs

	# plt.imshow(orientation_pevs, aspect='auto', clim=(0,1))
	# plt.xlabel('Time')
	# plt.ylabel('Factors')
	# plt.title('Orientation PEVs for Session {}'.format(s))
	# plt.colorbar()
	# plt.show()


def mouse_one_hot_pevs(factor_data, one_hot):

	n_factors = factor_data.shape[2]
	n_time_steps = factor_data.shape[0]

	one_hot = np.squeeze(one_hot)
	one_hot = np.stack([one_hot, np.ones_like(one_hot)], axis=1)

	one_hot_pevs = np.zeros([n_factors, n_time_steps])
	for n, t in product(range(n_factors), range(n_time_steps)):
		pev, weights = pev_analysis(one_hot, factor_data[t,:,n])
		one_hot_pevs[n,t] = pev

	return one_hot_pevs


##############################################################################

def single_file_analysis(fn, plot=False):
	datafile = pickle.load(open(fn, 'rb'))

	parameters = datafile['parameters']

	neural_data = datafile['neural_data']
	recon_data  = datafile['recon_data']
	factor_data = datafile['factor_data']
	index_data  = datafile['index_data']

	num_sessions = len(neural_data)
	for s in range(num_sessions):

		mouse          = index_data[s]['mouse_id']
		hits           = index_data[s]['Hits']
		miss           = index_data[s]['Miss']
		correct_reject = index_data[s]['CR']
		false_alarm    = index_data[s]['FA']

		orientations   = index_data[s]['Ori']
		react_times    = index_data[s]['RTs']

		hit_miss_inds = np.where((hits+miss) == 1)[1]

		orientation_pevs = mouse_orientation_pevs(factor_data[s], orientations)
		one_hot_pevs = mouse_one_hot_pevs(factor_data[s][:,hit_miss_inds,:], hits[:,hit_miss_inds])

		if plot:
			fig, ax = plt.subplots(1,2,figsize=(12,8), sharex=True, sharey=True)
			im = ax[0].imshow(orientation_pevs, clim=(0,0.5), aspect='auto')
			ax[1].imshow(one_hot_pevs, clim=(0,0.5), aspect='auto')

			ax[0].set_title('Orientation')
			ax[1].set_title('Hit/Miss')

			for i in range(2):
				ax[i].set_xlabel('Frame')
				ax[i].set_ylabel('Factors')

			fig.suptitle('Orientation PEVs, Session {} (Mouse {}, $C_{{recon}}={}$)'.format(s, mouse, 100))
			fig.subplots_adjust(hspace=0.5, right=0.8)
			cbar_ax = fig.add_axes([0.85,0.15,0.05,0.7])
			cb = fig.colorbar(im, cax=cbar_ax)
			cb.ax.set_title('PEV\nValue')
			plt.savefig('./plotdir/recon100/session{:0>3}_mouse{}_PEVs.png'.format(s, mouse))	
			plt.clf()
			plt.close()

		return orientation_pevs, one_hot_pevs, parameters


def obtain_pevs():
	def string_isolate(s, a, b='_'):
		ind_a = s.find(a)
		ind_a += len(a)
		ind_b = s.find(b, ind_a)
		result = s[ind_a:ind_b]
		return int(result)


	fns = [fn for fn in os.listdir('./savedir/') if 'mouse_sweep' in fn]
	# n_input   = []
	# n_hidden  = []
	# n_latent  = []
	# n_factors = []
	max_pevs = np.zeros([2, 4, 3, 3, 3])

	input_map = {20:0,50:1,100:2,150:3}
	hidden_map = {25:0,50:1,100:2}
	latent_map = {16:0,32:1,64:2}
	factors_map = {4:0,8:1,16:2}

	i = 0
	f = len(fns)
	for fn in fns:
		i += 1
		print('Analyzing file {} of {}.'.format(i,f), end='\r')
		ori, one_hot, par = single_file_analysis('./savedir/'+fn)

		inp_ind = input_map[par['n_input']]
		hid_ind = hidden_map[par['n_hidden_enc']]
		lat_ind = latent_map[par['n_latent']]
		fac_ind = factors_map[par['n_factors']]

		max_pevs[0,inp_ind,hid_ind,lat_ind,fac_ind] = ori.max()
		max_pevs[1,inp_ind,hid_ind,lat_ind,fac_ind] = one_hot.max()

	np.save(open('./max_pevs.npy', 'wb'), max_pevs)
	print('\nDone.')


def distance_clustering():
	fn = './savedir/full_mouse_factors16_iters250000_recon100_factorization_data.pkl'
	data = pickle.load(open(fn, 'rb'))

	# Distances are in pixels, metric is 2.69 pixels per micron
	neuron_locs = []
	for i in range(len(data['neural_data'])):
		session_locs = []
		for j in range(data['neural_data'][i].shape[-1]):
			xy_loc = np.squeeze(data['index_data'][i]['XYLoc'])[j][0].astype(np.float32)
			session_locs.append(xy_loc/2.69)
		neuron_locs.append(session_locs)

	neuron_ind = 0
	scatter_x = []
	scatter_y = []
	scatter_a = []
	for s in range(len(data['neural_data'])):
		neurons = data['neural_data'][s].shape[-1]
		neurals = data['neural_data'][s]
		weights = data['model_weights']['W_fac_rates'][:,neuron_ind:neuron_ind+neurons]
		dists   = neuron_locs[s]

		for n1 in range(neurons):
			for n2 in range(n1+1,neurons):

				x = np.sum(np.sqrt(np.square(weights[:,n1] - weights[:,n2])))
				y = np.sum(np.sqrt(np.square(dists[n1] - dists[n2])))
				a = np.sum(np.abs(neurals[:,:,n1] - neurals[:,:,n2]))

				scatter_x.append(x)
				scatter_y.append(y)
				scatter_a.append(a)

	r, p = spearmanr(scatter_x, scatter_y)
	print('R : {}'.format(r))
	print('P : {}'.format(p))

	plt.scatter(scatter_x, scatter_y, s=0.1, c='k', label='Neuron Pair')
	plt.title('Spatial Clustering (All Sessions)'.format(s))
	plt.xlabel('Euclid. Weight Distance (Factors $\\rightarrow$ Reconstruction)')
	plt.ylabel('Spatial Distance (microns)')
	plt.legend()
	plt.show()



distance_clustering()
