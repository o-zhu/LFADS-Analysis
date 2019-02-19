import LFADS as model
import pickle
import numpy as np
import os, sys
import scipy.io as sio
from parameters import *


def run_model(neural_data, index_data):
	gpu_id = sys.argv[1] if len(sys.argv) > 1 else None

	try:
		print('Starting model.\n')
		model.main(neural_data, index_data, gpu_id)
	except KeyboardInterrupt:
		quit('\nQuit via KeyboardInterrupt.\n')

#################################################################

print('Warning:  Nan\'s replaced with zeros for NPFluo arrays.')
mat_files = [f for f in os.listdir('./datadir/') if '.mat' in f and f[0]=='i']

neural_data = []
index_data = []
for mf in mat_files:

	# Obtain mouse ID
	mouse_id = mf[1:-4]

	# Get sessions and array names
	data = np.squeeze(sio.loadmat('./datadir/'+mf)['LFABS'])
	names = data.dtype.names

	# Change array of arrays into a session list of dicts of arrays
	session_list = [{names[i]:d for i,d in enumerate(session)} for session in data]

	# Add mouse ID to session information
	for session in session_list:
		session['mouse_id'] = mouse_id

	# Iterate over sessions
	for i, session in enumerate(session_list):

		flso = session['SomaFluo']
		flnp = np.nan_to_num(session['NPFluo'])+1e-9

		baseline_flso = np.mean(flso[:3,:,:], axis=(0,1), keepdims=True)
		baseline_flnp = np.mean(flnp[:3,:,:], axis=(0,1), keepdims=True)

		f_soma = (flso - baseline_flso)/baseline_flso
		f_np   = (flnp - baseline_flnp)/baseline_flnp

		ndata = f_soma - 0.7*f_np

		neural_data.append(ndata)
		index_data.append(session)

print('Collected neural data for sweep.')

#################################################################

def grid_sweep():

	for n_input in [100,150]:
		for n_hidden in [25, 50, 100]:
			for n_latent in [16, 32, 64]:
				for n_factors in [4,8,16]:

					updates = {}
					savefn = 'mouse_sweep_input{}_hidden{}_latent{}_factors{}_iters{}_recon{}_'.format(\
						n_input, n_hidden, n_latent, n_factors, par['training_iters'], int(par['recon_cost']))
					updates['n_input'] = n_input
					updates['n_hidden_enc'] = n_hidden
					updates['n_hidden_con'] = n_hidden
					updates['n_hidden_gen'] = n_hidden
					updates['n_latent'] = n_latent
					updates['n_factors'] = n_factors
					updates['savefn'] = savefn

					update_parameters(updates)
					run_model(neural_data, index_data)


grid_sweep()
"""
Sweeps:
	n_input = [20, 50, 100, 150]
	n_hidden = [25, 50, 100]
	n_latent = [16, 32, 64]
	n_factors = [4, 8, 16]
"""
