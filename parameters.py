### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'			: './savedir/',
	'training_iters'	: 50000,
	'batch_size'		: 256,

	# Model configuration
	'n_input'			: 100,
	'n_hidden_enc'		: 64,
	'n_hidden_con'		: 64,
	'n_hidden_gen'		: 64,
	'n_latent'			: 32,
	'n_factors'			: 8,

	# Model parameters
	'KL_cost'			: 1.,
	'recon_cost'		: 100.,
	'weight_cost'		: 1e-2,
	'learning_rate'		: 1e-3,

}

def update_parameters(update_dict):

	print('\nUpdating parameters...')
	for key, val in update_dict.items():
		par[key] = val
		print(key.ljust(16), ':', val)
	print('\nParameters updated.')


par['savefn'] = 'full_mouse_factors{}_iters{}_recon{}_'.format(\
	par['n_factors'], par['training_iters'], int(par['recon_cost']))

print('--> Parameters successfully loaded.\n')