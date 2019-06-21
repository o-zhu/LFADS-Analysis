### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'			: './savedir/',
	'training_iters'	: 2000,

	# Model configuration
	'n_hidden_enc'		: 64,
	'n_hidden_con'		: 64,
	'n_hidden_gen'		: 64,
	'n_latent'			: 32,
	'n_factors'			: 24,

	# Model parameters
	'KL_cost'			: 3.,
	'recon_cost'		: 1.,
	'weight_cost'		: 1e-3,
	'learning_rate'		: 5e-4,

}

par['savefn'] = 'factors{}_iters{}_KL{}_'.format(\
	par['n_factors'], par['training_iters'], np.int32(100*par['KL_cost']))
