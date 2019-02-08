### Authors: Nicolas Y. Masse, Gregory D. Grant

import numpy as np

print("\n--> Loading parameters...")

##############################
### Independent parameters ###
##############################

par = {
	# Setup parameters
	'savedir'			: './savedir/',
	'training_iters'	: 5000,

	# Model configuration
	'n_hidden_enc'		: 64,
	'n_hidden_con'		: 64,
	'n_hidden_gen'		: 64,
	'n_latent'			: 32,
	'n_factors'			: 16,

	# Model parameters
	'KL_cost'			: 1.,
	'recon_cost'		: 1.,
	'weight_cost'		: 1e-3,
	'learning_rate'		: 1e-3,

}