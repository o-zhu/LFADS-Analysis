### Authors: Nicolas Y. Masse, Gregory D. Grant

# Required packages
import tensorflow as tf
import numpy as np
import pickle
import os, sys, time
import itertools as it

# Plotting suite
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Model parameters
from parameters import *

# Match GPU IDs to nvidia-smi command
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

# Ignore Tensorflow startup warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'


class Model:
	"""
	A model for performing LFADS encoding based on
	https://www.biorxiv.org/content/biorxiv/early/2017/06/20/152884.full.pdf
	"""

	def __init__(self, input_data, session_mask, cost_mult):

		print('\nDefining graph...')

		# Load constant(s)
		self.cost_mult = cost_mult
		self.session_mask = session_mask

		# Load neural data
		self.input_data = input_data

		# Parse data shape
		shape = input_data.get_shape().as_list()
		self.n_input    = shape[2]
		self.time_steps = shape[0]
		
		# Declare model variables
		self.declare_variables()

		# Set up loss terms
		self.KL_loss_lambda  = lambda mu, si : -0.5*tf.reduce_mean(1+si-tf.square(mu)-tf.exp(si))
		self.mse_loss_lambda = lambda x, y : tf.reduce_mean(tf.square(x-y))

		# Build the LFADS encoder
		self.run_model()

		# Optimize the model
		self.optimize()

		print('Graph defined.\n')


	def make_var(self, name, input_shape, output_shape):
		if input_shape == 1:
			v = tf.zeros([input_shape, output_shape])
		else:
			m = np.sqrt(6/(input_shape+output_shape))
			v = tf.random_uniform([input_shape, output_shape], -m, m)
		
		return tf.get_variable(name, initializer=v)


	def declare_variables(self):
		""" Initialize all required variables """

		### Notes:
		# Four LSTMS: encoder forward, encoder backward, controller, and generator
		# Two latent spaces: Encoder and controller
		# One recon. output: Rates
		# Other connections: Generator to factors, factors to controller

		# Collect variables prefixes
		lstm_prefixes   = ['Wf', 'Wi', 'Wo', 'Wc', 'Uf', 'Ui', 'Uo', 'Uc', 'bf', 'bi', 'bo', 'bc']
		latent_prefixes = ['W_mu', 'W_si', 'b_mu', 'b_si']
		gencon_prefixes = ['W_enc_con', 'W_gen_fac', 'W_fac_con', \
			'b_enc_con', 'b_gen_fac', 'b_fac_con']
		session_prefixes = ['W_sess_inp', 'b_sess_inp', 'W_fac_rates', 'b_fac_rates']

		# Add variable suffixes
		lstm_suffixes   = ['_enc_f', '_enc_b', '_gen', '_con']
		latent_suffixes = ['_enc', '_con']

		# Declare required variables using prefixes and suffixes
		# Notes: Using Xavier initialization, default scope is top
		self.var_dict = {}
		self.var_assign = {}
		self.session_vars = {}

		# LSTM variables
		for p, s in it.product(lstm_prefixes, lstm_suffixes):
			d1 = par['n_hidden'+s[:4]]

			if 'b' in p:
				d0 = 1
			elif 'U' in p:
				d0 = d1
			elif s in ['_enc_f', '_enc_b']:
				d0  = par['n_input']
			elif s == '_con':
				d0  = 2*par['n_hidden_enc'] + par['n_factors']
			elif s == '_gen':
				d0  = par['n_latent']

			self.var_dict[p+s] = self.make_var(p+s, d0, d1)

		# Latent variables
		for p, s in it.product(latent_prefixes, latent_suffixes):

			if 'enc' in s:
				d0 = 2*par['n_hidden_enc']
				d1 = par['n_hidden_gen']
			elif 'con' in s:
				d0 = 2*par['n_hidden_enc'] + par['n_factors']
				d1 = par['n_latent']

			if 'b_' in p:
				d0 = 1

			self.var_dict[p+s] = self.make_var(p+s, d0, d1)

		# Connection variables
		for p in gencon_prefixes:

			if 'enc_con' in p:
				d0 = par['n_hidden_enc']
				d1 = par['n_hidden_con']
			elif 'gen_fac' in p:
				d0 = par['n_hidden_gen']
				d1 = par['n_factors']
			elif 'fac_con' in p:
				d0 = par['n_factors']
				d1 = par['n_hidden_con']

			if 'b_' in p:
				d0 = 1

			self.var_dict[p] = self.make_var(p, d0, d1)

		# Session variables
		for p in session_prefixes:

			if p == 'W_sess_inp':
				d0, d1 = self.n_input, par['n_input']
				m = self.session_mask[:,tf.newaxis]
			elif p == 'W_fac_rates':
				d0, d1 = par['n_factors'], self.n_input
				m = self.session_mask[tf.newaxis,:]
			elif p == 'b_fac_rates':
				d0, d1 = 1, self.n_input
				m = self.session_mask[tf.newaxis,:]

			self.var_dict[p] = m*self.make_var(p, d0, d1)


	def run_model(self):

		# Collect model data
		self.recon = []
		self.factors = []

		# Aggregate losses while running model for convenience
		self.KL_loss    = 0.
		self.recon_loss = 0.

		# Project session neurons to input space
		enc_input_data = tf.tensordot(self.input_data, self.var_dict['W_sess_inp'], axes=[[2],[0]])
		enc_input_data = tf.unstack(enc_input_data, axis=0)
		self.input_data = tf.unstack(self.input_data, axis=0)

		###
		# Start by running the forward and backward encoders
		self.enc_f = [None]*self.time_steps
		self.enc_b = [None]*self.time_steps

		hf = tf.zeros([par['batch_size'], par['n_hidden_enc']])
		cf = tf.zeros([par['batch_size'], par['n_hidden_enc']])
		hb = tf.zeros([par['batch_size'], par['n_hidden_enc']])
		cb = tf.zeros([par['batch_size'], par['n_hidden_enc']])

		for t in range(self.time_steps):
			ft = t 								# Forward time step
			bt = self.time_steps - (t+1)	# Backward time step

			hf, cf = self.recurrent_cell(hf, cf, enc_input_data[ft], '_enc_f')
			hb, cb = self.recurrent_cell(hb, cb, enc_input_data[bt], '_enc_b')

			self.enc_f[ft] = hf
			self.enc_b[bt] = hb
		#
		###

		# Sample initial state for generator, using final states of encoders
		Z = tf.concat([self.enc_f[-1], self.enc_b[0]], axis=-1)
		g0_mu = Z @ self.var_dict['W_mu_enc'] + self.var_dict['b_mu_enc']
		g0_si = Z @ self.var_dict['W_si_enc'] + self.var_dict['b_si_enc']

		# Add KL loss from zeroth generator latent encoding
		self.KL_loss += self.KL_loss_lambda(g0_mu, g0_si)

		# Make initial generator and controller states
		hg = g0_mu + tf.exp(0.5*g0_si)*tf.random_normal([par['batch_size'], par['n_hidden_gen']], 0, 1)
		cg = tf.zeros([par['batch_size'], par['n_hidden_gen']])
		hc = tf.zeros([par['batch_size'], par['n_hidden_con']])
		cc = tf.zeros([par['batch_size'], par['n_hidden_con']])

		# Make initial factor state
		f = tf.nn.relu(hg @ self.var_dict['W_gen_fac'] + self.var_dict['b_gen_fac'])

		# Loop through time (1 to T for both forward and backward encoders)
		for x, hf, hb in zip(self.input_data, self.enc_f, self.enc_b):

			# 1. Combine factors and encodings; submit to controller
			Z = tf.concat([hf, hb, f], axis=-1)
			hc, cc = self.recurrent_cell(hc, cc, Z, '_con')

			# 2. Sample from controller
			con_mu = Z @ self.var_dict['W_mu_con'] + self.var_dict['b_mu_con']
			con_si = Z @ self.var_dict['W_si_con'] + self.var_dict['b_si_con']
			con = con_mu + tf.exp(0.5*con_si)*tf.random_normal([par['batch_size'], par['n_latent']], 0, 1)

			# -- Add KL loss from controller state
			self.KL_loss += self.KL_loss_lambda(con_mu, con_si)/self.time_steps

			# 3. Project sample to generator
			hg, cg = self.recurrent_cell(hg, cg, con, '_gen')

			# 4. Project generator to factors
			f = tf.nn.relu(hg @ self.var_dict['W_gen_fac'] + self.var_dict['b_gen_fac'])
			self.factors.append(f)

			# 5. Project factors to rates
			r = f @ self.var_dict['W_fac_rates'] + self.var_dict['b_fac_rates']
			self.recon.append(r)

			# -- Add reconstruction loss from rates output (only for relevant session)
			sm = self.session_mask[tf.newaxis,:]
			self.recon_loss += self.mse_loss_lambda(sm*r, sm*x)/self.time_steps

		# Stack records
		self.recon = tf.stack(self.recon, axis=0)
		self.factors = tf.stack(self.factors, axis=0)


	def recurrent_cell(self, h, c, x, s):
		""" Compute LSTM state from previous state, inputs, and selected vars """

		# f : forgetting gate, i : input gate, c : cell state, o : output gate
		f  = tf.sigmoid(x @ self.var_dict['Wf'+s] + h @ self.var_dict['Uf'+s] + self.var_dict['bf'+s])
		i  = tf.sigmoid(x @ self.var_dict['Wi'+s] + h @ self.var_dict['Ui'+s] + self.var_dict['bi'+s])
		o  = tf.sigmoid(x @ self.var_dict['Wo'+s] + h @ self.var_dict['Uo'+s] + self.var_dict['bo'+s])
		cn = tf.tanh(x @ self.var_dict['Wc'+s] + h @ self.var_dict['Uc'+s] + self.var_dict['bc'+s])
		
		# Calculate current state
		c  = f * c + i * cn
		h  = o * tf.tanh(c)

		return h, c


	def optimize(self):

		# Scale loss terms
		self.KL_loss    = self.cost_mult*par['KL_cost']*self.KL_loss
		self.recon_loss = par['recon_cost']*self.recon_loss

		# Calculate recurrent weight loss for the generator LSTM
		self.weight_loss = 0.
		for name in ['Uf_gen', 'Ui_gen', 'Uo_gen', 'Uc_gen']:
			self.weight_loss += par['weight_cost']*tf.reduce_sum(tf.square(self.var_dict[name]))
		self.weight_loss = self.cost_mult*par['weight_cost']*self.weight_loss

		# Collect loss terms
		self.total_loss = self.KL_loss + self.recon_loss + self.weight_loss

		# Build optimizer operation
		opt = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
		self.train = opt.minimize(self.total_loss)


def main(neural_data, index_data, gpu_id=None):

	print('Input data should be a list of arrays of neural data.')
	print('Ensure that the input arrays are of shape [time x trials x neurons].')
	print('Given array shapes:')
	for i, n in enumerate(neural_data):
		print('   S{}: {}'.format(i, n.shape))

	# Get neural data sizes and set up session masks
	time_steps = neural_data[0].shape[0]
	total_neurons = np.sum(n.shape[-1] for n in neural_data)

	ind = 0
	session_inds = []
	session_masks = np.zeros([len(neural_data), total_neurons], dtype=np.float32)
	for i, n in enumerate(neural_data):
		session_masks[i,ind:ind+n.shape[-1]] = 1.
		ind_post = ind + n.shape[-1]
		session_inds.append([ind, ind_post])
		ind = ind_post

	# Print parameters being used
	print('\nRunning with parameters:')
	[print('{:<16} : {}'.format(k,v)) for k, v in par.items()]

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, [time_steps, par['batch_size'], total_neurons], 'input')
	m = tf.placeholder(tf.float32, [total_neurons], 'session_mask')
	c = tf.placeholder(tf.float32, [], 'cost_multiplier')

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, m, c)

		sess.run(tf.global_variables_initializer())

		for i in range(par['training_iters']):

			# Sample from all available sessions to put together "session data"
			session_data = np.zeros([time_steps, par['batch_size'], total_neurons])
			for nd, ind in zip(neural_data, session_inds):
				s_inds = np.random.randint(nd.shape[-1], size=par['batch_size'])
				session_data[:,:,ind[0]:ind[1]] = nd[:,s_inds,:]
			
			# Select a session by using a specific session mask
			session_mask = session_masks[np.random.randint(len(neural_data)),:]

			cost_mult = i/par['training_iters']
			_, recon, factors, KL_loss, recon_loss, weight_loss = \
				sess.run([model.train, model.recon, model.factors, model.KL_loss, model.recon_loss,\
					model.weight_loss], feed_dict={x:session_data, m:session_mask, c:cost_mult})

			if i%500 == 0:

				session = session_data * session_mask

				fig, ax = plt.subplots(1,2, figsize=[12,8])
				clim = (np.minimum(session.min(), recon.min()), np.maximum(session.max(), recon.max()))
				ax[0].imshow(session[:,0,:].T, aspect='auto', clim=clim)
				ax[0].set_xlabel('Time')
				ax[0].set_ylabel('Neurons')
				ax[0].set_title('Neural Data')
				ax[1].imshow(recon[:,0,:].T, aspect='auto', clim=clim)
				ax[1].set_xlabel('Time')
				ax[1].set_ylabel('Neurons')
				ax[1].set_title('Reconstruction')
				plt.suptitle('Reconstruction Comparison')
				plt.savefig('./plotdir/{}reconstruction.png'.format(par['savefn']), bbox_inches='tight')
				plt.clf()
				plt.close()

				trials = 4
				fig, ax = plt.subplots(2, trials, figsize=[12,8])
				for t in range(trials):
					ax[0,t].imshow(session[:,t,:].T, aspect='auto', clim=clim)
					ax[1,t].imshow(factors[:,t,:].T, aspect='auto', clim=(0,factors.max()))

				ax[1,0].set_xlabel('Time')
				ax[1,0].set_ylabel('Neurons')
				ax[0,0].set_title('Neural Data')
				ax[1,0].set_title('Factors')
				plt.suptitle('Neural State Factorization')
				plt.savefig('./plotdir/{}factorization.png'.format(par['savefn']), bbox_inches='tight')
				plt.clf()
				plt.close()

				print('Iter {:>6} | Recon. Loss: {:5.6f} | KL Loss: {:5.6f} | Weight Loss: {:5.6f} |'.format(\
					i, recon_loss, KL_loss, weight_loss))

		print('\nLFADS training complete.  Encoding session factors.')

		# Record session reconstruction and factors across sesions
		recon_data = []
		factor_data = []

		# Iterate over sessions
		for j, (s_inds, sm, nd) in enumerate(zip(session_inds, session_masks, neural_data)):

			# Record session reconstruction and factors within session
			session_recon = []
			session_factors = []

			# Get number of trials in the session, convert to batches
			session_trials = nd.shape[1]
			batches = int(np.ceil(session_trials/par['batch_size']))

			# Iterate over session batches
			for b in range(batches):

				# Designate neural data batch and trial indices
				batch = np.zeros([time_steps,par['batch_size'],total_neurons], dtype=np.float32)
				b_inds = [b*par['batch_size'],(b+1)*par['batch_size']]

				# If not enough trials remain, do a partial batch.  Otherwise, do a full batch
				if b_inds[1] > session_trials:
					curr_trials = session_trials - b_inds[0]
					batch[:,:curr_trials,s_inds[0]:s_inds[1]] = nd[:,b_inds[0]:session_trials,:]
				else:
					curr_trials = par['batch_size']
					batch[:,:,s_inds[0]:s_inds[1]] = nd[:,b_inds[0]:b_inds[1],:]

				# Get reconstruction and factors from the trained model
				recon, factors = sess.run([model.recon, model.factors], feed_dict={x:batch, m:sm})

				# Record reconstruction and factors
				session_recon.append(recon[:,:curr_trials,s_inds[0]:s_inds[1]])
				session_factors.append(factors[:,:curr_trials])

			# Combine reconstruction and factor records to match neural data sizes
			session_recon = np.concatenate(session_recon, axis=1)
			session_factors = np.concatenate(session_factors, axis=1)

			# Add records to overall records
			recon_data.append(session_recon)
			factor_data.append(session_factors)

		# Record trained model weights
		weights, = sess.run([model.var_dict], feed_dict={m:np.ones_like(session_mask)})

	print('Neural, reconstruction, and factors data successfully processed.')
	print('Compiling data for saving.')

	savedata = {
		'neural_data'		: neural_data,
		'index_data'		: index_data,
		'recon_data'		: recon_data,
		'factor_data'		: factor_data,
		'model_weights'		: weights,
		'parameters'		: par
		# 'final_losses'		: {'recon':recon_loss, 'KL_loss':KL_loss, 'weight_loss':weight_loss},
	}

	pickle.dump(savedata, open(par['savedir']+par['savefn']+'factorization_data.pkl', 'wb'))
	print('Data saved!  Model compete.\n')