### Authors: Nicolas Y. Masse, Gregory D. Grant

# Edits to ignore the controller, parameter use_controller=False

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

	def __init__(self, input_data, kl_cost):

		print('\nDefining graph...')

		# Load constant(s)
		self.KL_cost = kl_cost

		# Load neural data
		self.input_data = tf.unstack(input_data, axis=0)

		# Parse data shape
		shape = input_data.get_shape().as_list()
		self.time_steps = shape[0]
		self.batch_size = shape[1]
		self.n_input    = shape[2]
		
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

		#if par['use_controller']:
		gencon_prefixes = ['W_enc_con', 'W_gen_fac', 'W_fac_con', 'W_fac_rates', 'b_enc_con', 'b_gen_fac', 'b_fac_con', 'b_fac_rates']

		# Add variable suffixes
		lstm_suffixes   = ['_enc_f', '_enc_b', '_gen', '_con']
		latent_suffixes = ['_enc', '_con']

		# Declare required variables using prefixes and suffixes
		# Notes: Using Xavier initialization, default scope is top
		self.var_dict = {}

		# LSTM variables
		for p, s in it.product(lstm_prefixes, lstm_suffixes):
			d1 = par['n_hidden'+s[:4]]

			if 'b' in p:
				d0 = 1
			elif 'U' in p:
				d0 = d1
			elif s in ['_enc_f', '_enc_b']:
				d0  = self.n_input
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
			elif 'fac_rates' in p:
				d0 = par['n_factors']
				d1 = self.n_input

			if 'b_' in p:
				d0 = 1

			self.var_dict[p] = self.make_var(p, d0, d1)


	def run_model(self):

		# Collect model data
		self.recon = []
		self.factors = []

		# Aggregate losses while running model for convenience
		self.KL_loss    = 0.
		self.recon_loss = 0.

		###
		# Start by running the forward and backward encoders
		self.enc_f = [None]*len(self.input_data)
		self.enc_b = [None]*len(self.input_data)

		hf = tf.zeros([self.batch_size, par['n_hidden_enc']])
		cf = tf.zeros([self.batch_size, par['n_hidden_enc']])
		hb = tf.zeros([self.batch_size, par['n_hidden_enc']])
		cb = tf.zeros([self.batch_size, par['n_hidden_enc']])

		for t in range(self.time_steps):
			ft = t 								# Forward time step
			bt = len(self.input_data) - (t+1)	# Backward time step

			hf, cf = self.recurrent_cell(hf, cf, self.input_data[ft], '_enc_f')
			hb, cb = self.recurrent_cell(hb, cb, self.input_data[bt], '_enc_b')

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
		hg = g0_mu + tf.exp(0.5*g0_si)*tf.random_normal([self.batch_size, par['n_hidden_gen']], 0, 1)
		cg = tf.zeros([self.batch_size, par['n_hidden_gen']])

		if par['use_controller']:
			hc = tf.zeros([self.batch_size, par['n_hidden_con']])
			cc = tf.zeros([self.batch_size, par['n_hidden_con']])

		# Make initial factor state
		f = tf.nn.relu(hg @ self.var_dict['W_gen_fac'] + self.var_dict['b_gen_fac'])

		# Loop through time (1 to T for both forward and backward encoders)
		for x, hf, hb in zip(self.input_data, self.enc_f, self.enc_b):

			if par['use_controller']:
				# 1. Combine factors and encodings; submit to controller
				Z = tf.concat([hf, hb, f], axis=-1)
				hc, cc = self.recurrent_cell(hc, cc, Z, '_con')

				# 2. Sample from controller
				con_mu = Z @ self.var_dict['W_mu_con'] + self.var_dict['b_mu_con']
				con_si = Z @ self.var_dict['W_si_con'] + self.var_dict['b_si_con']
				con = con_mu + tf.exp(0.5*con_si)*tf.random_normal([self.batch_size, par['n_latent']], 0, 1)

				# -- Add KL loss from controller state
				self.KL_loss += self.KL_loss_lambda(con_mu, con_si)/self.time_steps

				gen_input = tf.concat([f,con],axis=-1)

			else:
				gen_input = f

			# 3. Project sample to generator
			
			hg, cg = self.recurrent_cell(hg, cg, gen_input, '_gen')
			

			# 4. Project generator to factors
			f = tf.nn.relu(hg @ self.var_dict['W_gen_fac'] + self.var_dict['b_gen_fac'])
			self.factors.append(f)

			# 5. Project factors to rates
			r = f @ self.var_dict['W_fac_rates'] + self.var_dict['b_fac_rates']
			self.recon.append(r)

			# -- Add reconstruction loss from rates output
			self.recon_loss += self.mse_loss_lambda(r, x)/self.time_steps

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
		self.KL_loss    = self.KL_cost*self.KL_loss
		self.recon_loss = par['recon_cost']*self.recon_loss

		# Calculate recurrent weight loss for the generator LSTM
		self.weight_loss = 0.
		for name in ['Uf_gen', 'Ui_gen', 'Uo_gen', 'Uc_gen']:
			self.weight_loss += par['weight_cost']*tf.reduce_sum(tf.square(self.var_dict[name]))

		# Collect loss terms
		self.total_loss = self.KL_loss + self.recon_loss + self.weight_loss

		# Build optimizer operation
		opt = tf.train.AdamOptimizer(learning_rate=par['learning_rate'])
		self.train = opt.minimize(self.total_loss)


def main(neural_data, gpu_id=None):

	print('Input data should be an array of trial data.')
	print('Ensure that the input array is of shape [time x trials x neurons].')
	print('Given array shape: {}'.format(neural_data.shape))

	print('\nRunning with parameters:')
	[print('{:<16} : {}'.format(k,v)) for k, v in par.items()]

	if gpu_id is not None:
		os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id

	tf.reset_default_graph()
	x = tf.placeholder(tf.float32, neural_data.shape, 'input')
	k = tf.placeholder(tf.float32, [], 'KL_cost')

	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8) if gpu_id == '0' else tf.GPUOptions()
	with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:

		device = '/cpu:0' if gpu_id is None else '/gpu:0'
		with tf.device(device):
			model = Model(x, k)

		sess.run(tf.global_variables_initializer())

		for i in range(par['training_iters']):

			KL_cost = par['KL_cost']*(i/par['training_iters'])
			_, recon, factors, KL_loss, recon_loss, weight_loss = \
				sess.run([model.train, model.recon, model.factors, model.KL_loss, model.recon_loss,\
					model.weight_loss], feed_dict={x:neural_data, k:KL_cost})

			if i%100 == 0:

				fig, ax = plt.subplots(1,2, figsize=[12,8])
				clim = (np.minimum(neural_data.min(), recon.min()), np.maximum(neural_data.max(), recon.max()))
				ax[0].imshow(neural_data[:,0,:].T, aspect='auto')
				ax[0].set_xlabel('Time')
				ax[0].set_ylabel('Neurons')
				ax[0].set_title('Neural Data')
				ax[1].imshow(recon[:,0,:].T, aspect='auto')
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
					ax[0,t].imshow(neural_data[:,t,:].T, aspect='auto', clim=clim)
					ax[1,t].imshow(factors[:,t,:].T, aspect='auto', clim=(0,factors.max()))

				ax[1,0].set_xlabel('Time')
				ax[1,0].set_ylabel('Neurons')
				ax[0,0].set_title('Neural Data')
				ax[1,0].set_title('Factors')
				plt.suptitle('Neural State Factorization')
				plt.savefig('./plotdir/{}factorization.png'.format(par['savefn']), bbox_inches='tight')
				plt.clf()
				plt.close()

				print('Iter {:>5} | Recon. Loss: {:5.3f} | KL Loss: {:5.3f} | Weight Loss: {:5.3f} |'.format(\
					i, recon_loss, KL_loss, weight_loss))

			weights, = sess.run([model.var_dict])

	print('\nLFADS model complete - neural data successfully processed.')
	print('Compiling reconstructions and factorization data.')
	savedata = {
		'reconstruction'	: recon,
		'factorization'		: factors,
		'final_losses'		: {'recon':recon_loss, 'KL_loss':KL_loss, 'weight_loss':weight_loss},
		'model_weights'		: weights,
		'parameters'		: par
	}

	pickle.dump(savedata, open(par['savedir']+par['savefn']+'factorization_data.pkl', 'wb'))
	print('Data saved!  Model compete.\n')