import LFADS as model
import pickle
import numpy as np
import sys

file = 'activities.pckl'

y, h, syn_x, syn_u = pickle.load(open(file, 'rb'))
neural_data = h

print('Starting shape: {}'.format(neural_data.shape))
#neural_data = np.transpose(neural_data, [1,2,0])
print('Usable shape:  {}'.format(neural_data.shape))

gpu_id = sys.argv[1] if len(sys.argv) > 1 else None

try:
	print('Starting model.\n')
	model.main(neural_data.astype(np.float32), gpu_id)
except KeyboardInterrupt:
	quit('\nQuit via KeyboardInterrupt.\n')