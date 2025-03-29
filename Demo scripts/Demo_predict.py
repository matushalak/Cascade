

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Demo script to predict spiking activity from calcium imaging data

The function "load_neurons_x_time()" loads the input data as a matrix. It can
be modified to load npy-files, mat-files or any other standard format.

The line "spike_prob = cascade.predict( model_name, traces )" performs the
predictions. As input, it uses the loaded calcium recordings ('traces') and
the pretrained model ('model_name'). The output is a matrix with the inferred spike rates.

"""



"""

Import python packages

"""

import os, sys
if 'Demo scripts' in os.getcwd():
    sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
    os.chdir('..')  # change to main directory
print('Current working directory: {}'.format( os.getcwd() ))

from cascade2p import checks
checks.check_packages()

import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth

"""

Define function to load dF/F traces from disk

"""
from mat73 import loadmat as hdf_loadmat
class Dict_to_Class:
    'recursively gets rid of dictionaries'
    def __init__(self, attrs:dict):
        for att_k, att_v in attrs.items():
            if isinstance(att_v, dict):
                setattr(self, att_k, Dict_to_Class(att_v))
            else:
                setattr(self, att_k, att_v)

class SPSIG:
    ''''
    Works for both SPSIG and SPSIG_Res
    Turns SPSIG.mat file into
    '''
    def __init__(self,
                 SPSIG_mat_path:str): # path to ..._SPSIG.mat file
        
        try:
            # old version < v7.3 mat file
            SPSIG_dict = sio.loadmat(SPSIG_mat_path, simplify_cells = True)
            
        except NotImplementedError:
            # Load > v7.3 .mat hdf5 file
            SPSIG_dict = hdf_loadmat(SPSIG_mat_path)
        
        # set attributes to keys of SPSIG_dict
        for k, v in SPSIG_dict.items():
            if k.startswith('__'):
                continue
            
            elif isinstance(v, dict):
                setattr(self, k, Dict_to_Class(v))
            else:
                setattr(self, k, v)


def load_neurons_x_time(file_path):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""

    # replace this with your own code if necessary
    # traces = np.load(file_path)

    # # here numpy dictionary with key 'dff'
#    traces = np.load(file_path, allow_pickle=True).item()['dff']

    # # In case your data is in another format:
    # traces = traces.T        # transpose, if loaded matrix has shape (time, neurons)
    # traces = traces / 100    # normalize to fractions, in case df/f is in Percent

    # traces should be 2d array with shape (neurons, nr_timepoints)

    spsig = SPSIG(file_path)
    traces = spsig.sigCorrected
    frame_rate = spsig.freq
    del spsig # free up memory ?
    if traces.shape[1] < traces.shape[0]:
       traces = traces.T # to match (neurons, nr_timepoints) expected shape
    
    return traces, frame_rate






"""

Load dF/F traces, define frame rate and plot example traces

"""


example_file = '/Volumes/my_SSD/NiNdata/test/group1/Epsilon/20211111/Bar_Tone_LR/Epsilon_20211111_003_normcorr_SPSIG.mat'
# frame_rate = 7.5 # in Hz

traces, frame_rate = load_neurons_x_time( example_file )
print('Number of neurons in dataset:', traces.shape[0])
print('Number of timepoints in dataset:', traces.shape[1])

noise_levels = plot_noise_level_distribution(traces,frame_rate)


#np.random.seed(3952)
# neuron_indices = np.random.randint(traces.shape[0], size=10)
# plot_dFF_traces(traces,neuron_indices,frame_rate)


"""

Load list of available models

"""

# cascade.download_model( 'update_models',verbose = 1)

# yaml_file = open('Pretrained_models/available_models.yaml')
# X = yaml.load(yaml_file)
# list_of_models = list(X.keys())

# for model in list_of_models:
#   print(model)




"""

Select pretrained model and apply to dF/F data

"""
# these seem relevant for out data
# 'GC8m_EXC_15Hz_smoothing100ms_high_noise'
# 'GC8_EXC_15Hz_smoothing100ms_high_noise'
model_name = 'GC8m_EXC_15Hz_smoothing100ms_high_noise'
if model_name not in os.listdir('Pretrained_models'):
    cascade.download_model( model_name,verbose = 1)

print('Using {} model'.format(model_name))
spike_prob = cascade.predict( model_name, traces )


"""

Save predictions to disk

"""


folder = os.path.dirname(example_file)
save_path = os.path.join(folder, 'full_prediction_'+os.path.basename(example_file))

# save as numpy file
#np.save(save_path, spike_prob)
sio.savemat(save_path+'.mat', {'spike_prob':spike_prob})

"""

Plot example predictions

"""
breakpoint()
neuron_indices = np.random.randint(traces.shape[0], size=10)
plot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob)



"""

Plot noise-matched examples from the ground truth

"""

# median_noise = np.round(np.median(noise_levels))
# nb_traces = 8
# duration = 50 # seconds
# plot_noise_matched_ground_truth( model_name, median_noise, frame_rate, nb_traces, duration )
