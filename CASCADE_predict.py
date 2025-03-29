#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@matushalak modified for SPSIG format for Huub Terra's Dark-rearing project

Script to predict spiking activity from calcium imaging data

The function "load_neurons_x_time()" loads the input data as a matrix. It can
be modified to load npy-files, mat-files or any other standard format.

The line "spike_prob = cascade.predict( model_name, traces )" performs the
predictions. As input, it uses the loaded calcium recordings ('traces') and
the pretrained model ('model_name'). The output is a matrix with the inferred spike rates.

"""
import os, sys, glob, re

import numpy as np
import scipy.io as sio
import ruamel.yaml as yaml
yaml = yaml.YAML(typ='rt')

from cascade2p import cascade # local folder
from cascade2p.utils import plot_dFF_traces, plot_noise_level_distribution, plot_noise_matched_ground_truth
from matplotlib import pyplot as plt
from tkinter import filedialog
from mat73 import loadmat as hdf_loadmat

"""

Main function that runs CASCADE on a file and saves CASCADE output in the correct folders

"""
def main(spsig_file:str,
         PLOT:bool = False):
    """

    Load dF/F traces, define frame rate and plot example traces

    """
    print(f'Estimating spikes from {spsig_file}')

    traces, frame_rate = load_neurons_x_time( spsig_file )
    print('Number of neurons in dataset:', traces.shape[0])
    print('Number of timepoints in dataset:', traces.shape[1])

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
    # break it up into chunks if more than 100 neurons
    if traces.shape[0] > 100:
        spike_prob = np.empty_like(traces)
        hundreds  = traces.shape[0] // 100
        ranges = [(r * 100, r * 100 + 100) for r in range(hundreds)]
        for start, end in ranges:
            spike_prob[start:end,:] = cascade.predict(model_name, traces[start:end, :])
    else:
        spike_prob = cascade.predict( model_name, traces )

    """

    Save predictions to disk

    """
    folder = os.path.dirname(spsig_file)
    save_path = os.path.join(folder, 'full_prediction_'+os.path.basename(spsig_file))

    # save 
    sio.savemat(save_path+'.mat', {'spike_prob':spike_prob})

    """

    Plot example predictions

    """
    if PLOT:
        print('Feel free to explore the dataset in the terminal with: \nneuron_indices = np.random.randint(traces.shape[0], size=10)\nplot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob)\nplt.show()')
        neuron_indices = np.random.randint(traces.shape[0], size=10)
        plot_dFF_traces(traces,neuron_indices,frame_rate,spike_prob)
        plt.show()

        breakpoint()

"""

Classes that nicely process matlab files into python classes easier to work with 

"""

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

"""

Define function to load dF/F traces from disk, SPSIG specific

"""
def load_neurons_x_time(file_path):
    """Custom method to load data as 2d array with shape (neurons, nr_timepoints)"""
    # traces should be 2d array with shape (neurons, nr_timepoints)

    spsig = SPSIG(file_path)
    traces = spsig.sigCorrected
    frame_rate = spsig.freq
    del spsig # free up memory ?
    if traces.shape[1] < traces.shape[0]:
       traces = traces.T # to match (neurons, nr_timepoints) expected shape
    
    return traces, frame_rate

def get_SPSIG_files(root : bool = False) -> list:
    if not root:
        root = filedialog.askdirectory()
    spsigs = '**/*_SPSIG.mat'
    
    SPSIG_files = []
    
    for spsig_path in glob.glob(os.path.join(root,spsigs), recursive = True):
        # Regex breakdown:
        # .*/                => match any characters ending with a slash
        # (g[12])            => capture group: "g1" or "g2"
        # /                  => literal slash
        # ([^/]+)            => capture "Name" (any characters except slash)
        # /                  => literal slash
        # (\d{8})            => capture "date" in the format YYYYMMDD (8 digits)
        # /                  => literal slash
        # (Bar_Tone_LR(?:2)?) => capture "Bar_Tone_LR" optionally followed by a 2
        # /.*                => followed by a slash and the rest of the path
        group_name_date = r'.*/(g[12])/([^/]+)/(\d{8})/(Bar_Tone_LR(?:2)?)/.*'
        re_match = re.match(group_name_date, spsig_path)
        if re_match is None:
            continue
        
        # Add spsig file to extract spikes from
        assert re_match is not None, f'something wrong with: {spsig_path}'
        SPSIG_files.append(spsig_path)
    
    return SPSIG_files

def progress_bar(current_iteration: int,
                 total_iterations: int,
                 character: str = '🍎'):
    bar_length = 50
    filled_length = round(bar_length * current_iteration / total_iterations)
    # Build the progress bar
    bar = character * filled_length
    no_bar = ' -' * (bar_length - filled_length)
    progress = round((current_iteration / total_iterations) * 100)
    print(bar + no_bar, f'{progress} %', end='\r')        



if __name__ == '__main__':
    if 'Demo scripts' in os.getcwd():
        sys.path.append( os.path.abspath('..') ) # add parent directory to path for imports
        os.chdir('..')  # change to main directory
        print('Current working directory: {}'.format( os.getcwd() ))
    
    SPSIG_files = get_SPSIG_files()

    # align all files
    total_iter = len(SPSIG_files)
    print('Found {} ..._SPSIG.mat files, starting to extract spikes!'.format(total_iter))

    for i, file in enumerate(SPSIG_files):
        main(file)
        print('Spike estimation from {} completed!'.format(file))
        progress_bar(i, total_iter)