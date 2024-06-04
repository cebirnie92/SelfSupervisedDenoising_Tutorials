"""
Helper Functions for Self-Supervised Seismic Denoising Tutorial Series
======================================================================

This module contains a set of helper functions focused on noise generation, data preparation, 
and ensuring reproducible experiments. These functions are essential components of the 
self-supervised denoising tutorial series, which focuses on seismic denoising using 
blind-mask methodologies.

These helper functions are intended to simplify the tutorial process and do not 
require any modification or deep understanding. They are designed to work out-of-the-box 
with the accompanying tutorial series, enabling you to focus on learning and 
implementing the core concepts of self-supervised denoising.

Contents:
---------
- regular_patching_2D: Regularly samples and extracts patches from a 2D array.
- add_whitegaussian_noise: Adds white Gaussian noise to data patches.
- add_bandlimited_noise: Adds bandlimited noise to data patches.
- add_trace_wise_noise: Adds trace-wise noise to data patches.
- butter_bandpass: Creates a bandpass filter.
- butter_bandpass_filter: Applies a bandpass filter to a trace.
- array_bp: Applies a bandpass filter to an array of traces.
- band_limited_noise: Generates bandlimited noise.
- set_seed: Sets random seeds for reproducibility.
- weights_init: Initializes weights of a neural network.
- make_data_loader: Creates data loaders for training and validation of a blind-spot neural network.
- num_active_pixs: Compute the number of active pixels in a patch.

Note: These functions are provided as-is and do not require any modification or deep understanding.
"""


import numpy as np
import random
import itertools
from scipy.signal import filtfilt, butter
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader 


def regular_patching_2D(data, 
                        patchsize=[64, 64], 
                        step=[16, 16], 
                        verbose=True):
    """ Regular sample and extract patches from a 2D array
    :param data: np.array [y,x]
    :param patchsize: tuple [y,x]
    :param step: tuple [y,x]
    :param verbose: boolean
    :return: np.array [patch#, y, x]
    """

    # find starting indices
    x_start_indices = np.arange(0, data.shape[0] - patchsize[0], step=step[0])
    y_start_indices = np.arange(0, data.shape[1] - patchsize[1], step=step[1])
    starting_indices = list(itertools.product(x_start_indices, y_start_indices))

    if verbose:
        print('Extracting %i patches' % len(starting_indices))

    patches = np.zeros([len(starting_indices), patchsize[0], patchsize[1]])

    for i, pi in enumerate(starting_indices):
        patches[i] = data[pi[0]:pi[0]+patchsize[0], pi[1]:pi[1]+patchsize[1]]

    return patches


def add_whitegaussian_noise(d, sc=0.5):
    """ Add white gaussian noise to data patch
    
    Parameters
    ----------
    d: np.array [y,x]
        Data to add noise to
    sc: float 
        noise scaling value
        
    Returns
    -------
        d+n: np.array 
            Created noisy data
        n: np.array 
            Additive noise        
    """
    
    n = np.random.normal(size=d.shape)

    return d + (n * sc), n


def add_bandlimited_noise(d, lc=2, hc=80, sc=0.5):
    """ Add bandlimited noise to data patch
    
    Parameters
    ----------
    d: np.array [y,x]
        Data to add noise to
    lc: float 
        Low cut for bandpass
    hc: float 
        High cut for bandpass
    sc: float 
        Noise scaling value
        
    Returns
    -------
        d+n: np.array 
            Created noisy data
        n: np.array 
            Additive noise        
    """
    n = band_limited_noise(size=d.shape, lowcut=lc, highcut=hc)

    return d + (n * sc), n


def add_trace_wise_noise(d,
                         num_noisy_traces,
                         noisy_trace_value,
                         num_realisations,
                        ):  
    """ Add trace-wise noise to data patch
    
    Parameters
    ----------
    d: np.array [shot,y,x]
        Data to add noise to
    num_noisy_traces: int 
        Number of noisy traces to add to shots
    noisy_trace_value: int 
        Value of noisy traces
    num_realisations: int 
        Number of repeated applications per shot
        
    Returns
    -------
        alldata: np.array 
            Created noisy data
    """
    
    alldata=[]
    for k in range(len(d)):        
        clean=d[k]    
        data=np.ones([num_realisations,d.shape[1],d.shape[2]])
        for i in range(len(data)):    
            corr = np.random.randint(0,d.shape[2], num_noisy_traces) 
            data[i] = clean.copy()
            data[i,:,corr] = np.ones([1,d.shape[1]])*noisy_trace_value
        alldata.append(data)
        
    alldata=np.array(alldata) 
    alldata=alldata.reshape(num_realisations*d.shape[0],d.shape[1],d.shape[2])
    print(alldata.shape)

    return alldata


def butter_bandpass(lowcut, highcut, fs, order=5):
    """ Bandpass filter
    
    Parameters
    ----------
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
    order: int 
        Filter order
        
    Returns
    -------
        b : np.array 
            The numerator coefficient vector of the filter
        a : np.array 
            The denominator coefficient vector of the filter
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to trace
    
    Parameters
    ----------
    data: np.array [1D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
    order: int 
        Filter order
        
    Returns
    -------
        y : np.array 
            Bandpassed data
    """
    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def array_bp(data, lowcut, highcut, fs, order=5):
    """ Apply bandpass filter to array of traces
    
    Parameters
    ----------
    data: np.array [2D]
        Data onto which to apply bp filter
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
    order: int 
        Filter order
        
    Returns
    -------
        bp : np.array [2D]
            Bandpassed data
    """
    bp = np.vstack([butter_bandpass_filter(data[:, ix], lowcut, highcut, fs, order)
                    for ix in range(data.shape[1])])

    return bp


def band_limited_noise(size, lowcut, highcut, fs=250):
    """ Generate bandlimited noise
    
    Parameters
    ----------
    size: tuple 
        Size of array on which to create the noise
    lowcut: int
        Low cut for bandpass
    highcut: int 
        High cut for bandpass
    fs: int 
        Sampling frequency
        
    Returns
    -------
        bpnoise : np.array 
            Bandpassed noise
    """

    basenoise = np.random.normal(size=size)
    # Pad top and bottom due to filter effects
    basenoise_pad = np.vstack([np.zeros([50, size[1]]), basenoise, np.zeros([50, size[1]])])
    # Bandpass base noise
    bpnoise =  array_bp(basenoise_pad, lowcut, highcut, fs, order=5)[:,50:-50]

    return bpnoise.T

def set_seed(seed):
    """
    Use this to set ALL the random seeds to a fixed value and take out any randomness from cuda kernels
    
    Parameters
    ----------
    seed: int 
        Integer to be used for the seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

    return True


def weights_init(m):
    """Initialise weights of NN
    
    Parameters
    ----------
    m: torch.model 
        NN
    """
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight)
        nn.init.constant_(m.bias, 0)
        

def make_data_loader(noisy_patches, 
                     corrupted_patches, 
                     masks, 
                     n_training,
                     n_test,
                     batch_size,
                     torch_generator
                    ):
    """Make data loader to be used for the training and validation of a blind-spot NN
    
    Parameters
    ----------
    noisy_patches: np.array
        Patches of noisy data to be network target
    corrupted_patches: np.array
        Patches of processed noisy data to be network input
    masks: np.array
        Masks corresponding to corrupted_patches, indicating location of active pixels
    n_training: int
        Number of samples to be used for training
    n_test: int
        Number of samples to be used for validation
    batch_size: int
        Size of data batches to be used during training
    torch_generator: torch.generator
        For reproducibility of data loader
        
    Returns
    -------
        train_loader : torch.DataLoader
            Training data separated by batch
        test_loader : torch.DataLoader
            Validation data separated by batch
    """
    
    # Define Train Set
    # Remember to add 1 to 2nd dim - Pytorch is [#data, #channels, height, width]
    train_X = np.expand_dims(corrupted_patches[:n_training],axis=1)
    train_y = np.expand_dims(noisy_patches[:n_training],axis=1)    
    msk = np.expand_dims(masks[:n_training],axis=1)   
    # Convert to torch tensors and make TensorDataset
    train_dataset = TensorDataset(torch.from_numpy(train_X).float(), 
                                  torch.from_numpy(train_y).float(), 
                                  torch.from_numpy(msk).float(),)

    # Define Test Set
    test_X = np.expand_dims(corrupted_patches[n_training:n_training+n_test],axis=1)
    test_y = np.expand_dims(noisy_patches[n_training:n_training+n_test],axis=1)
    msk = np.expand_dims(masks[n_training:n_training+n_test],axis=1) 
    test_dataset = TensorDataset(torch.from_numpy(test_X).float(), 
                                 torch.from_numpy(test_y).float(), 
                                 torch.from_numpy(msk).float(),)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=torch_generator)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

def num_active_pixs(patchshape, percent_active=5, verbose=True):
    '''Compute the number of active pixels in a patch.

    Parameters:
    -----------
    patchshape : tuple
        Shape of the patch (height, width).
    percent_active : float, optional
        Percentage of active pixels to be selected within the patch. Default is 5.
    verbose : bool, optional
        If True, prints the number of active pixels selected. Default is True.
    
    Returns:
    --------
    num_activepixels : int
        Number of active pixels to be selected within the patch.
    '''
    # Compute the total number of pixels within a patch
    total_num_pixels = patchshape[0]*patchshape[1]
    # Compute the number that should be active pixels based on the choosen percentage
    num_activepixels = int(np.floor((total_num_pixels/100) * percent_active))
    if verbose: print("Number of active pixels selected: \n %.2f percent equals %i pixels"%(percent_active,num_activepixels))
    return int(num_activepixels)


def patch_selection(patches, co_percentile=25, verbose=True):
    """
    Selects patches based on the sum of squares along the last two axes.
    
    Parameters:
    -----------
    patches : numpy.ndarray
        Input numpy array containing patches.
    co_percentile : int, optional
        The percentile threshold for selecting patches (default is 25).
    verbose : bool, optional
        Flag to print information about the number of patches removed and remaining (default is True).
    
    Returns:
    --------
        selected_patches : numpy.ndarray 
            Numpy array with selected patches above the percentile threshold.
    """
    # Sum along the last two axes
    sum_patches = np.mean(patches**2, axis=(1, 2))
    
    # Find the indices of the bottom 10% of values
    threshold = np.percentile(sum_patches, co_percentile)
    rm_indices = np.where(sum_patches < threshold)[0]
    
    # Remove the indices from the original numpy array
    selected_patches = np.delete(patches, rm_indices, axis=0)
    if verbose: print('%i patches removed, %i patches remaining'%(len(rm_indices), len(selected_patches)))
    
    return selected_patches
