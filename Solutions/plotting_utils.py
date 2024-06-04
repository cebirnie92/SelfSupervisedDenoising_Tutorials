"""
Plotting Functions for Self-Supervised Seismic Denoising Tutorial Series
========================================================================

This module contains a set of plotting functions designed to help students focus on the 
tutorial content without spending time on writing basic visualization routines. These functions 
are essential for visualizing the different steps and results in the self-supervised denoising 
tutorial series, which focuses on seismic denoising using blind-mask methodologies.

Contents:
---------
- plot_corruption: Plots the original, corrupted, and corruption mask data patches.
- plot_training_metrics: Plots the training and validation accuracy and loss over epochs.
- plot_synth_results: Plots the clean, noisy, and denoised data patches alongside the noise removed.
- plot_field_results: Plots the noisy and denoised data patches for field results where no clean data is available.

Note: These functions are provided as-is and do not require any modification or deep understanding.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_corruption(noisy, 
                    crpt, 
                    mask, 
                    seismic_cmap='RdBu', 
                    vmin=-0.25, 
                    vmax=0.25):
    """Plotting function of N2V pre-processing step
    
    Parameters
    ----------
    noisy: np.array
        Noisy data patch
    crpt: np.array
        Pre-processed data patch
    mask: np.array
        Mask corresponding to pre-processed data patch
    seismic_cmap: str
        Colormap for seismic plots
    vmin: float
        Minimum value on colour scale
    vmax: float
        Maximum value on colour scale
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    
    fig,axs = plt.subplots(1,3,figsize=[15,5])
    axs[0].imshow(noisy, cmap=seismic_cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(crpt, cmap=seismic_cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(mask, cmap='binary_r')

    axs[0].set_title('Original')
    axs[1].set_title('Corrupted')
    axs[2].set_title('Corruption Mask')
    
    fig.tight_layout()
    return fig,axs

def plot_training_metrics(train_accuracy_history,
                         test_accuracy_history,
                          train_loss_history,
                          test_loss_history
                         ):
    """Plotting function of N2V training metrics
    
    Parameters
    ----------
    train_accuracy_history: np.array
        Accuracy per epoch throughout training
    test_accuracy_history: np.array
        Accuracy per epoch throughout validation
    train_loss_history: np.array
        Loss per epoch throughout training
    test_accuracy_history: np.array
        Loss per epoch throughout validation
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    fig,axs = plt.subplots(1,2,figsize=(15,4))
    
    axs[0].plot(train_accuracy_history, 'r', lw=2, label='train')
    axs[0].plot(test_accuracy_history, 'k', lw=2, label='validation')
    axs[0].set_title('RMSE', size=16)
    axs[0].set_ylabel('RMSE', size=12)

    axs[1].plot(train_loss_history, 'r', lw=2, label='train')
    axs[1].plot(test_loss_history, 'k', lw=2, label='validation')
    axs[1].set_title('Loss', size=16)
    axs[1].set_ylabel('Loss', size=12)
    
    for ax in axs:
        ax.legend()
        ax.set_xlabel('# Epochs', size=12)
    fig.tight_layout()
    return fig,axs


def plot_synth_results(clean, 
                       noisy, 
                       denoised,
                       cmap='RdBu', 
                       vmin=-0.25, 
                       vmax=0.25):
    """Plotting function of synthetic results from denoising
    
    Parameters
    ----------
    clean: np.array
        Clean data patch
    noisy: np.array
        Noisy data patch
    denoised: np.array
        Denoised data patch
    cmap: str
        Colormap for plots
    vmin: float
        Minimum value on colour scale
    vmax: float
        Maximum value on colour scale
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    
    fig,axs = plt.subplots(1,4,figsize=[15,4])
    axs[0].imshow(clean, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(noisy, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[2].imshow(denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[3].imshow(noisy-denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    axs[0].set_title('Clean')
    axs[1].set_title('Noisy')
    axs[2].set_title('Denoised')
    axs[3].set_title('Noise Removed')

    fig.tight_layout()
    return fig,axs

def plot_field_results(noisy, 
                       denoised,
                       cmap='RdBu', 
                       vmin=-0.25, 
                       vmax=0.25):
    """Plotting function of field results from denoising, i.e., where no clean is available
    
    Parameters
    ----------
    noisy: np.array
        Noisy data patch
    denoised: np.array
        Denoised data patch
    cmap: str
        Colormap for plots
    vmin: float
        Minimum value on colour scale
    vmax: float
        Maximum value on colour scale
        
    Returns
    -------
        fig : pyplot.figure
            Figure object
        axs : pyplot.axs
            Axes of figure
    """
    
    fig,axs = plt.subplots(1,2,figsize=[15,8])
    axs[0].imshow(noisy, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
    axs[1].imshow(denoised, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    axs[0].set_title('Noisy')
    axs[1].set_title('Denoised')

    fig.tight_layout()
    return fig,axs