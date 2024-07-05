from qm.qua import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from analysis.find_ZZ_free_analysis import real_detune_X_flux, try_fit_ramsey

def plot_ZZfree(data, ax1=None, ax2=None):
    """
    data in shape (2,2,M,N)
    first 2 is postive and negative
    second 2 is postive and negative
    M is flux point
    N is evo_time_point
    """
    if ax1 is None or ax2 is None:
        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))

    flux, freq_X, freq_I = real_detune_X_flux(data)
    ZZfree_flux = flux[np.argmin(abs(freq_X - freq_I))]
    
    # Plot freq_X and freq_I on the first subplot
    ax1.set_xlabel("flux [V]")
    ax1.set_ylabel("Frequency [MHz]")
    ax1.plot(flux, freq_X, label="w/ X")
    ax1.plot(flux, freq_I, label="w/o X")
    ax1.legend()
    
    # Plot the difference freq_X - freq_I on the second subplot
    ax2.set_xlabel("flux [mV]")
    ax2.set_ylabel("Difference")
    ax2.plot(flux, freq_X - freq_I, label="diff")
    ax2.text(0.07, 0.9, f"min diff at : {ZZfree_flux:.3f}", fontsize=10, transform=ax2.transAxes)
    ax2.legend()
    
    plt.tight_layout()
    
    return ZZfree_flux

def plot_tau_X_flux(data):
    """
    Plot tau X flux data.
    
    Parameters:
    data (xarray.Dataset): Data in shape (2, 2, M, N)
        - The first 2 corresponds to positive and negative.
        - The second 2 corresponds to positive and negative.
        - M is the flux point.
        - N is the evolutionary time point.
    """
    flux = data.coords["flux"].values
    time = data.coords["time"].values
    q = list(data.data_vars.keys())[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Tau X Flux Data')

    labels = [["X-Positive", "X-Negative"],
              ["I-Positive", "I-Negative"]]
    
    for i in range(2):
        for j in range(2):
            ax = axes[i, j]
            c = ax.pcolormesh(flux, time, data[q][0, i, j, :, :].T, cmap='RdBu', shading='auto')
            ax.set_title(labels[i][j])
            ax.set_xlabel('Flux [V]')
            ax.set_ylabel('Evolutionary Time [ns]')
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

def plot_pureZZ(data):
    """
    Plot tau X flux data.
    
    Parameters:
    data (xarray.Dataset): Data in shape (M, N)
        - M is the flux point.
        - N is the evolutionary time point.
    """
    flux = data.coords["flux"].values
    time = data.coords["time"].values
    q = list(data.data_vars.keys())[0]
    Crosstalk = np.zeros(len(flux))
    for i in range(len(Crosstalk)):
        Crosstalk[i] = try_fit_ramsey(data[q][0, i, :])
    ZZfree_flux = flux[np.argmin(abs(Crosstalk))]

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(8, 10))
    
    c = ax1.pcolormesh(flux, time, data[q][0, :, :].T, cmap='RdBu', shading='auto')
    ax1.set_title("Tau X Flux")
    ax1.set_xlabel('Flux [V]')
    ax1.set_ylabel('Evolutionary Time [ns]')

    ax2.plot(flux, Crosstalk, label="Crosstalk")
    ax1.set_title("Crosstalk X Flux")
    ax2.set_xlabel("flux [mV]")
    ax2.set_ylabel("Crosstalk [MHz]")
    ax2.text(0.07, 0.9, f"min diff at : {ZZfree_flux:.3f}", fontsize=10, transform=ax2.transAxes)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()