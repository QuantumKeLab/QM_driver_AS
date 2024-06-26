from qm.qua import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
import numpy as np
from analysis.find_ZZ_free_analysis import real_detune_X_flux

def plot_ZZfree(data, ax=None ):
    """
    data in shape (2,2,M,N)
    first 2 is postive and negative
    second 2 is postive and negative
    M is flux point
    N is evo_time_point
    """
    if ax == None:
        fig, ax = plt.subplots()

    flux, freq_X, freq_I = real_detune_X_flux(data)
    ZZfree_flux = flux[np.argmin(abs(freq_X-freq_I))]
    ax.set_xlabel("flux [mV]")

    ax.plot(flux, freq_X, label=f"w/ X")
    ax.plot(flux, freq_I, label=f"w/o X")
    ax.plot(flux, freq_X-freq_I, label=f"diff")
    ax.text(0.07, 0.9, f"min diff at : {ZZfree_flux:.3f}", fontsize=10, transform=ax.transAxes)

    ax.legend()
    plt.tight_layout()
    
    return ZZfree_flux