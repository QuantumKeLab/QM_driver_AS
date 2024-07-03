from qualang_tools.plot.fitting import Fit
import numpy as np

def try_fit_ramsey(data):
    """
    data in shape (N)
    N is evo_time_point
    """
    evo_time = data.coords["time"].values
    try:
        fit = Fit()
        ana_dict = fit.ramsey(evo_time, data, plot=False)
        detune = ana_dict['f'][0]*1e3
    except:
        print("an error occured when fit ramsey")
        detune = 0.
    return detune

def calculate_real_detune(data):
    """
    data in shape (2,N)
    2 is postive and negative
    N is evo_time_point
    """
    freq_pos = try_fit_ramsey(data[0])
    freq_neg = try_fit_ramsey(data[1])
    real_detune = (freq_pos-freq_neg)/2.
    return real_detune

def real_detune_X_flux(data):
    """
    data in shape (2,2,M,N)
    first 2 is postive and negative
    second 2 is postive and negative
    M is flux point
    N is evo_time_point
    """
    flux_range = data.coords["flux"].values
    q = list(data.data_vars.keys())[0]
    frequency_X=np.zeros(len(flux_range))
    frequency_I=np.zeros(len(flux_range))
    for flux_idx, flux in enumerate(flux_range):
        frequency_I[flux_idx] = calculate_real_detune(data[q][0, 1, :, flux_idx, :])
        frequency_X[flux_idx] = calculate_real_detune(data[q][0, 0, :, flux_idx, :])
    return flux_range, frequency_X, frequency_I