import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks

def get_acfs(cell, nlags, burn_in=.2):
    """
    Compute the ACF for the three main variables in the model.

    burn_in is the proportion of timepoints to be thrown out
    """
    T = len(cell.RB_hist)
    ti = round(burn_in*T)
    acf_RB = acf(cell.RB_hist[ti:], fft=False, nlags=nlags)
    acf_M = acf(cell.M_hist[ti:], fft=False, nlags=nlags)
    acf_RBc = acf(cell.RB_c_hist[ti:], fft=False, nlags=nlags)
    return (acf_RB, acf_M, acf_RBc)

def is_periodic(acf_, prominence=0, th=.1):
    """
    Determine if the signal is periodic based on its ACF

    """
    
    peaks, _ = find_peaks(acf_, height=None, prominence=prominence)
    dist_peaks = [peaks[i+1]-peaks[i] for i in range(len(peaks)-1)]

    periodic = np.std(dist_peaks)/np.median(dist_peaks) < th

    if periodic:
        return peaks, np.median(dist_peaks)
    else:
        return peaks, None

def get_cycle_stats(cell):
    """
    """
    crt_cycle=1
    cycle_ids = [crt_cycle]
    stats = dict()
    stats["RB_avg"], stats["RBc_avg"], stats["RB_delta"], stats["RBc_delta"] = [], [], [], []

    for k in range(1, len(cell.phase_hist)):

        if cell.phase_hist[k]=="G1" and cell.phase_hist[k-1]=="G2":
            crt_cycle+=1

        cycle_ids.append(crt_cycle)

    cycle_ids = np.array(cycle_ids)

    for cycle in range(1, crt_cycle):

        stats["RB_avg"].append(np.mean(np.array(cell.RB_hist)[cycle_ids==cycle]))
        stats["RBc_avg"].append(np.mean(np.array(cell.RB_c_hist)[cycle_ids==cycle]))
        stats["RB_delta"].append(
            np.max(np.array(cell.RB_hist)[cycle_ids==cycle])- np.min(np.array(cell.RB_hist)[cycle_ids==cycle]))
        stats["RBc_delta"].append(
            np.max(np.array(cell.RB_c_hist)[cycle_ids==cycle])- np.min(np.array(cell.RB_c_hist)[cycle_ids==cycle]))

    return stats