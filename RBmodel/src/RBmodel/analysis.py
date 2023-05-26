import numpy as np
from statsmodels.tsa.stattools import acf
from scipy.signal import find_peaks
from sklearn.linear_model import LinearRegression


def get_acfs(cell, nlags, burn_in=.2):
    """
    Compute the ACF for the three main variables in the model.

    Parameters:
    ----------
    cell: cell_models.cell
        a cell object with some history
    nlags: int
        size of lag to use in ACF computation
    burn_in: float
        proportion of time points to be thrown out
    """

    T = len(cell.RB_hist)
    ti = round(burn_in*T)
    acf_RB = acf(cell.RB_hist[ti:], fft=False, nlags=nlags)
    acf_M = acf(cell.M_hist[ti:], fft=False, nlags=nlags)
    acf_RBc = acf(cell.RB_c_hist[ti:], fft=False, nlags=nlags)

    return (acf_RB, acf_M, acf_RBc)


def is_periodic(acf_, prominence=0, th=.1):
    """
    Determine if the signal is periodic based on its ACF.

    Parameters:
    ----------
    acf_: np.array
        autocorrelation function
    prominence: float
        required prominence of peaks
    th: float
        threshold on the CV of peaks relative distances
    """

    peaks, _ = find_peaks(acf_, height=None, prominence=prominence)
    dist_peaks = [peaks[i+1]-peaks[i] for i in range(len(peaks)-1)]

    # signal is periodic if the distance between peaks is always the same
    periodic = np.std(dist_peaks)/np.median(dist_peaks) < th

    if periodic:
        return peaks, np.median(dist_peaks)
    else:
        return peaks, None


def get_cycle_stats(cell):
    """
    Compute cell cycle statistics. 

    Returns dict with relevant statistics of the cell cycle.

    Parameters:
    ----------
    cell: cell_models.cell
        cell object with a given history
    """

    crt_cycle = 1
    cycle_ids = [crt_cycle]
    stats = dict()
    stats["RB_avg"], stats["RBc_avg"], stats["RB_delta"], stats["RBc_delta"] = [
    ], [], [], []

    for k in range(1, len(cell.phase_hist)):

        if cell.phase_hist[k] == "G1" and cell.phase_hist[k-1] == "G2":
            crt_cycle += 1

        cycle_ids.append(crt_cycle)

    cycle_ids = np.array(cycle_ids)

    for cycle in range(1, crt_cycle):

        stats["RB_avg"].append(
            np.mean(np.array(cell.RB_hist)[cycle_ids == cycle]))
        stats["RBc_avg"].append(
            np.mean(np.array(cell.RB_c_hist)[cycle_ids == cycle]))
        stats["RB_delta"].append(
            np.max(np.array(cell.RB_hist)[cycle_ids == cycle]) - np.min(np.array(cell.RB_hist)[cycle_ids == cycle]))
        stats["RBc_delta"].append(
            np.max(np.array(cell.RB_c_hist)[cycle_ids == cycle]) - np.min(np.array(cell.RB_c_hist)[cycle_ids == cycle]))

    return stats


def get_phase_durations(cell):
    """
    Maps the phase history to the phase durations.

    Returns phase duration and statistics of the cell cycle.

    Parameters:
    ----------
    cell: cell_models.cell
        a cell object with a given history
    """

    phase_vec = np.array(cell.phase_hist) == "G1"

    # convert phase vec into a relative duration
    # if it is true: then it is in G1: it should be negative numbers growing until the zero at the transition
    # it it is false: then it is in G2: it should be positive numbers growing past the transition

    G1_growth, G2_growth = [], []
    G1_length, G2_length = [], []
    G1_mean_RB, G2_Delta_RB = [], []
    M_births = []
    RB_birth, RB_division = [], []
    RBc_birth, RBc_division = [], []

    phase_duration = np.zeros(len(phase_vec))

    # you probably don't have a complete first cycle
    # so we try to find the first switch
    k = 0
    burn = True
    while k < len(phase_vec):

        # find the next index such that the phase changes
        try:
            next_switch = np.where((phase_vec[k:] != phase_vec[k]))[0][0]

            # we are burning the first few phases
            if burn:
                if phase_vec[k+next_switch] == 1:
                    burn = False
                k = k + next_switch
                continue

            if phase_vec[k] == 1:  # G1
                phase_duration[k:k +
                            next_switch] = np.linspace(-next_switch+1, 0, next_switch)
                G1_growth.append(cell.M_hist[k+next_switch-1]/cell.M_hist[k])
                G1_length.append(next_switch*cell.dt)
                G1_mean_RB.append(np.mean(cell.RB_hist[k:k+next_switch]))
                M_births.append(cell.M_hist[k])
                RB_birth.append(cell.RB_hist[k])
                RB_division.append(cell.RB_hist[k+next_switch-1])
                RBc_birth.append(cell.RB_c_hist[k])
                RBc_division.append(cell.RB_c_hist[k+next_switch-1])

            elif phase_vec[k] == 0:  # G2
                phase_duration[k:k +
                            next_switch] = np.linspace(0, next_switch-1, next_switch)
                G2_growth.append(cell.M_hist[k+next_switch-1]/cell.M_hist[k])
                G2_length.append(next_switch*cell.dt)
                G2_Delta_RB.append(cell.RB_hist[k+next_switch-1] - cell.RB_hist[k])

            k = k+next_switch

        except:
            # next_switch = len(phase_vec) - k
            if phase_vec[k] == 1: # G1, we are good because we have not added the elements yet
                pass
            elif phase_vec[k] == 0:
                M_births.pop(-1)
                G1_growth.pop(-1)
                G1_length.pop(-1)
                G1_mean_RB.pop(-1)
                RB_birth.pop(-1)
                RB_division.pop(-1)
                RBc_birth.pop(-1)
                RBc_division.pop(-1)

            break

    def _compute_delta(M_births, G_growth):
        """
        Transform relative (ratio) growth to Delta growth
        """
        return (np.array(G_growth)-1) * np.array(M_births)

    stats = {
        'birth': M_births,
        'growth': (G1_growth, G2_growth),
        'length': (G1_length, G2_length),
        'RB': (G1_mean_RB, G2_Delta_RB),
        'RB_G1': (RB_birth, RB_division),
        'RBc_G1': (RBc_birth, RBc_division),
        'delta': (_compute_delta(M_births, G1_growth),  _compute_delta(M_births, G2_growth))
    }

    return phase_duration * cell.dt, stats


def get_mean_stats(stats):
    """
    Extracts medians from the statistics for a given cell.

    Parameters:
    ----------
    stats: dict
        dict containing the stats for each metric of interest, as outputted by get_phase_durations()
    """

    stats_dict = dict()
    stats_dict["G1_growth"] = np.median(stats['growth'][0])
    stats_dict["G2_growth"] = np.median(stats['growth'][1])
    stats_dict["G1_length"] = np.median(stats['length'][0])
    stats_dict["G2_length"] = np.median(stats['length'][1])
    stats_dict["DeltaRB/meanRB"] = np.mean(stats['RB'][1]) / \
        np.mean(stats['RB'][0])

    return stats_dict

def compute_slopes(stats):
    """
    Compute the slopes as a function of M birth
    """
    slopes = dict()
    
    X = np.array(stats["birth"]).reshape(-1, 1)
    ph_nm = ["G1", "G2"]
    
    for key in stats:
        
        if key == "birth":
            continue
            
        for (k, ph) in enumerate(ph_nm):
           
            # each stat is represented (for now) as a tuple with
            # the first element being G1, the second G2
            y = stats[key][k]
            reg = LinearRegression().fit(X, y)
            
            key_nm = f"slopes_{ph}_{key}"
            
            slopes[key_nm] = reg.coef_[0]
        
    return slopes
    
