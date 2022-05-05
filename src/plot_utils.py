import numpy as np
import matplotlib.pyplot as plt
from src.analysis import get_acfs, is_periodic


def plot_vars_vs_time(ax, cell):
    """
    """
    ax.plot(np.array(cell.M_hist)/cell.M_hist[0], label="M")
    ax.plot(np.array(cell.RB_hist)/cell.RB_hist[0], label="RB")
    ax.plot(np.array(cell.RB_c_hist)/cell.RB_c_hist[0], label="[RB]")

    
    phase_vec = np.array(cell.phase_hist)=="G1"
    x_min = 0
    x_max = len(phase_vec)

    phase=0
    for k in range(len(phase_vec)):
        if (phase_vec[k] == True) and phase==0:
            x_min = k
            phase=1
        elif (phase_vec[k] == False) and phase==1:
            x_max=k-1
            phase=0
            ax.axvspan(x_min, x_max, color = 'lightgray', alpha=.3)

    if x_min > x_max:
        ax.axvspan(x_min, len(phase_vec), color = 'lightgray', alpha=.3)
    
    ax.set_yscale('log')
    ax.legend(loc="upper left")
    ax.grid()
    ax.set_xlabel("Time")
    ax.set_ylabel("Variables")

def plot_phase_RB(ax, cell):
    """
    Phase space plot
    """
    phase_vec = np.array(cell.phase_hist)=="G1"
    # ax.plot(cell.M_hist, cell.RB_hist, alpha=.5)
    ax.scatter(
        np.array(cell.M_hist)[phase_vec==True], 
        np.array(cell.RB_hist)[phase_vec==True], 
        alpha=.3, s=1, label="G1"
    )
    ax.scatter(
        np.array(cell.M_hist)[phase_vec==False], 
        np.array(cell.RB_hist)[phase_vec==False], 
        alpha=.3, s=1, label="G2"
    )
#     ax.axhline(cell.RB_division, color='red')
#     m_vec = np.linspace(0, 2)
#     ax.plot(m_vec, m_vec*cell.RB_transition, color='red')
    ax.set_xlabel("M")
    ax.set_ylabel("RB amount")
    ax.grid()
    ax.legend()

    if cell.division=="timer":
        pass # no line for a timer
    if cell.transition=="size":
        ax.axvline(cell.transition_th, color="red")

def plot_phase_RBc(ax, cell):
    """
    """
    phase_vec = np.array(cell.phase_hist)=="G1"
    ax.scatter(
        np.array(cell.M_hist)[phase_vec==True], 
        np.array(cell.RB_c_hist)[phase_vec==True], 
        alpha=.3, s=1, label="G1"
    )
    ax.scatter(
        np.array(cell.M_hist)[phase_vec==False], 
        np.array(cell.RB_c_hist)[phase_vec==False], 
        alpha=.3, s=1, label="G2"
    )
    ax.grid()
    ax.set_xlabel("M")
    ax.set_ylabel("[RB]")
    ax.legend()

    if cell.division=="timer":
        pass # no line for a timer
    if cell.transition=="size":
        ax.axvline(cell.transition_th, color="red")

    return


def plot_autocorrelations(ax, cell, nlags=4000, prominence=.1):
    """
    """

    (acf_RB, acf_M, acf_RBc) = get_acfs(cell, nlags) 
    labels = ["RB", "M", "[RB]"]
    periods=[]
    for i, acf_ in enumerate([acf_RB, acf_M, acf_RBc]):
        peaks, period = is_periodic(acf_) 
        ax.plot(acf_, label=labels[i])
        ax.plot(peaks, acf_[peaks], "x", color="grey", ms=10, linewidth=3)
        periods.append(period)
        
    ax.grid()
    ax.set_xlabel("Lag")
    ax.set_ylabel("ACF")
    ax.legend()
    return periods