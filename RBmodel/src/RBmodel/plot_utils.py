import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

from RBmodel.analysis import get_acfs, is_periodic
from RBmodel import cell_models
from RBmodel import load_utils


def plot_vars_vs_time(ax, cell, collapse_time=False):
    """
    """

    data_norm = pd.DataFrame(columns=["M", "RB", "[RB]"])
    data_norm ["M"] = np.array(cell.M_hist)/cell.M_hist[-1]
    data_norm["RB"] = np.array(cell.RB_hist)/cell.RB_hist[-1]
    data_norm["[RB]"] = np.array(cell.RB_c_hist)/cell.RB_c_hist[-1]

    phase_durations, _ = load_utils.get_phase_durations(cell)

    if collapse_time:
        ms=1
        alpha = .3
        ax.scatter(phase_durations, data_norm["M"], label="M", s=ms, alpha=alpha)
        ax.scatter(phase_durations, data_norm["RB"], label="RB", s=ms, alpha=alpha)
        ax.scatter(phase_durations, data_norm["[RB]"], label="[RB]", s=ms, alpha=alpha)

        ax.axvline(0, color="red")
        ax.grid()
        ax.set_xlabel("Time [hr]")
        ax.set_ylabel("Variables")
        ax.legend(loc="upper left")

    else:
        ax.plot(data_norm["M"], label="M")
        ax.plot(data_norm["RB"], '--', label="RB")
        ax.plot(data_norm["[RB]"], label="[RB]")

        phase_vec = np.array(cell.phase_hist) == "G1"
        x_min = 0
        x_max = len(phase_vec)

        phase = phase_vec[0]
        for k in range(len(phase_vec)):
            if (phase_vec[k] == True) and phase == 0:
                x_min = k
                phase = 1
            elif (phase_vec[k] == False) and phase == 1:
                x_max = k-1
                phase = 0
                ax.axvspan(x_min, x_max, color='lightgray', alpha=.3)

        if x_min > x_max:
            ax.axvspan(x_min, len(phase_vec), color='lightgray', alpha=.3)

        ax.set_yscale('log')
        ax.legend(loc="upper left")
        # ax.grid()
        ax.set_xlabel("Time")
        ax.set_ylabel("Variables")

        ax.set_ylim([data_norm.quantile(0.1).min()*0.8, data_norm.quantile(0.9).max()*1.5])


def plot_phase_RB(ax, cell):
    """
    Phase space plot
    """
    phase_vec = np.array(cell.phase_hist) == "G1"

    ax.scatter(
        np.array(cell.M_hist)[phase_vec == True],
        np.array(cell.RB_hist)[phase_vec == True],
        alpha=.3, s=1, label="G1"
    )
    ax.scatter(
        np.array(cell.M_hist)[phase_vec == False],
        np.array(cell.RB_hist)[phase_vec == False],
        alpha=.3, s=1, label="G2"
    )

    ax.set_xlabel("M")
    ax.set_ylabel("RB amount")
    ax.grid()
    ax.legend()

    pc_points = .5
    npoints = int(pc_points * len(cell.RB_hist))
    ymin = .9 * np.min(np.array(cell.RB_hist)[npoints:-1])
    ymax = 1.1 * np.max(np.array(cell.RB_hist)[npoints:-1])
    ax.set_ylim([ymin, ymax])

    if cell.division == "timer":
        pass  # no line for a timer
    if cell.transition == "size":
        ax.axvline(cell.transition_th, color="red")

    return


def plot_phase_RBc(ax, cell):
    """
    """
    phase_vec = np.array(cell.phase_hist) == "G1"
    ax.scatter(
        np.array(cell.M_hist)[phase_vec == True],
        np.array(cell.RB_c_hist)[phase_vec == True],
        alpha=.3, s=1, label="G1"
    )
    ax.scatter(
        np.array(cell.M_hist)[phase_vec == False],
        np.array(cell.RB_c_hist)[phase_vec == False],
        alpha=.3, s=1, label="G2"
    )

    pc_points = .5
    npoints = int(pc_points * len(cell.RB_c_hist))
    ymin = .9 * np.min(np.array(cell.RB_c_hist)[npoints:-1])
    ymax = 1.1 * np.max(np.array(cell.RB_c_hist)[npoints:-1])
    ax.set_ylim([ymin, ymax])

    ax.grid()
    ax.set_xlabel("M")
    ax.set_ylabel("[RB]")
    ax.legend()

    if cell.division == "timer":
        pass  # no line for a timer
    if cell.transition == "size":
        ax.axvline(cell.transition_th, color="red")

    return


def plot_autocorrelations(ax, cell, nlags=4000, prominence=.1):
    """
    """

    (acf_RB, acf_M, acf_RBc) = get_acfs(cell, nlags)
    labels = ["RB", "M", "[RB]"]
    periods = []
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


def run_and_plot_test(
    alpha=2, beta0=3, delta=1,
    gamma=.9, epsilon=.01, dt=.1,
    division="timer", transition="size",
    time_SG2=1e-1, transition_th=2., k_trans=5, T=300
):
    """
    Run a model and gets the relevant plots out
    """

    params = {
    'alpha': alpha,
    'beta0': beta0,
    'delta': delta,
    'gamma': gamma,
    'epsilon': epsilon,
    'dt': dt,
    'duration_SG2': time_SG2, # hr
    'transition_th': transition_th,
    'k_trans': k_trans, 
    'division': division,
    'transition': transition
}

    cell = cell_models.cell(params=params)
    cell.grow(T)

    _, ax = plt.subplots(3, 2, figsize=(15, 10))
    # t_vec = np.arange(T+1)

    plot_vars_vs_time(ax[0, 0], cell, collapse_time=False)
    plot_phase_RB(ax[1, 0], cell)
    plot_phase_RBc(ax[1, 1], cell)

    periods = plot_autocorrelations(ax[0, 1], cell, nlags=T/2, prominence=0)
    if any([p is None for p in periods]):
        print("Not periodic")
    else:
        if np.std(periods)/np.mean(periods) < .05:
            print(f"All signals with period approx. {np.mean(periods)}.")
        else:
            print("Not periodic")

    plot_vars_vs_time(ax[2,0], cell, collapse_time=True)
    plot_stats(ax[2, 1], cell)

    return cell, periods


def plot_data(df, G1_th=10):
    """
    Plots the experimental data from .csv or .xlsx files
    """

    _, ax = plt.subplots(3, 3, figsize=(15, 10))

    for i, j in product(range(3), range(3)):

        k = 3*i + j

        if k >= len(df.columns):
            return

        col = df.columns[k]

        idx_G1 = df.index[df.G1_length > G1_th]
        color = "dodgerblue"
        ax[i, j].hist(df.loc[idx_G1, col], density=True, alpha=.5, color=color)
        ax[i, j].axvline(df.loc[idx_G1, col].mean(),
                         label="mean G1+", color=color)
        ax[i, j].axvline(df.loc[idx_G1, col].median(),
                         linestyle="--", label="median G1+", color=color)

        idx_G1 = df.index[df.G1_length <= G1_th]
        color = "darkorange"
        ax[i, j].hist(df.loc[idx_G1, col], density=True, alpha=.5, color=color)
        ax[i, j].axvline(df.loc[idx_G1, col].mean(),
                         label="mean G1-", color=color)
        ax[i, j].axvline(df.loc[idx_G1, col].median(),
                         linestyle="--", label="median G1-", color=color)
        ax[i, j].set_xlabel(col)
        ax[i, j].set_ylabel("Density")
        ax[i, j].grid()
        ax[i, j].legend()

    return

def plot_stats(ax, cell):
    """
    """

    _, stats = load_utils.get_phase_durations(cell)
    mean_stats = load_utils.get_mean_stats(stats)

    xx = list(mean_stats.keys())
    yy = list(mean_stats.values())
    ax.bar(xx, yy)
    ax.grid()

    return