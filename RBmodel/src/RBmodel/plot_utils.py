import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product

from RBmodel.analysis import get_acfs, is_periodic
from RBmodel import cell_models
from RBmodel import analysis
import os


def plot_vars_vs_time(ax, cell, collapse_time=False):
    """
    Plot the relevant variables vs time.

    Parameters:
    ----------
    ax: plotting axis
        current axis on which to plot
    cell: cell_models.cell
        cell object with some history
    collapse_time: bool
        whether or not to plot along the cell cycle phase instead of time
    """

    # normalize all the variables to last value
    data_norm = pd.DataFrame(columns=["M", "RB", "[RB]"])
    data_norm["M"] = np.array(cell.M_hist)/cell.M_hist[-1]
    data_norm["RB"] = np.array(cell.RB_hist)/cell.RB_hist[-1]
    data_norm["[RB]"] = np.array(cell.RB_c_hist)/cell.RB_c_hist[-1]

    phase_durations, _ = analysis.get_phase_durations(cell)

    if collapse_time:
        ms = 1
        alpha = .3
        ax.scatter(phase_durations,
                   data_norm["M"], label="M", s=ms, alpha=alpha)
        ax.scatter(phase_durations,
                   data_norm["RB"], label="RB", s=ms, alpha=alpha)
        ax.scatter(phase_durations,
                   data_norm["[RB]"], label="[RB]", s=ms, alpha=alpha)

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
        ax.set_xlabel("Time")
        ax.set_ylabel("Variables")
        ax.set_ylim([data_norm.quantile(0.1).min()*0.8,
                     data_norm.quantile(0.9).max()*1.5])
        return


def plot_phase_space(ax, cell, yvar, x='M', plot_guide_lines=True):
    """
    Plot phase space between M and RB amount

    Parameters:
    ----------
    ax: plotting axis
        current axis on which to plot
    cell: cell_models.cell
        cell object with some history
    """
    if yvar == "RB":
        yvals = cell.RB_hist
        ylabel = "RB"
    elif yvar == "RBc":
        yvals = cell.RB_c_hist
        ylabel = "RBc"

    phase_vec = np.array(cell.phase_hist) == "G1"

    ax.scatter(
        np.array(cell.M_hist)[phase_vec == True],
        np.array(yvals)[phase_vec == True],
        alpha=.3, s=1, label="G1"
    )
    ax.scatter(
        np.array(cell.M_hist)[phase_vec == False],
        np.array(yvals)[phase_vec == False],
        alpha=.3, s=1, label="G2"
    )

    ax.set_xlabel("M")
    ax.set_ylabel(ylabel)
    ax.grid()
    ax.legend()

    pc_points = .5
    npoints = int(pc_points * len(yvals))
    ymin = .5 * np.min(np.array(yvals)[npoints:-1])
    ymax = 1.5 * np.max(np.array(yvals)[npoints:-1])
    ax.set_ylim([ymin, ymax])

    M_ = np.linspace(.5 * np.min(cell.M_hist), np.max(cell.M_hist) * 1.5, 20)
    if cell.division == "timer" and yvar=="RB":
        ax.plot(M_, M_ * cell.transition_th, color="red")
    if cell.transition == "size":
        ax.axvline(cell.transition_th, color="red")

    if plot_guide_lines:
        alpha = cell.params["alpha"]
        beta0 = cell.params["beta0"]
        eps = cell.params["epsilon"]
        delta = cell.params["delta"]

        ax.plot(M_, M_**delta * alpha / beta0, '--', color="gray")
        ax.plot(M_, M_**delta * alpha / beta0 / eps, '--', color="gray")


    return


def plot_autocorrelations(ax, cell, nlags=4000, prominence=.1):
    """
    Plot the autocorrelation of the different signals.

    Parameters:
    ----------
    ax: plotting axis
        axis on which to plot
    cell: cell_models.cell
        cell object with some history
    nlags: int
        lags to take into account for ACF computation
    prominence: float
        required prominence of peaks
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
    gamma=.9, epsilon=.01, eta=1, dt=.1, transition_th=2.,
    k_trans=5, division="timer", transition="size",
    time_SG2=1e-1, max_cycles=1e5, T=300,
):
    """
    Run a model and gets the relevant plots out

    TODO: functionalize the scatter plots (size control)
    """

    params = {
        'alpha': alpha,
        'beta0': beta0,
        'delta': delta,
        'gamma': gamma,
        'epsilon': epsilon,
        'eta': eta,
        'dt': dt,
        'duration_SG2': time_SG2,  # hr
        'transition_th': transition_th,
        'k_trans': k_trans,
        'division': division,
        'transition': transition,
        'max_cycles': max_cycles,
    }

    cell = cell_models.cell(params=params)
    cell.burn_in(1000)
    cell.grow(T)

    fig, ax = plt.subplots(4, 2, figsize=(10, 15))

    plot_vars_vs_time(ax[0, 0], cell, collapse_time=False)
    plot_phase_space(ax[1, 0], cell, yvar="RB")
    plot_phase_space(ax[1, 1], cell, yvar="RBc", plot_guide_lines=False)

    periods = plot_autocorrelations(ax[0, 1], cell, nlags=T/2, prominence=0)
    if any([p is None for p in periods]):
        print("Not periodic")
    else:
        if np.std(periods)/np.mean(periods) < .05:
            print(f"All signals with period approx. {np.mean(periods)}.")
        else:
            print("Not periodic")

    plot_vars_vs_time(ax[2, 0], cell, collapse_time=True)
    plot_stats(ax[2, 1], cell)

    _, stats = analysis.get_phase_durations(cell)
    slopes = analysis.compute_slopes(stats)

    color='gray'
    s=2

    x = stats['birth'] - np.mean(stats['birth'])
    y0 = stats['growth'][0] - np.mean(stats['growth'][0])
    y1 = stats['delta'][0] - np.mean(stats['delta'][0])
    slope_1 = slopes['slopes_G1_delta']
    slope_0 = slopes['slopes_G1_growth']

    ax[3,0].scatter(
        x, y0,
        color=color, s=s
        )


    ax[3,1].scatter(
        x, y1,
        color=color, 
        s=s
    )

    ax[3, 0].plot(x, slope_0*x)
    ax[3, 1].plot(x, slope_1*x)

    ax[3, 0].grid()
    ax[3, 0].set_xlabel("M at birth")
    ax[3, 0].set_ylabel("M_G1S/M_birth")

    ax[3, 1].grid()
    ax[3, 1].set_xlabel("M at birth")
    ax[3, 1].set_ylabel("M_G1S - M_birth")

    ax[3, 0].set_title(f"slope: {np.round(slope_0, decimals=3)}")
    ax[3, 1].set_title(f"slope: {np.round(slope_1, decimals=3)}")

    fig.tight_layout()

    return cell, periods


def plot_data(df, G1_th=10):
    """
    Plots the experimental data from .csv or .xlsx files

    Parameters:
    ----------
    df: DataFrame
        DataFrame of interest, created from data files
    G1_th: int
        minimum G1 duration to consider
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
    Plot the stats of the cell cycle.

    Parameters:
    ----------
    ax: plotting axis
        axis on which to plot
    cell: cell_models.cell
        cell object with some history
    """

    _, stats = analysis.get_phase_durations(cell)
    mean_stats = analysis.get_mean_stats(stats)

    xx = list(mean_stats.keys())
    yy = list(mean_stats.values())
    ax.bar(xx, yy)
    ax.grid()

    return

def plot_size_control(stats=None, cell=None, ax=None):
    """
    Series of plot to illustrate how the cell does size control

    TODO: modularize, and reuse in run_and_plot_test
    """
    if stats is None:
        _, stats = analysis.get_phase_durations(cell)

    if ax is None:
        _, ax = plt.subplots(2, 3, figsize=(10, 8))

    slopes = analysis.compute_slopes(stats)

    CV = stats["CV_M_birth"]
    plt.suptitle(f"M birth CV: {np.round(100 * CV, decimals=2)} [%]")

    color = 'gray'
    s = 2
    x = stats['birth'] - np.mean(stats['birth'])
    y0 = stats['growth'][0] - np.mean(stats['growth'][0])
    y1 = stats['delta'][0] - np.mean(stats['delta'][0])
    y2 = stats['delta'][1] - np.mean(stats['delta'][1])
    slope_0 = slopes['slopes_G1_growth']
    slope_1 = slopes['slopes_G1_delta']
    slope_2 = slopes["slopes_G2_delta"]

    ax[0,0].scatter(
        x, y0,
        color=color, s=s
        )


    ax[0, 2].scatter(
        x, y1,
        color=color, 
        s=s
    )

    ax[1, 2].scatter(x, y2, color=color, s=s)
    ax[0, 0].plot(x, slope_0*x)
    ax[0, 2].plot(x, slope_1*x)
    ax[1, 2].plot(x, slope_2*x)

    ax[0, 0].set_title(f"slope: {np.round(slope_0, decimals=3)}")
    ax[0, 2].set_title(f"slope: {np.round(slope_1, decimals=3)}")
    ax[1, 2].set_title(f"slope: {np.round(slope_2, decimals=3)}")

    ax[0,1].scatter(stats['birth'], stats['length'][0], color=color, s=s)
    ax[1,0].scatter(stats['birth'], stats['RB_G1'][0], color=color, s=s)
    ax[1,1].scatter(stats['birth'], stats['RBc_G1'][0], color=color, s=s)
    # ax[1,2].scatter(stats['birth'], stats['delta'][1], color=color, s=s)


    ax[0, 0].grid()
    ax[0, 0].set_xlabel("M at birth")
    ax[0, 0].set_ylabel("M_G1S/M_birth")

    ax[0, 1].grid()
    ax[0, 1].set_xlabel("M at birth")
    ax[0, 1].set_ylabel("G1 Length")

    ax[0, 2].grid()
    ax[0, 2].set_xlabel("M at birth")
    ax[0, 2].set_ylabel("M_G1S - M_birth")

    ax[1, 0].grid()
    ax[1, 0].set_xlabel("M at birth")
    ax[1, 0].set_ylabel("RB amount at birth")

    ax[1, 1].grid()
    ax[1, 1].set_xlabel("M at birth")
    ax[1, 1].set_ylabel("Rb Concentration at birth")

    ax[1, 2].grid()
    ax[1, 2].set_xlabel("M at birth")
    ax[1, 2].set_ylabel("MG2 - M_birth")


    plt.tight_layout()

    return




