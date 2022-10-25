import os
from tkinter import W
import pandas as pd
import numpy as np


def load_datasets():
    """
    Loads .xlsx datasets from Shuyuan
    """
    dfs = dict()

    for fnm in os.listdir("data/"):

        if fnm.endswith("csv") or "$" in fnm:
            continue
        nm = fnm.split(".xlsx")[0]
        dfs[nm] = dict()

        sheets = ["Area", "Frame", "Clover"]

        for sh in sheets:
            dfs[nm][sh] = pd.read_excel(
                os.path.join("data/", fnm), index_col=0, sheet_name=sh
            )

    return dfs


def process_dataset(df, window_size=5, dw=3):
    """
    Extract relevant metrics from the raw datasets. 
    """

    offset = int((window_size-1)/2)

    return get_idx_transition(df), get_cc_length(df), get_relative_variation_RB(df, dw=dw), get_masses(df, offset, dw)

def get_idx_transition(df_crt):
    """
    Returns a dict linking the cell id to the time index of the G1/S transition.
    """
    return {
        cell_id: df_crt['Frame'].index[df_crt['Frame'][cell_id] == 0][0] for cell_id in df_crt['Frame'].columns
    }

def get_cc_length(df_crt):
    """
    Returns a dict linking the cell id to the total duration of the cell cycle
    """
    return {
        cell_id: df_crt["Frame"].index[df_crt["Frame"][cell_id].isna()][0] for cell_id in df_crt["Frame"].columns
    }

def get_avg_RB(df_crt, cell_id, idx_transition=None, only_G1=True):
    """
    Returns the average RB amount for a given cell.
    """
    if only_G1:
        return df_crt["Clover"][cell_id].loc[:idx_transition].mean()
    else:
        return df_crt["Clover"][cell_id].mean()

def get_Delta_RB(df_crt, cell_id, dw=3):
    """
    Returns the variation of RB within a given cell cycle.
    """
    RB = df_crt["Clover"][cell_id].dropna().sort_values()
    return RB[-dw:].mean() - RB[:dw].mean()

def get_relative_variation_RB(df_crt, dw=3, only_G1=True):
    """
    Computes the variation of RB
    """
    idx_transitions = get_idx_transition(df_crt)
    return {
        cell_id: get_Delta_RB(df_crt, cell_id, dw=dw)/get_avg_RB(df_crt, cell_id, idx_transitions[cell_id], only_G1=only_G1) for cell_id in df_crt["Clover"].columns
    }

def get_masses(df_crt, offset, dw):
    """
    Extracts mass at transition and relative growth in G1 and G2.
    """
    idx_transition = get_idx_transition(df_crt)
    growth_G1 = dict()
    growth_G2 = dict()
    mass_transition = dict()

    for cell_id in df_crt['Area'].columns:
        mass_transition_ = df_crt['Area'].loc[
            idx_transition[cell_id]-offset:idx_transition[cell_id]+offset, cell_id
        ].mean()
        mass_init = df_crt['Area'].loc[1:1+dw, cell_id].mean()
        mass_final = df_crt['Area'][cell_id].max()
        mass_transition[cell_id] = mass_transition_
        growth_G1[cell_id] = mass_transition_/mass_init
        growth_G2[cell_id] = mass_final/mass_transition_

    return growth_G1, growth_G2, mass_transition


def convert_data(df, window_size=5, dw=3):

    idx_transition, cc_length, RB_var, (growth_G1, growth_G2, mass_transition) = process_dataset(
        df, window_size=window_size, dw=dw,)

    df_params = pd.DataFrame(
        columns=["Area_transition", "G1_length", "Cycle_length", "Delta_RB/Avg_RB"])

    for cell_id in mass_transition:
        df_params.loc[cell_id, "Area_transition"] = mass_transition[cell_id]
        df_params.loc[cell_id, "G1_growth"] = growth_G1[cell_id]
        df_params.loc[cell_id, "G2_growth"] = growth_G2[cell_id]
        df_params.loc[cell_id, "G1_length"] = idx_transition[cell_id]/3
        df_params.loc[cell_id, "Cycle_length"] = cc_length[cell_id]/3
        df_params.loc[cell_id, "Delta_RB/Avg_RB"] = RB_var[cell_id]

    df_params["G2_length"] = df_params["Cycle_length"] - df_params["G1_length"]

    return df_params

def get_phase_durations(cell):  
    """
    Maps the phase history to the phase durations.
    """
    
    phase_vec = np.array(cell.phase_hist) == "G1"

    # convert phase vec into a relative duration
    # if it is true: then it is in G1: it should be negative numbers growing until the zero at the transition
    # it it is false: then it is in G2: it should be positive numbers growing past the transition

    G1_growth, G2_growth = [], []
    G1_length, G2_length = [], []
    G1_mean_RB, G2_Delta_RB = [], []

    phase_duration = np.zeros(len(phase_vec))
    k = 0
    while k<len(phase_vec):

        # find the next index such that the phase changes
        try:
            next_switch = np.where((phase_vec[k:] != phase_vec[k]))[0][0]
        except:
            next_switch = len(phase_vec) - k 
        if phase_vec[k] == 1: # G1
            phase_duration[k:k+next_switch] = np.linspace(-next_switch+1, 0, next_switch)
            G1_growth.append(cell.M_hist[k+next_switch-1]/cell.M_hist[k])
            G1_length.append(next_switch*cell.dt)
            G1_mean_RB.append(np.mean(cell.RB_hist[k:k+next_switch]))

        elif phase_vec[k] == 0: # G2
            phase_duration[k:k+next_switch] = np.linspace(0, next_switch-1, next_switch)
            G2_growth.append(cell.M_hist[k+next_switch-1]/cell.M_hist[k])
            G2_length.append(next_switch*cell.dt)
            G2_Delta_RB.append(cell.RB_hist[k+next_switch-1] - cell.RB_hist[k])

        k = k+next_switch

        stats = {
            'growth': (G1_growth, G2_growth), 
            'length': (G1_length, G2_length), 
            'RB': (G1_mean_RB, G2_Delta_RB)
        }
        
    return phase_duration * cell.dt, stats

def get_mean_stats(stats):
    """
    Extracts means from the statistics for a given cell.
    """

    stats_dict = dict()
    stats_dict["G1_growth"] = np.median(stats['growth'][0])
    stats_dict["G2_growth"] = np.median(stats['growth'][1])
    stats_dict["G1_length"] = np.median(stats['length'][0])
    stats_dict["G2_length"] = np.median(stats['length'][1])
    stats_dict["DeltaRB/meanRB"] = np.mean(stats['RB'][1])/np.mean(stats['RB'][0])
    
    return stats_dict
