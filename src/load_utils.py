import os
import pandas as pd


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

    def get_avg_RB(df_crt, cell_id):
        """
        Returns the average RB amount for a given cell.
        """
        return df_crt["Clover"][cell_id].mean()

    def get_Delta_RB(df_crt, cell_id, dw=3):
        """
        Returns the variation of RB within a given cell cycle.
        """
        RB = df_crt["Clover"][cell_id].dropna().sort_values()
        return RB[:dw].mean() - RB[-dw:].mean()

    def get_relative_variation_RB(df_crt, dw=3):
        """
        Computes the variation of RB
        """
        return {
            cell_id: get_Delta_RB(df_crt, cell_id, dw=dw)/get_avg_RB(df_crt, cell_id) for cell_id in df_crt["Clover"].columns
        }

    def get_masses(df_crt, offset, dw):
        """
        Extracts mass at transition and relative growth in G1 and G2.
        """
        idx_transition = get_idx_transition(df)
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

    return get_idx_transition(df), get_cc_length(df), get_relative_variation_RB(df, dw=dw), get_masses(df, offset, dw)


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
