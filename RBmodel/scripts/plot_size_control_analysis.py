import pandas as pd
import os
from RBmodel import plot_utils
import matplotlib.pyplot as plt

EXP_DIR = "/Users/lucasfuentes/RB_model/data/size_control"

def plot_size_control_slope_analysis(df_exp, save_dir):
    """
    Plot the results from `size_control_analysis.py`
    
    For each type of G1/S transition threshold, we plot one figure where we summarize all the slopes
    """
    slopes_to_plot = ["slopes_G1_growth", "slopes_G1_length", "slopes_G1_delta", "CV_M_birth"]
    
    for transition in ["size", "RBc"]:
        
        df_exp_ = df_exp.loc[df_exp["transition"] == transition]
        uniq_k = df_exp["k_trans"].unique()
        
        fig, ax = plt.subplots(layout='constrained',figsize=(10, 4 * uniq_k.shape[0]))
        subfigs = fig.subfigures(uniq_k.shape[0], 1)
        
        for (i,k) in enumerate(uniq_k):
            
            ax = subfigs[i].subplots(1, len(slopes_to_plot))
            subfigs[i].suptitle(f"k={k}")
            
            # plot the slopes as a function of epsilon, for different values of delta
            
            df_ = df_exp_.loc[df_exp_["k_trans"] == k].copy()
            uniq_d = df_["delta"].unique()

            for d in uniq_d:
                df_tmp = df_.loc[df_["delta"] == d]

                for (j, s) in enumerate(slopes_to_plot):

                    ax[j].plot(df_tmp["epsilon"], df_tmp[s], "o--", label=f"delta={d}")

                    ax[j].set_ylabel(s)

                    ax[j].set_xlabel("epsilon")

            ax[2].legend()
            
#             subfigs[1].tight_layout()

            # plt.legend(loc="outside")      
#         fig.tight_layout()
        fig.suptitle(f"transition = {transition}")
        plt.savefig(os.path.join(save_dir, f"{transition}.pdf"), dpi=200)
        plt.close()
        
    return

df_exp = pd.read_csv(os.path.join(EXP_DIR, "results.csv"), index_col = 0)

plot_size_control_slope_analysis(df_exp, "/Users/lucasfuentes/RB_model/figs")