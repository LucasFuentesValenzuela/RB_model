from RBmodel import cell_models, analysis
import itertools
import pandas as pd
import os
import pickle

SAVE_DIR = "/Users/lucasfuentes/RB_model/data/size_control"

params = cell_models.DEFAULT_PARAMS.copy()

df_exp = pd.DataFrame(
    columns = list(params.keys()) + [
        "stats_file_path", 
    ]
)

T_burn = 1000
T = int(1e6) # number of timesteps

# parameters we will test
transition_vec = ["size", "RBc"]
delta_vec = [.1, .5, .8, 1.]
epsilon_vec = [.1, .2, .5, .8, 1.]
k_trans_vec = [1., 5., 10, 100]

exp_id = 0

for t, d, e, k in itertools.product(transition_vec, delta_vec, epsilon_vec, k_trans_vec):

    try: 
        # update the params
        params_crt = params.copy()
        params_crt["transition"] = t
        params_crt["delta"] = d
        params_crt["epsilon"] = e
        params_crt["k_trans"] = k

        cell = cell_models.cell(params=params_crt)
        cell.burn_in(T_burn)
        cell.grow(T)
        _, stats = analysis.get_phase_durations(cell)
        slopes = analysis.compute_slopes(stats)

    except:
        continue

    for p in params_crt:
        df_exp.loc[exp_id, p] = params_crt[p]
    
    for sl in slopes:
        df_exp.loc[exp_id, sl] = slopes[sl]

    fnm = os.path.join(SAVE_DIR, f"{exp_id}.pkl")
    df_exp.loc[exp_id, "stats_file_path"] = fnm

    exp_id += 1

    with open(fnm, 'wb') as f:
        pickle.dump(stats, f)
    
# save
df_exp.to_csv(os.path.join(SAVE_DIR, "results.csv"))


