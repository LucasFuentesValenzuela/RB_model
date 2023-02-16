# NOTE: these results are valid only for delta=1

import numpy as np

def analytical_RBc(RBc0, params, t, phase='G1'):
    """
    Analytical evolution of the RB concentration for delta=1. 

    Parameters:
    ----------
    - RBc0: initial RB concentration
    - params: dict of parameters for the model
    - t: maximum time of integration
    - phase: 'G1' or 'G2'
    """
    
    if phase == 'G1': 
        beta = params['beta0']
    else:
        beta = params['beta0'] * params['epsilon']
    
    C1 = params['alpha']/(beta+params['gamma'])
    C2 = beta + params['gamma']
    
    return (RBc0 - C1) * np.exp(-C2*t) + C1

def analytical_M(M0, params, t):
    """
    Analytical evolution of the M for delta = 1. 

    Parameters:
    ----------
    - M0: initial mass
    - params: dict of parameters for the model
    - t: maximum time of integration
    """
    return M0 * np.exp(params['gamma']*t)

def analytical_RB(RB0, M0, params, t, phase='G1'):
    """
    Analytical evolution of the RB amount for delta = 1.

    Parameters:
    ----------
    - RB0: initial RB concentration amount
    - M0: initial mass
    - params: dict of parameters for the model
    - t: maximum time of integration
    - phase: 'G1' or 'G2'
    """
    
    if phase == 'G1': 
        beta = params['beta0']
    else:
        beta = params['beta0'] * params['epsilon']
        
    C1 = params['alpha']/(beta+params['gamma'])
        
    return RB0 * np.exp(-beta*t) + C1 * M0 * (np.exp(params['gamma']*t) - np.exp(-beta*t))
    

def theoretical_M_ratio(params): 
    """
    Analytical ratio between the mass at division and G1/S, for delta=1.

    Parameters:
    -----------
    - params: dict of parameters for the model.
    """

    tau = params['duration_SG2']
    A1 = np.exp(params['gamma'] * tau)/2
    EXP = - params['gamma']/(params['gamma'] + params['beta0'])
    
    C1 = params['alpha']/(params['beta0']+params['gamma'])
    C2 = params['alpha']/(params['beta0']*params['epsilon']+params['gamma'])
    
    Theta = params['transition_th']
    
    num = Theta - C1
    
    deno = (Theta + C2) * np.exp(-tau*(params['beta0'] * params['epsilon']+params['gamma'])) + C2 - C1
    
    return A1 * (num/deno)**EXP