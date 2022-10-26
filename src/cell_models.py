from decimal import DefaultContext
from tkinter import W
import numpy as np
import pandas as pd

DEFAULT_PARAMS = {
    'alpha': 2,
    'beta0': 0.09,
    'delta': 1.,
    'gamma': 0.03,
    'epsilon': .2,
    'dt': 1e-1,
    'duration_SG2': 12, # hr
    'transition_th': 1.,
    'k_trans': 5, 
    'division': "timer",
    'transition': "size"
}


class population(object):
    """
    Population of cells.
    """

    def __init__(
        self, N_cells, params=DEFAULT_PARAMS
    ):
        """
        Initialize a population of cells
        """
        self.cells = [cell(params=params) for _ in range(N_cells)]
        self.params = params
        self.N_cells = N_cells

    def grow(self, T):
        """
        Grow the population for T timesteps.
        """

        for cell in self.cells:
            cell.grow(T)

        return

    def gather_results(self):
        """
        Consolidate the time series for every cell.
        """
        empty_df = pd.DataFrame(columns = [k for k in range(self.N_cells)])
        M_df, RB_df, RBc_df, phase_df = empty_df.copy(), empty_df.copy(), empty_df.copy(), empty_df.copy()

        for k, cell in enumerate(self.cells): 

            M_df[k] = cell.M_hist
            RB_df[k] = cell.RB_hist
            RBc_df[k] = cell.RB_c_hist
            phase_df[k] = cell.phase_hist

        return {"M": M_df, "RB": RB_df, "RBc": RBc_df, "phase": phase_df}


class cell(object):
    """
    Representation of a cell as a dynamical system.
    """

    def __init__(
        self,
        params=DEFAULT_PARAMS, 
    ):
        """
        Initialize a cell
        """
        self.M = 1
        self.M_birth = self.M
        self.RB = 2  # amount
        self.phase = "G1"
        self.time_SG2 = 0

        self.division = params["division"]
        self.transition = params["transition"]
        self.dt = params["dt"]

        if self.division == "concentration":
            self.division_th = self.RB/self.M  # concentration
        elif self.division == "amount":
            self.division_th = self.RB
        elif self.division == "timer":
            self.division_th = params["duration_SG2"]

        if self.transition == "size":  # here we assume the size is controlled as a multiple of birth
            self.transition_th = params["transition_th"] * self.M_birth  # in concentration or size
        else:
            print("Transition other than size needs implementation")
            return

        # parameters
        self.params = params
        self.check_params()
        self.init_hists()

        return

    def RB_c(self):
        return self.RB/self.M

    def divide(self):
        self.M = self.M/2
        self.RB = self.RB/2
        self.phase = "G1"
        return

    def transit(self):
        """
        """

        self.phase = "G2"
        self.time_SG2 = 0
        return

    def grow(self, T):
        """
        """

        for _ in range(T):
            self.step()

        return

    def step(self):
        """
        Propagate one step forward
        """

        def step_size(self):
            """
            Update the size/mass of the cell

            Main processes: 
            - G1 growth
            - G2 growth
            - division
            """

            self.M = self.M + \
                self.params["dt"] * self.params["gamma"] * \
                (self.M)**self.params["delta"]

            if self.phase == "G2":
                self.time_SG2 = self.time_SG2 + self.dt

                if (self.division == "concentration") and (self.RB_c() > self.division_th):
                    self.divide()
                elif (self.division == "amount") and (self.RB > self.division_th):
                    self.divide()
                elif (self.division == "timer") and (self.time_SG2 > self.division_th):
                    self.divide()
            return

        def step_concentrations(self):
            """
            Update amounts/concentrations of RB and pRB

            Main processes: 
            - synthesis of RB
            - degradation of RB at different rates
            """

            beta = self.beta()

            self.RB = self.RB + self.params['dt'] * \
                (self.params['alpha']*self.M - beta*self.RB)
            return

        def check_transit(self):
            """
            """
            if (self.transition == "RBc") and (self.phase == "G1"):
                transition_probability = np.maximum(self.transition_th -  self.Rb_c(), 0) * self.params["k_trans"]
            elif (self.transition == "size") and (self.phase == "G1"):
                transition_probability = np.maximum(
                    (self.M - self.transition_th)/self.transition_th, 0
                    ) * self.params["k_trans"] * self.params["dt"]
            else:
                transition_probability = 0

            if np.random.binomial(1, np.minimum(transition_probability, 1)): 
                self.transit()
            return

        step_size(self)
        step_concentrations(self)
        check_transit(self)
        self.update_hists()

    def beta(self):
        """
        """
        if self.phase == "G1":
            return self.params["beta0"]
        elif self.phase == "G2":
            return self.params["beta0"] * self.params["epsilon"]

    def init_hists(self):
        """
        """
        self.M_hist = [self.M]
        self.RB_hist = [self.RB]
        self.RB_c_hist = [self.RB_c()]
        self.phase_hist = [self.phase]
        return

    def update_hists(self):
        """
        """
        self.M_hist.append(self.M)
        self.RB_hist.append(self.RB)
        self.RB_c_hist.append(self.RB_c())
        self.phase_hist.append(self.phase)
        return

    def check_params(self):
        """
        """

        alpha, beta0, epsilon = self.params['alpha'], self.params['beta0'], self.params['epsilon']

        if self.division == "concentration" and self.transition == "RBc":
            if any([
                not (alpha/beta0/epsilon > self.division_th),
                not (self.division_th > self.transition_th),
                not (self.transition_th > alpha/beta0)
            ]):
                print("Parameters invalid")
                return

        elif self.division == "amount" and self.transition == "RBc":
            if any([
                not (alpha/beta0/epsilon > self.RB_transition),
                not(self.RB_transition > alpha/beta0)
            ]):
                print("Parameters invalid")
                print(f"alpha/beta0/epsilon = {alpha/beta0/epsilon}")
                print(f"self.RB_transition = {self.RB_transition}")
                print(f"alpha/beta0 = {alpha/beta0}")
                return
        else:
            # print("Params check not implemented")
            pass
