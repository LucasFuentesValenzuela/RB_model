from tkinter import W
import numpy as np

# TODO: work out what the phase space would/should be for something like this
# you have only a few parameters, and you can probably compare a few of them
# to each other, right?
# do that, and extract the data that you need from it.


class cell_v1(object):
    """
    Version 1 of the model
    """

    def __init__(self, alpha=1, beta0=1, delta=1, gamma=1, epsilon=.2, RB_thresh=3e-1, dt=1e-3):
        """
        Initialize a cell
        """
        self.M = 1
        self.RB = 1  # amount
        self.phase = "G1"
        # self.M_division = 2*self.M
        self.RB_division = self.RB/self.M # concentration
        self.RB_transition = RB_thresh # in concentration

        # parameters
        self.params = {
            "alpha": alpha,
            "beta0": beta0,
            "delta": delta,
            "gamma": gamma,
            "epsilon": epsilon,
            "dt": dt,
        }

        self.init_hists()

    def RB_c(self):
        return self.RB/self.M

    def divide(self):
        self.M = self.M/2 
        self.RB = self.RB/2
        self.phase = "G1"
        return

    def transit(self):
        self.phase = "G2"
        return

    def grow(self):
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

            # if self.phase=="G1":
            #     return
            # else:
            #     return

            M_tmp = self.M + \
                self.params["dt"] * self.params["gamma"] * \
                (self.M)**self.params["delta"]
            if self.RB_c() > self.RB_division:
                self.divide()
            else:
                self.M = M_tmp
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

            if self.RB_c() < self.RB_transition:
                self.transit()

            return

        step_size(self)
        step_concentrations(self)
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
