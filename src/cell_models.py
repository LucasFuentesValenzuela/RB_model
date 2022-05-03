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

    def __init__(
            self, alpha=2, beta0=3, delta=1, 
            gamma=.9, epsilon=.01, RB_thresh=1.1, dt=1e-3, 
            division="concentration"
        ):
        """
        Initialize a cell
        """
        self.M = 1
        self.RB = 2  # amount
        self.phase = "G1"
        self.division = division

        if division=="concentration":
            self.RB_division = self.RB/self.M  # concentration
        elif division=="amount":
            self.RB_division = self.RB
        self.RB_transition = RB_thresh  # in concentration

        # parameters
        self.params = {
            "alpha": alpha,
            "beta0": beta0,
            "delta": delta,
            "gamma": gamma,
            "epsilon": epsilon,
            "dt": dt,
        }

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

            M_tmp = self.M + \
                self.params["dt"] * self.params["gamma"] * \
                (self.M)**self.params["delta"]
            if (self.division=="concentration") and (self.RB_c() > self.RB_division):
                self.divide()
            elif (self.division=="amount") and (self.RB > self.RB_division):
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

    def check_params(self):
        """
        """

        alpha, beta0, epsilon = self.params['alpha'], self.params['beta0'], self.params['epsilon']

        if self.division=="concentration":
            if any([
                not (alpha/beta0/epsilon > self.RB_division),
                not (self.RB_division > self.RB_transition),
                not (self.RB_transition > alpha/beta0)
            ]):
                print("Parameters invalid")
                return

        elif self.division=="amount":
            return
