from tkinter import W
import numpy as np

rho = 1 # mass density of the cell

class cell():
    """
    Version 1 of the model
    """

    def __init__(self, alpha=1, beta0=1, delta=1, gamma=1, epsilon=1, RB_thresh=1e-1, dt=1e-3):
        """
        Initialize a cell
        """
        self.V = 1
        self.M = rho*self.V
        self.RB = 1 # concentration
        self.phase = "G1"
        self.M_division = 2*self.M
        self.RB_threshold = RB_thresh

        # parameters
        self.params={
            "alpha": alpha, 
            "beta0": beta0, 
            "delta": delta, 
            "gamma": gamma, 
            "epsilon": epsilon,
            "dt": dt,
        }

        self.init_hists()

    def RB_amount(self):
        return self.V*self.RB

    def divide(self):
        self.M = self.M/2 # concentration remains the same
        self.phase="G1"
        return

    def transit(self):
        self.phase="G2"
        return
    
    def grow(self):
        """
        Propagate one step forward
        """

        step_size()
        step_concentrations()
        self.update_hists()

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

            M_tmp = self.M + self.params["dt"] * self.params["gamma"] * (self.M)**self.params["delta"]
            if M_tmp > self.M_division:
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

            self.RB = self.RB + self.params['dt'] * (self.params['alpha']*self.M - beta*self.RB)

            if self.RB < self.RB_threshold:
                self.transit()

            return

    def beta(self):
        """
        """
        if self.phase=="G1":
            return self.params["beta0"]
        elif self.phase=="G2":
            return self.params["beta0"] * self.params["epsilon"]

    def init_hists(self):
        """
        """ 
        self.M_hist = [self.M]
        self.RB_hist = [self.RB]
        return

    def update_hists(self):
        """
        """
        self.M_hist.append(self.M)
        self.RB_hist.append(self.RB)
        return


