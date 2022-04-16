import numpy as np

rho = 1 # mass density of the cell

class cell():

    def __init__(self):
        """
        Initialize a cell
        """
        self.V = 1
        self.RB = 1 # concentration
        self.pRB = 0 # concentration
        self.phase = "G1"

    def mass(self):
        return rho*self.V

    def RB_amount(self):
        return self.V*self.RB
    
    def pRB_amount(self):
        return self.V*self.pRB

    def step(self):
        """
        Propagate one step forward
        """

        step_size()
        step_concentrations()

        def step_size(self):
            """
            Update the size/mass of the cell

            Main processes: 
            - G1 growth
            - G2 growth
            - division
            """
            
            if self.phase=="G1":
                return
            else:
                return

        def step_concentrations(self):
            """
            Update amounts/concentrations of RB and pRB

            Main processes: 
            - synthesis of RB
            - phosphorilation (conversion into pRB)
            - degradation of RB and pRB at different rates
            - dephosphorilation
            """

            return


