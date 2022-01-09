import numpy as np


class Auxiliary:

    def Id(self,t0): # Current function
        if 10 < t0 < 15:
            return 15
        else:
            return 0

    # Several functions to calculate steady-state gating variable values.
    def alpha_n(self, Vm):
        return (0.01 * (10.0 - Vm)) / (np.exp(1.0 - (0.1 * Vm)) - 1.0)

    def beta_n(self, Vm):
        return (0.125 * np.exp(-Vm / 80.0))

    def alpha_m(self, Vm):  # sodium ion-channel rate functions.
        return (0.1 * (25.0 - Vm)) / (np.exp(2.5 - (0.1 * Vm)) - 1.0)

    def beta_m(self, Vm):
        return (4.0 * np.exp(-Vm / 18.0))

    def alpha_h(self, Vm):  # sodium inactivation
        return (0.07 * np.exp(-Vm / 20.0))

    def beta_h(self, Vm):
        return (1.0 / (np.exp(3.0 - (0.1 * Vm)) + 1.0))

    # n, m, and h steady-state values
    def n_inf(self, Vm=0.0):
        return self.alpha_n(Vm) / (self.alpha_n(Vm) + self.beta_n(Vm))

    def m_inf(self, Vm=0.0):
        return self.alpha_m(Vm) / (self.alpha_m(Vm) + self.beta_m(Vm))

    def h_inf(self, Vm=0.0):
        return self.alpha_h(Vm) / (self.alpha_h(Vm) + self.beta_h(Vm))