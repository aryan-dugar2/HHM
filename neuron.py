from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from auxiliary import Auxiliary


class Neuron(Auxiliary):

    def __init__(self, v_k=-15, v_na=120, v_l=15):
        """Initialises key neuron parameters. """
        self.gK = 36.0  # Potassium channel conductance
        self.gNa = 120.0  # Sodium channel conductance
        self.gL = 0.3  # Leak conductance
        self.Cm = 1.0  # Membrane capacitance

        self.VK = v_k  # Potential difference due to potassium gradient
        self.VNa = v_na  # Potential difference due to sodium gradient
        self.Vl = v_l  # Potential difference due to membrane leak

        self.time = np.linspace(0, 30, 10000)
        self.Vy = self.simulate()

    def compute_derivatives(self, y, t0):
        """Computes time-derivatives of V, n_inf, m_inf and h_inf, returning it in a 1 x 4 array."""
        dy = np.zeros((4,))  # returns [0. 0. 0. 0.]

        Vm = y[0]  # Vm = -60
        n = y[1]  # equated to n_inf.
        m = y[2]  # equated to m_inf
        h = y[3]  # equated to h_inf - all from Y.

        # Pre-calculations
        GK = (self.gK / self.Cm) * np.power(n, 4.0)
        GNa = (self.gNa / self.Cm) * np.power(m, 3.0) * h
        GL = self.gL / self.Cm

        # Computing derivatives. Id() is inherited from Auxiliary class and returns the current at given time.
        dy[0] = self.Id(t0) / self.Cm - (GK * (Vm - self.VK)) - (GNa * (Vm - self.VNa)) - (GL * (Vm - self.Vl))

        dy[1] = (self.alpha_n(Vm) * (1.0 - n)) - (self.beta_n(Vm) * n)  # dn/dt

        dy[2] = (self.alpha_m(Vm) * (1.0 - m)) - (self.beta_m(Vm) * m)  # dm/dt

        dy[3] = (self.alpha_h(Vm) * (1.0 - h)) - (self.beta_h(Vm) * h)  # dh/dt

        return dy

    def simulate(self, v0=0.0):
        """ Carries out numerical integration to visualise dynamics. v0 = initial voltage (mV)"""
        Y = np.array([v0, self.n_inf(), self.m_inf(), self.h_inf()])
        Vy = odeint(self.compute_derivatives, Y, self.time)

        return Vy

    def plot(self, param, param_name):
        """Plot single parameter"""
        plt.plot(self.time, param)
        plt.xlabel("Time (s)")
        plt.ylabel(param_name)
        plt.show()

    def plotAll(self):
        """Plots n_inf, m_inf, h_inf and V vs. Time (S)"""
        fig, ax = plt.subplots(4, 1)
        ax[0].plot(self.time, self.Vy[:, 1])  # V,n,m,t
        ax[0].set_ylabel("n_inf")
        ax[1].plot(self.time, self.Vy[:, 2])  # V,n,m,t
        ax[1].set_ylabel("m_inf")
        ax[2].plot(self.time, self.Vy[:, 3])  # V,n,m,t
        ax[2].set_ylabel("h_inf")
        ax[3].plot(self.time, self.Vy[:, 0])  # V,n,m,t
        ax[3].set_ylabel("V (mV)")
        ax[3].set_xlabel("Time (S)")
        plt.show()


def main():
    """Sample usage of the classes."""
    neuron = Neuron()

    # To plot V and steady-state gating variable values.
    neuron.plotAll()

    # To plot just V
    # neuron.plot(neuron.Vy[:,0],"V")


if __name__ == "__main__":
    main()
