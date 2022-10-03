#----------------------------------------------------------------------------------------
# Module housing element object internal force vectors.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 2, 2022
#----------------------------------------------------------------------------------------
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("MODULE WARNING. NumPy not installed.")

try:
    import Lib
except ImportError:
    sys.exit("MODULE WARNING. 'Lib.py' not found, check configuration.")

__methods__     = []
register_method = Lib.register_method(__methods__) 

@register_method
def compute_forces(self, Parameters):
    # Compute internal forces.
    self.get_G_Forces(Parameters)
    return

@register_method
def get_G_Forces(self, Parameters):
    # Assemble solid internal force vectors.
    self.get_G1()
    self.get_G2(Parameters)

    try:
        self.G_int = self.G_1 + self.G_2
        # print(self.G_1)
        # print(self.G_2)
        # input()
    except FloatingPointError:
        print("ERROR. Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
        raise FloatingPointError
    return

@register_method
def get_G1(self):
    # Compute G_1^INT.
    self.G_1 = np.einsum('ijk, ki', self.Bu, np.einsum('...I, ...', self.FPK.reshape((8,9)), self.weights*self.j), dtype=np.float64)
    return

@register_method
def get_G2(self, Parameters):
    # Compute G_2^INT.
    self.G_2 = np.einsum('ijk, ki', self.Nu, np.einsum('...i, ...', self.rho, self.weights*self.j*Parameters.grav), dtype=np.float64)
    # print(self.G_2)
    # self.G_2 = np.einsum('ijk, ki', self.Nu, self.rho*self.weights*self.j*np.array([0,0,Parameters.grav]))
    return