#----------------------------------------------------------------------------------------
# Module housing element object consistent tangents.
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
def get_G_Tangents(self):
    # Assemble solid consistent tangents.
    self.G_Mtx = np.zeros((24,24,8))
    self.get_G_uu_1()

    try:
        self.G_Mtx += self.G_uu_1
    except FloatingPointError:
        print("ERROR. Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
        raise FloatingPointError
    return

@register_method
def get_G_uu_1(self):
    # Compute G_uu_1.
    dPdF = np.einsum('ai..., AI...', self.identity, self.SPK) + Parameters.lambd*np.einsum('Aa...,Ii...', self.F_inv, self.F_inv)\
           + (Parameters.lambd*np.log(self.J) - Parameters.mu)*(np.einsum('Ai..., Ia...', self.F_inv, self.F_inv)\
                                                                + np.einsum('ai..., AI...', self.identity, self.C_inv))

    self.G_uu_1 = np.einsum('iI..., aA..., aA...', self.Bu, self.Bu, dPdF*self.weights*self.j, dtype=np.float64)
    return
