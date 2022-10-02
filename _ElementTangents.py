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
def compute_tangents(self, Parameters):
    # Compute element tangents.
    self.get_G_Tangents(Parameters)

@register_method
def get_G_Tangents(self, Parameters):
    # Assemble solid consistent tangents.
    self.G_Mtx = np.zeros((24,24,8))
    self.get_G_uu_1(Parameters)

    try:
        self.G_Mtx += self.G_uu_1
    except FloatingPointError:
        print("ERROR. Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
        raise FloatingPointError
    return

@register_method
def get_G_uu_1(self, Parameters):
    # Compute G_uu_1.
    # debug1 = np.einsum('ai..., AI...', self.identity, self.SPK)
    # debug2 = Parameters.lambd*np.einsum('...Aa,...Ii', self.F_inv, self.F_inv)
    # debug3 = np.einsum('...Ai, ...Ia', self.F_inv, self.F_inv)
    # debug4 = np.einsum('ai..., ...AI', self.identity, self.C_inv)
    # print(debug1.shape)
    # print(debug2.shape)
    # print(debug3.shape)
    # print(debug4.shape)
    # debug5 = np.einsum('..., ...aiAI', Parameters.lambd*np.log(self.J) - Parameters.mu,debug4)
    # print(debug5.shape)
    self.dPdF = np.einsum('ai..., AI...', self.identity, self.SPK)\
                + Parameters.lambd*np.einsum('...Aa,...Ii', self.F_inv, self.F_inv)\
                + np.einsum('..., ...aiAI', Parameters.lambd*np.log(self.J) - Parameters.mu,\
                                            (np.einsum('...Ai, ...Ia', self.F_inv, self.F_inv)\
                                             + np.einsum('ai..., ...AI', self.identity, self.C_inv)))

    
    print(self.dPdF.shape)
    print(self.Bu.shape)
    self.G_uu_1 = np.einsum('iI..., aA..., aiAI...', self.Bu, self.Bu, self.dPdF, dtype=np.float64)
    return
