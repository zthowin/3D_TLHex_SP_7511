#----------------------------------------------------------------------------------------
# Module housing element object consistent tangents.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 4, 2022
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
    return

@register_method
def get_G_Tangents(self, Parameters):
    # Assemble solid consistent tangents.
    self.G_Mtx = np.zeros((24,24))
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
    if Parameters.constitutive_model == 'neo-Hookean':
        self.dPdF = np.einsum('...ai, ...AI -> ...iIaA', self.identity, self.SPK)\
                    + Parameters.lambd*np.einsum('...Aa, ...Ii -> ...iIaA', self.F_inv, self.F_inv)\
                    - np.einsum('..., ...iIaA -> ...iIaA', Parameters.lambd*np.log(self.J) - Parameters.mu,\
                                                 (np.einsum('...Ai, ...Ia -> ...iIaA', self.F_inv, self.F_inv)\
                                                  + np.einsum('...ai, ...AI -> ...iIaA', self.identity, self.C_inv)))
    elif Parameters.constitutive_model == 'Saint Venant-Kirchhoff':
        self.dPdF = np.einsum('...ai, ...AI -> ...iIaA', self.identity, self.SPK)\
                    + Parameters.lambd*np.einsum('...iI, ...aA -> ...iIaA', self.F, self.F)\
                    + Parameters.mu*(np.einsum('...iA, ...aI -> ...iIaA', self.F, self.F)\
                                     + np.einsum('...iB, ...aB, ...AI -> ...iIaA', self.F, self.F, self.identity))
    else:
        sys.exit("ERROR. Constitutive model not recognized, check inputs.")

    self.dPdF_voigt = np.zeros((8,9,9), dtype=np.float64)
    for alpha in range(9):
        if alpha == 0:
            i = 0
            I = 0
        elif alpha == 1:
            i = 0
            I = 1
        elif alpha == 2:
            i = 0
            I = 2
        elif alpha == 3:
            i = 1
            I = 0
        elif alpha == 4:
            i = 1
            I = 1
        elif alpha == 5:
            i = 1
            I = 2
        elif alpha == 6:
            i = 2
            I = 0
        elif alpha == 7:
            i = 2
            I = 1
        elif alpha == 8:
            i = 2
            I = 2
        for beta in range(9):
            if beta == 0:
                a = 0
                A = 0
            elif beta == 1:
                a = 0
                A = 1
            elif beta == 2:
                a = 0
                A = 2
            elif beta == 3:
                a = 1
                A = 0
            elif beta == 4:
                a = 1
                A = 1
            elif beta == 5:
                a = 1
                A = 2
            elif beta == 6:
                a = 2
                A = 0
            elif beta == 7:
                a = 2
                A = 1
            elif beta == 8:
                a = 2
                A = 2

            self.dPdF_voigt[:,alpha,beta] = self.dPdF[:,i,I,a,A]

    self.G_uu_1 = np.einsum('kiI, kij, kjJ, k -> IJ', self.Bu, self.dPdF_voigt, self.Bu, self.weights*self.j)
    return
