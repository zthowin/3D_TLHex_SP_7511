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
    self.G_int = np.zeros((24), dtype=np.float64)
    self.get_G1()
    self.get_G2(Parameters)

    try:
        self.G_int += self.G_1 + self.G_2
    except FloatingPointError:
        print("ERROR. Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
        raise FloatingPointError
    return

@register_method
def get_G1(self):
    # Compute G_1^INT.
    #------------------------------------------
    # Generate voigt form of FPK stress tensor.
    #------------------------------------------
    # self.FPK_voigt = np.zeros((8,9), dtype=np.float64)
    # for alpha in range(9):
    #     if alpha == 0:
    #         i = 0
    #         I = 0
    #     elif alpha == 1:
    #         i = 0
    #         I = 1
    #     elif alpha == 2:
    #         i = 0
    #         I = 2
    #     elif alpha == 3:
    #         i = 1
    #         I = 0
    #     elif alpha == 4:
    #         i = 1
    #         I = 1
    #     elif alpha == 5:
    #         i = 1
    #         I = 2
    #     elif alpha == 6:
    #         i = 2
    #         I = 0
    #     elif alpha == 7:
    #         i = 2
    #         I = 1
    #     elif alpha == 8:
    #         i = 2
    #         I = 2
    #     self.FPK_voigt[:,alpha] = self.FPK[:,i,I]
    # self.G_1 = np.einsum('kij, ki, k -> j', self.Bu, self.FPK_voigt, self.weights*self.j, dtype=np.float64)
    # print(self.FPK.shape)
    self.G_1 = np.einsum('kij, ki, k -> j', self.Bu, self.FPK.reshape((8,9)), self.weights*self.j, dtype=np.float64)
    return

@register_method
def get_G2(self, Parameters):
    # Compute G_2^INT.
    self.grav_body       = np.zeros((8,3))
    self.grav_body[:,2]  = -Parameters.grav
    self.G_2 = np.einsum('kij, ki -> j', self.Nu, np.einsum('...i, ...i, ... -> ...i',\
                                                            self.rho, self.grav_body, self.weights*self.j),\
                                                            dtype=np.float64)
    return