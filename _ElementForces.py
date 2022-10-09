#----------------------------------------------------------------------------------------
# Module housing element object internal force vectors.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 6, 2022
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
    self.G_int = np.zeros((24), dtype=Parameters.float_dtype)
    self.get_G1(Parameters)
    self.get_G2(Parameters)      
    self.get_GEXT(Parameters)

    try:
        self.G_int += self.G_1 + self.G_2 - self.G_EXT
    except FloatingPointError:
        print("ERROR. Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
        raise FloatingPointError
    return

@register_method
def get_G1(self, Parameters):
    # Compute G_1^INT.
    self.FPK_voigt = np.zeros((8,9), dtype=Parameters.float_dtype)
    for i in range(9):
        if i == 0:
            alpha = 0
            beta  = 0
        elif i == 1:
            alpha = 0
            beta  = 1
        elif i == 2:
            alpha = 0
            beta  = 2
        elif i == 3:
            alpha = 1
            beta  = 0
        elif i == 4:
            alpha = 1
            beta  = 1
        elif i == 5:
            alpha = 1
            beta  = 2
        elif i == 6:
            alpha = 2
            beta  = 0
        elif i == 7:
            alpha = 2
            beta  = 1
        elif i == 8:
            alpha = 2
            beta  = 2
        self.FPK_voigt[:,i] = self.FPK[:,alpha,beta]

    self.G_1 = np.einsum('kij, ki, k -> j', self.Bu, self.FPK_voigt, self.weights*self.j, dtype=Parameters.float_dtype)
    return

@register_method
def get_G2(self, Parameters):
    # Compute G_2^INT.
    self.grav_body       = np.zeros((8,3), dtype=Parameters.float_dtype)
    self.grav_body[:,2]  = -Parameters.grav
    
    self.G_2 = np.einsum('kij, ki, k -> j', -self.Nu, self.rho_0*self.grav_body, self.weights*self.j, dtype=Parameters.float_dtype)
    return

@register_method
def get_GEXT(self, Parameters):
    # Compute G^EXT (for topmost element only).
    if self.ID == (Parameters.numEl - 1) and Parameters.tractionProblem:
        self.traction      = np.zeros((4,3), dtype=Parameters.float_dtype)
        self.traction[:,2] = -Parameters.traction
    
        self.evaluate_Shape_Functions_2D(Parameters)
        self.G_EXT = np.einsum('kij, ki, k -> j', self.Nu_2D, self.traction, self.weights[4:8]*self.j_2D, dtype=Parameters.float_dtype)
    else:
        self.G_EXT = np.zeros((24), dtype=Parameters.float_dtype)
    return
