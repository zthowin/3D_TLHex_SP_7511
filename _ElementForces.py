#----------------------------------------------------------------------------------------
# Module housing element object internal force vectors.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 12, 2022
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
    self.G_int = np.zeros((Parameters.numElDOF), dtype=Parameters.float_dtype)
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
    self.FPK_voigt = np.zeros((Parameters.numGauss,Parameters.numDim**2), dtype=Parameters.float_dtype)
    for alpha in range(Parameters.numDim**2):
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
        self.FPK_voigt[:,alpha] = self.FPK[:,i,I]

    self.G_1 = np.einsum('kij, ki, k -> j', self.Bu, self.FPK_voigt, self.weights*self.j, dtype=Parameters.float_dtype)
    return

@register_method
def get_G2(self, Parameters):
    # Compute G_2^INT.
    self.grav_body       = np.zeros((Parameters.numGauss,Parameters.numDim), dtype=Parameters.float_dtype)
    self.grav_body[:,2]  = -Parameters.grav
    
    self.G_2 = np.einsum('kij, ki, k -> j', -self.Nu, self.rho_0*self.grav_body, self.weights*self.j, dtype=Parameters.float_dtype)
    return

@register_method
def get_GEXT(self, Parameters):
    # Compute G^EXT (for topmost element only).
    # Coded for Q8 Hex.
    if self.ID == (Parameters.numEl - 1) and Parameters.tractionProblem:
        self.traction      = np.zeros((4,3), dtype=Parameters.float_dtype)
        self.traction[:,2] = -Parameters.traction
    
        self.evaluate_Shape_Functions_2D(Parameters)
        self.G_EXT = np.einsum('kij, ki, k -> j', self.Nu_2D, self.traction, self.weights[4:8]*self.j_2D, dtype=Parameters.float_dtype)
    else:
        self.G_EXT = np.zeros((Parameters.numElDOF), dtype=Parameters.float_dtype)
    return
