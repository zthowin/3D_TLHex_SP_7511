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
    self.G_int = np.zeros((24), dtype=np.float64)
    self.get_G1()
    self.get_G2(Parameters)      
    self.get_GEXT(Parameters)

    try:
        self.G_int += self.G_1 + self.G_2 - self.G_EXT
    except FloatingPointError:
        print("ERROR. Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
        raise FloatingPointError
    return

@register_method
def get_G1(self):
    # Compute G_1^INT.
    #---------------------------------------------------------------------------
    # Note that the reshape is arbitrary; if we did not have 1D uniaxial strain,
    # we would need to use Voigt notation (here, du_i/dX_j = du_j/dX_i = 0).
    #---------------------------------------------------------------------------
    self.G_1 = np.einsum('kij, ki, k -> j', self.Bu, self.FPK.reshape((8,9)), self.weights*self.j, dtype=np.float64)
    return

@register_method
def get_G2(self, Parameters):
    # Compute G_2^INT.
    self.grav_body       = np.zeros((8,3))
    self.grav_body[:,2]  = -Parameters.grav
    
    self.G_2 = np.einsum('kij, ki, k -> j', -self.Nu, self.rho_0*self.grav_body, self.weights*self.j, dtype=np.float64)
    return

@register_method
def get_GEXT(self, Parameters):
    # Compute G^EXT.
    if self.ID == 1 and Parameters.tract > 0:
        self.traction      = np.zeros((4,3))
        self.traction[:,2] = -Parameters.tract
    
        self.evaluate_Shape_Functions_2D()
        self.G_EXT = np.einsum('kij, ki, k -> j', self.Nu_2D, self.traction, self.weights[4:8]*self.j_2D, dtype=np.float64)
    else:
        self.G_EXT = np.zeros((24), dtype=np.float64)
    return
