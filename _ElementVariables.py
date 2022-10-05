#----------------------------------------------------------------------------------------
# Module housing element object variables.
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
def compute_variables(self, Parameters):
    # Compute variables related to strains and stresses.
    self.get_dudX(Parameters)
    self.get_F()
    self.get_J()
    self.get_F_inv()
    self.get_C()
    self.get_C_inv()
    self.get_SPK(Parameters)
    self.get_FPK()
    self.get_b()
    self.get_v()
    self.get_E()
    self.get_e()
    self.get_Hencky()
    self.get_Cauchy()
    self.get_mean_Cauchy()
    self.get_von_Mises()
    self.get_rho_0(Parameters)
    self.get_rho()
    return

@register_method
def get_dudX(self, Parameters):
    # Compute solid displacement gradient.
    self.dudX = np.einsum('...ij, j -> ...i', self.Bu, self.u_global, dtype=np.float64)
    return

@register_method
def get_F(self):
    # Compute deformation gradient.
    #----------------------------------------------------
    # Reshape the identity matrix for all 8 Gauss points.
    #----------------------------------------------------
    shape = (8,3,3)
    self.identity = np.zeros(shape)
    np.einsum('ijj -> ij', self.identity)[:] = 1
    #---------------------------------------------------------------------------
    # Note that the dudX reshape is arbitrary; if we did not have 1D uniaxial
    # strain, we would need to use Voigt notation (here, du_i/dX_j = du_j/dX_i).
    #---------------------------------------------------------------------------
    self.F = (self.identity + self.dudX.reshape((8,3,3)))
    return

@register_method
def get_F_inv(self):
    # Compute inverse of deformation gradient.
    self.F_inv = np.linalg.inv(self.F)
    return

@register_method
def get_J(self):
    # Compute Jacobian of deformation.
    self.J = np.linalg.det(self.F)
    return

@register_method
def get_C(self):
    # Compute right Cauchy-Green tensor.
    self.C = np.einsum('...iI, ...iJ -> ...IJ', self.F, self.F, dtype=np.float64)
    return

@register_method
def get_C_inv(self):
    # Compute inverse of right Cauchy-Green tensor.
    self.C_inv = np.linalg.inv(self.C)
    return

@register_method
def get_SPK(self, Parameters):
    # Compute second Piola-Kirchoff stress tensor.
    self.SPK = Parameters.mu*self.identity + np.einsum('..., ...IJ -> ...IJ',\
                                                       Parameters.lambd*np.log(self.J) - Parameters.mu,\
                                                       self.C_inv, dtype=np.float64)
    return

@register_method
def get_FPK(self):
    # Compute first Piola-Kirchoff stress tensor.
    self.FPK = np.einsum('...iI, ...IJ -> ...iI', self.F, self.SPK, dtype=np.float64)
    return

@register_method
def get_Cauchy(self):
    # Compute Cauchy stress tensor.
    self.sigma = np.einsum('...iI, ...jI, ... -> ...ij', self.FPK, self.F, 1/self.J, dtype=np.float64)
    return

@register_method
def get_mean_Cauchy(self):
    # Compute the mean Cauchy stress, i.e., thermodynamic pressure.
    self.sigma_mean = (1/3)*np.einsum('...ii', self.sigma)
    return

@register_method
def get_von_Mises(self):
    # Compute the von Mises stress.
    self.von_mises = np.sqrt(3/2)*np.linalg.norm((self.sigma - np.einsum('..., ...ij -> ...ij', self.sigma_mean, self.identity)))
    return

@register_method
def get_b(self):
    # Compute left Cauchy-Green tensor.
    self.b = np.einsum('...iI, ...jI -> ...ij', self.F, self.F, dtype=np.float64)
    return

@register_method
def get_v(self):
    # Compute the left stretch tensor.
    self.v = np.sqrt(self.b)
    return

@register_method
def get_E(self):
    # Compute Green-Lagrange strain tensor.
    self.E = (self.C - self.identity)/2
    return

@register_method
def get_e(self):
    # Compute Euler-Almansi strain tensor.
    self.e = (self.identity - np.linalg.inv(self.b))/2
    return

@register_method
def get_Hencky(self):
    # Compute Hencky strain.
    self.Hencky = np.log(self.v)
    return

@register_method
def get_rho(self):
    # Compute mass density in current configuration.
    self.rho = np.einsum('..., ...i -> ...i', self.J, self.rho_0, dtype=np.float64)
    return

@register_method
def get_rho_0(self, Parameters):
    # Compute mass density in reference configuration.
    self.rho_0 = Parameters.rhoS_0*np.ones((8,3), dtype=np.float64)
    return