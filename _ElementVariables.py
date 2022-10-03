#----------------------------------------------------------------------------------------
# Module housing element object variables.
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
def compute_variables(self, Parameters):
    # Compute variables related to strains and stresses.
    self.get_dudX()
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
    self.get_rho_0(Parameters)
    self.get_rho()
    return

@register_method
def get_dudX(self):
    # Compute solid displacement gradient.
    self.dudX = np.einsum('ij..., j...', self.Bu, self.u_global, dtype=np.float64)
    return

@register_method
def get_F(self):
    # Reshape the identity matrix for all 8 Gauss points.
    shape = (8,3,3)
    self.identity = np.zeros(shape)
    np.einsum('ijj->ij', self.identity)[:] = 1
    # Compute deformation gradient.
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
    self.C = np.einsum('...iI, ...iJ', self.F, self.F, dtype=np.float64)
    return

@register_method
def get_C_inv(self):
    # Compute inverse of right Cauchy-Green tensor.
    self.C_inv = np.linalg.inv(self.C)
    return

@register_method
def get_SPK(self, Parameters):
    # Compute second Piola-Kirchoff stress tensor.
    self.SPK = Parameters.mu*self.identity + np.einsum('..., ...IJ', Parameters.lambd*np.log(self.J) - Parameters.mu,\
                                                                     self.C_inv, dtype=np.float64)
    return

@register_method
def get_FPK(self):
    # Compute first Piola-Kirchoff stress tensor.
    self.FPK = np.einsum('...iI, ...JI', self.F, self.SPK, dtype=np.float64)
    return

@register_method
def get_Cauchy(self):
    # Compute Cauchy stress tensor.
    self.sigma = np.einsum('...iI,...jI,...', self.FPK, self.F, 1/self.J, dtype=np.float64)
    return

@register_method
def get_b(self):
    # Compute left Cauchy-Green tensor.
    self.b = np.einsum('...iI,...jI', self.F, self.F, dtype=np.float64)
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
    self.rho = np.einsum('..., ...I', self.J, self.rho_0, dtype=np.float64)
    return

@register_method
def get_rho_0(self, Parameters):
    # Compute mass density in reference configuration.
    # Should be constant for single-phase (i.e., classical continuum mechanics) materials.
    self.rho_0 = Parameters.rhoS_0*np.ones((8,3), dtype=np.float64)
    return