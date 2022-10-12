#----------------------------------------------------------------------------------------
# Module housing element object variables.
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
def compute_variables(self, Parameters):
    # Compute variables related to strains and stresses.
    self.get_dudX(Parameters)
    self.get_F(Parameters)
    self.get_J()
    self.get_F_inv()
    self.get_C(Parameters)
    self.get_C_inv()
    self.get_E()
    self.get_SPK(Parameters)
    self.get_FPK(Parameters)
    self.get_b(Parameters)
    self.get_v()
    self.get_e()
    self.get_Hencky()
    self.get_Cauchy(Parameters)
    self.get_mean_Cauchy(Parameters)
    self.get_von_Mises(Parameters)
    self.get_rho_0(Parameters)
    self.get_rho(Parameters)
    return

@register_method
def get_dudX(self, Parameters):
    # Compute solid displacement gradient.
    self.dudX = np.einsum('...ij, j -> ...i', self.Bu, self.u_global, dtype=Parameters.float_dtype)
    return

@register_method
def get_F(self, Parameters):
    # Compute deformation gradient.
    #----------------------------------------------------
    # Reshape the identity matrix for all 8 Gauss points.
    #----------------------------------------------------
    self.identity = np.zeros((Parameters.numGauss,Parameters.numDim,Parameters.numDim), dtype=Parameters.float_dtype)
    np.einsum('ijj -> ij', self.identity)[:] = 1
    #-------------------------------------------------------
    # Create the 3x3 deformation matrix from the 9x1 vector.
    #-------------------------------------------------------
    self.dudX_mat = np.zeros((Parameters.numGauss,Parameters.numDim,Parameters.numDim), dtype=Parameters.float_dtype)
    for i in range(Parameters.numDim*Parameters.numDim):
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
        self.dudX_mat[:,alpha,beta] = self.dudX[:,i]

    self.F = self.identity + self.dudX_mat
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
def get_C(self, Parameters):
    # Compute right Cauchy-Green tensor.
    self.C = np.einsum('...iI, ...iJ -> ...IJ', self.F, self.F, dtype=Parameters.float_dtype)
    return

@register_method
def get_C_inv(self):
    # Compute inverse of right Cauchy-Green tensor.
    self.C_inv = np.linalg.inv(self.C)
    return

@register_method
def get_SPK(self, Parameters):
    # Compute second Piola-Kirchoff stress tensor.
    if Parameters.constitutive_model == 'neo-Hookean':
        self.SPK = Parameters.mu*self.identity + np.einsum('..., ...IJ -> ...IJ',\
                                                           Parameters.lambd*np.log(self.J) - Parameters.mu,\
                                                           self.C_inv, dtype=Parameters.float_dtype)
    elif Parameters.constitutive_model == 'Saint Venant-Kirchhoff':
        self.SPK = Parameters.lambd*np.einsum('...KK, ...IJ -> ...IJ', self.E, self.identity, dtype=Parameters.float_dtype)\
                   + 2*Parameters.mu*self.E
    else:
        sys.exit("ERROR. Constitutive model not recognized, check inputs.")
    return

@register_method
def get_FPK(self, Parameters):
    # Compute first Piola-Kirchoff stress tensor.
    self.FPK = np.einsum('...iI, ...IJ -> ...iI', self.F, self.SPK, dtype=Parameters.float_dtype)
    return

@register_method
def get_Cauchy(self, Parameters):
    # Compute Cauchy stress tensor.
    self.sigma = np.einsum('...iI, ...jI, ... -> ...ij', self.FPK, self.F, 1/self.J, dtype=Parameters.float_dtype)
    return

@register_method
def get_mean_Cauchy(self, Parameters):
    # Compute the mean Cauchy stress, i.e., thermodynamic pressure.
    self.sigma_mean = (1/3)*np.einsum('...ii', self.sigma, dtype=Parameters.float_dtype)
    return

@register_method
def get_von_Mises(self, Parameters):
    # Compute the von Mises stress.
    self.von_mises = np.sqrt(3/2)*np.linalg.norm((self.sigma - np.einsum('..., ...ij -> ...ij', self.sigma_mean, self.identity, dtype=Parameters.float_dtype)))
    return

@register_method
def get_b(self, Parameters):
    # Compute left Cauchy-Green tensor.
    self.b = np.einsum('...iI, ...jI -> ...ij', self.F, self.F, dtype=Parameters.float_dtype)
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
def get_rho(self, Parameters):
    # Compute mass density in current configuration.
    self.rho = np.einsum('..., ...i -> ...i', self.J, self.rho_0, dtype=Parameters.float_dtype)
    return

@register_method
def get_rho_0(self, Parameters):
    # Compute mass density in reference configuration.
    self.rho_0 = Parameters.rho_0*np.ones((8,3), dtype=Parameters.float_dtype)
    return
