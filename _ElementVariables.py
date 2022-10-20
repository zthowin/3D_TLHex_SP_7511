#----------------------------------------------------------------------------------------
# Module housing element object variables.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 20, 2022
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
    if Parameters.finiteStrain:
        self.get_F_inv()
        self.get_C(Parameters)
        self.get_C_inv()
        self.get_E()
        self.get_SPK(Parameters)
        self.get_FPK(Parameters)
        self.get_b(Parameters)
        self.get_v()
        self.get_e()
        self.get_Hencky(Parameters)
    elif Parameters.smallStrain:
        self.get_eps(Parameters)
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
    np.einsum('...ii -> ...i', self.identity)[:] = 1
    #-------------------------------------------------------
    # Create the 3x3 deformation matrix from the 9x1 vector.
    #-------------------------------------------------------
    self.dudX_mat = np.zeros((Parameters.numGauss,Parameters.numDim,Parameters.numDim), dtype=Parameters.float_dtype)
    if Parameters.finiteStrain:
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
            self.dudX_mat[:,i,I] = self.dudX[:,alpha]
    #-------------------------------------------------------
    # Create the 3x3 deformation matrix from the 6x1 vector.
    #-------------------------------------------------------       
    elif Parameters.smallStrain:
        for alpha in range(Parameters.numDim*2):
            if alpha == 0:
                i = 0
                I = 0
            elif alpha == 1:
                i = 1
                I = 1
            elif alpha == 2:
                i = 2
                I = 2
            elif alpha == 3:
                i = 1
                I = 2
            elif alpha == 4:
                i = 0
                I = 2
            elif alpha == 5:
                i = 0
                I = 1
            self.dudX_mat[:,i,I] = self.dudX[:,alpha]
        self.dudX_mat[:,2,1] = self.dudX_mat[:,1,2]
        self.dudX_mat[:,2,0] = self.dudX_mat[:,0,2]
        self.dudX_mat[:,1,0] = self.dudX_mat[:,0,1]

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
def get_eps(self, Parameters):
    # Compute the small strain tensor.
    self.eps = 0.5*(self.F + np.transpose(self.F, axes=[0,2,1])) - self.identity
    return

@register_method
def get_Cauchy(self, Parameters):
    # Compute Cauchy stress tensor.
    if Parameters.finiteStrain:
        self.sigma = np.einsum('...iI, ...jI, ... -> ...ij', self.FPK, self.F, 1/self.J, dtype=Parameters.float_dtype)
    elif Parameters.smallStrain:
        self.eps_v = np.einsum('...ii', self.eps, dtype=Parameters.float_dtype)
        if Parameters.linearElasticity:
            self.sigma = Parameters.lambd*np.einsum('..., ...ij -> ...ij', self.eps_v, self.identity)\
                         + 2*Parameters.mu*self.eps
        elif Parameters.nonlinearElasticity:
            self.sigma = Parameters.lambd*np.einsum('..., ...ij -> ...ij', np.log(1 + self.eps_v), self.identity)\
                         + 2*Parameters.mu*self.eps
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
def get_Hencky(self, Parameters):
    # Compute Hencky strain.
    #-----------------------------------------------------
    # Get eigenvalues, eigenvectors of left Cauchy-Green.
    # Motivated by polar decomposition, Holzapfel Eq. 2.93
    #-----------------------------------------------------
    self.b_w, self.b_v = np.linalg.eig(self.b)
    #------------------------------------
    # Compute the log of the \lambda_a's.
    # Holzapfel Eq. 2.122
    #------------------------------------
    self.log_principle_stretch = np.log(np.sqrt(self.b_w))
    #--------------------------------------------------
    # Place the principle stretches along the diagonal.
    #--------------------------------------------------
    self.Hencky_rotated = np.zeros((Parameters.numGauss,Parameters.numDim,Parameters.numDim), dtype=Parameters.float_dtype)
    np.einsum('...ii -> ...i', self.Hencky_rotated)[:] = self.log_principle_stretch
    #------------------------------------------------------------
    # Rotate Hencky strain back to the cartesian reference frame
    # using diadic(n_a, n_a) ( = b_v, b_v)
    # Holzapfel Eq. 2.108
    #------------------------------------------------------------
    self.Hencky = np.einsum('...ik, ...kl, ...jl -> ...ij', self.b_v, self.Hencky_rotated, self.b_v)
    return

@register_method
def get_rho(self, Parameters):
    # Compute mass density in current configuration.
    self.rho = np.einsum('..., ...i -> ...i', self.J, self.rho_0, dtype=Parameters.float_dtype)
    return

@register_method
def get_rho_0(self, Parameters):
    # Compute mass density in reference configuration.
    self.rho_0 = Parameters.rho_0*np.ones((Parameters.numGauss,Parameters.numDim), dtype=Parameters.float_dtype)
    return
