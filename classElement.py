#----------------------------------------------------------------------------------------
# Module housing top-level element class.
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

try:
  import _ElementVariables
except ImportError:
  sys.exit("MODULE WARNING. '_ElementVariables.py' not found, check configuration.")

try:
  import _ElementForces
except ImportError:
  sys.exit("MODULE WARNING. '_ElementForces.py' not found, check configuration.")

try:
  import _ElementTangents
except ImportError:
  sys.exit("MODULE WARNING. '_ElementTangents.py' not found, check configuration.")

@Lib.add_methods_from(_ElementVariables, _ElementForces, _ElementTangents)

class Element:
    
    def __init__(self, a_GaussOrder=None, a_ID=None):
        # Set Gauss quadrature order.
        self.set_Gauss_Order(a_GaussOrder)
        # Set element ID.
        self.set_Element_ID(a_ID)
        return
    
    def set_Gauss_Order(self, a_Order):
        # Initialize the gauss quadrature order.
        self.Gauss_Order = a_Order
        return

    def set_Element_ID(self, a_ID):
        # Initialize the element number.
        self.ID = a_ID
        return

    def set_Gauss_Points(self):
        # Initialize the Gauss quadrature points.
        if self.Gauss_Order == 2:
            const = 1/np.sqrt(3)
            self.points  = np.array([[-const, -const, -const],
                                     [const, -const, -const],
                                     [const, const, -const],
                                     [-const,const, -const],
                                     [-const, -const, const],
                                     [const, -const, const],
                                     [const, const, const],
                                     [-const, const, const]],
                                    dtype=np.float64)
        else:
            sys.exit("ERROR. Only trilinear elements have been implemented; check quadrature order.")
        return

    def set_Gauss_Weights(self):
        # Initialize the Gauss quadrature weights.
        if self.Gauss_Order == 2:
            self.weights = np.ones(8, dtype=np.float64)
        else:
            sys.exit("ERROR. Only trilinear elements have been implemented; check quadrature order.")
        return

    def set_Coordinates(self, a_Coords):
        # Initialize element coordinates.
        self.coordinates = a_Coords
        return

    def evaluate_Shape_Functions(self):
        # Initialize the shape functions used for interpolation.
        # Calculations are vectorized across the 8 Gauss points.
        
        #--------------------------------
        # Grab local element coordinates.
        #--------------------------------
        self.xi   = self.points[:,0]
        self.eta  = self.points[:,1]
        self.zeta = self.points[:,2]
        
        #---------
        # Set N_a.
        #---------
        self.N1 = (1 - self.xi)*(1 - self.eta)*(1 - self.zeta)/8
        self.N2 = (1 + self.xi)*(1 + self.eta)*(1 - self.zeta)/8
        self.N3 = (1 + self.xi)*(1 + self.eta)*(1 - self.zeta)/8
        self.N4 = (1 - self.xi)*(1 - self.eta)*(1 - self.zeta)/8
        self.N5 = (1 - self.xi)*(1 - self.eta)*(1 + self.zeta)/8
        self.N6 = (1 + self.xi)*(1 - self.eta)*(1 + self.zeta)/8
        self.N7 = (1 + self.xi)*(1 + self.eta)*(1 + self.zeta)/8
        self.N8 = (1 - self.xi)*(1 + self.eta)*(1 + self.zeta)/8
        
        #-----------------------------
        # Build shape function matrix.
        #-----------------------------
        self.Nu = np.zeros((8, 3, 24), dtype=np.float64)
        for i in range(3):
            self.Nu[:, i, 0 + i]  = self.N1
            self.Nu[:, i, 3 + i]  = self.N2
            self.Nu[:, i, 6 + i]  = self.N3
            self.Nu[:, i, 9 + i]  = self.N4
            self.Nu[:, i, 12 + i] = self.N5
            self.Nu[:, i, 15 + i] = self.N6
            self.Nu[:, i, 18 + i] = self.N7
            self.Nu[:, i, 21 + i] = self.N8
        
        #----------------------------------
        # Calculate derivatives w.r.t. \xi.
        #----------------------------------
        self.dN1_dxi = -(1/8)*(1 - self.eta)*(1 - self.zeta)
        self.dN2_dxi = -self.dN1_dxi
        self.dN3_dxi = (1/8)*(1 + self.eta)*(1 - self.zeta)
        self.dN4_dxi = -self.dN3_dxi
        self.dN5_dxi = -(1/8)*(1 - self.eta)*(1 + self.zeta)
        self.dN6_dxi = -self.dN5_dxi
        self.dN7_dxi = (1/8)*(1 + self.eta)*(1 + self.zeta)
        self.dN8_dxi = -self.dN7_dxi
        
        self.dN_dxi      = np.zeros((8,8), dtype=np.float64)
        self.dN_dxi[:,0] = self.dN1_dxi
        self.dN_dxi[:,1] = self.dN2_dxi
        self.dN_dxi[:,2] = self.dN3_dxi
        self.dN_dxi[:,3] = self.dN4_dxi
        self.dN_dxi[:,4] = self.dN5_dxi
        self.dN_dxi[:,5] = self.dN6_dxi
        self.dN_dxi[:,6] = self.dN7_dxi
        self.dN_dxi[:,7] = self.dN8_dxi

        #-----------------------------------
        # Calculate derivatives w.r.t. \eta.
        #-----------------------------------
        self.dN1_deta = -(1/8)*(1 - self.xi)*(1 - self.zeta)
        self.dN2_deta = -(1/8)*(1 + self.xi)*(1 - self.zeta)
        self.dN3_deta = -self.dN2_deta
        self.dN4_deta = -self.dN1_deta
        self.dN5_deta = -(1/8)*(1 - self.xi)*(1 + self.zeta)
        self.dN6_deta = -(1/8)*(1 + self.xi)*(1 + self.zeta)
        self.dN7_deta = -self.dN6_deta
        self.dN8_deta = -self.dN5_deta
        
        self.dN_deta      = np.zeros((8,8), dtype=np.float64)
        self.dN_deta[:,0] = self.dN1_deta
        self.dN_deta[:,1] = self.dN2_deta
        self.dN_deta[:,2] = self.dN3_deta
        self.dN_deta[:,3] = self.dN4_deta
        self.dN_deta[:,4] = self.dN5_deta
        self.dN_deta[:,5] = self.dN6_deta
        self.dN_deta[:,6] = self.dN7_deta
        self.dN_deta[:,7] = self.dN8_deta
        
        #------------------------------------
        # Calculate derivatives w.r.t. \zeta.
        #------------------------------------
        self.dN1_dzeta = -(1/8)*(1 - self.xi)*(1 - self.eta)
        self.dN2_dzeta = -(1/8)*(1 + self.xi)*(1 - self.eta)
        self.dN3_dzeta = -(1/8)*(1 + self.xi)*(1 + self.eta)
        self.dN4_dzeta = -(1/8)*(1 - self.xi)*(1 + self.eta)
        self.dN5_dzeta = -self.dN1_dzeta
        self.dN6_dzeta = -self.dN2_dzeta
        self.dN7_dzeta = -self.dN3_dzeta
        self.dN8_dzeta = -self.dN4_dzeta
        
        self.dN_dzeta      = np.zeros((8,8), dtype=np.float64)
        self.dN_dzeta[:,0] = self.dN1_dzeta
        self.dN_dzeta[:,1] = self.dN2_dzeta
        self.dN_dzeta[:,2] = self.dN3_dzeta
        self.dN_dzeta[:,3] = self.dN4_dzeta
        self.dN_dzeta[:,4] = self.dN5_dzeta
        self.dN_dzeta[:,5] = self.dN6_dzeta
        self.dN_dzeta[:,6] = self.dN7_dzeta
        self.dN_dzeta[:,7] = self.dN8_dzeta
        
        #-------------------
        # Compute jacobians.
        #-------------------
        self.get_Jacobian()
        
        #----------------------------------
        # Compute shape function gradients.
        #----------------------------------
        # print(self.Jeinv.shape, np.array([self.dN1_dxi, self.dN1_deta, self.dN1_dzeta]).T.shape)
        self.dN1_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN1_dxi, self.dN1_deta, self.dN1_dzeta]).T)
        self.dN2_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN2_dxi, self.dN2_deta, self.dN2_dzeta]).T)
        self.dN3_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN3_dxi, self.dN3_deta, self.dN3_dzeta]).T)
        self.dN4_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN4_dxi, self.dN4_deta, self.dN4_dzeta]).T)
        self.dN5_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN5_dxi, self.dN5_deta, self.dN5_dzeta]).T)
        self.dN6_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN6_dxi, self.dN6_deta, self.dN6_dzeta]).T)
        self.dN7_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN7_dxi, self.dN7_deta, self.dN7_dzeta]).T)
        self.dN8_dx = np.einsum('...ij, ...j', self.Jeinv, np.array([self.dN8_dxi, self.dN8_deta, self.dN8_dzeta]).T)

        #--------------------------------------
        # Construct strain-displacement matrix.
        #--------------------------------------
        self.Bu = np.zeros((8, 9, 24), dtype=np.float64)

        for i in range(3):
            self.Bu[:, i, 0]  = self.dN1_dx[:,i]
            self.Bu[:, i, 3]  = self.dN2_dx[:,i]
            self.Bu[:, i, 6]  = self.dN3_dx[:,i]
            self.Bu[:, i, 9]  = self.dN4_dx[:,i]
            self.Bu[:, i, 12] = self.dN5_dx[:,i]
            self.Bu[:, i, 15] = self.dN6_dx[:,i]
            self.Bu[:, i, 18] = self.dN7_dx[:,i]
            self.Bu[:, i, 21] = self.dN8_dx[:,i]

        for i in range(3,6):
            self.Bu[:, i, 1]  = self.dN1_dx[:,i-3]
            self.Bu[:, i, 4]  = self.dN2_dx[:,i-3]
            self.Bu[:, i, 7]  = self.dN3_dx[:,i-3]
            self.Bu[:, i, 10]  = self.dN4_dx[:,i-3]
            self.Bu[:, i, 13] = self.dN5_dx[:,i-3]
            self.Bu[:, i, 16] = self.dN6_dx[:,i-3]
            self.Bu[:, i, 19] = self.dN7_dx[:,i-3]
            self.Bu[:, i, 22] = self.dN8_dx[:,i-3]

        for i in range(6,9):
            self.Bu[:, i, 2]  = self.dN1_dx[:,i-6]
            self.Bu[:, i, 5]  = self.dN2_dx[:,i-6]
            self.Bu[:, i, 8]  = self.dN3_dx[:,i-6]
            self.Bu[:, i, 11] = self.dN4_dx[:,i-6]
            self.Bu[:, i, 14] = self.dN5_dx[:,i-6]
            self.Bu[:, i, 17] = self.dN6_dx[:,i-6]
            self.Bu[:, i, 20] = self.dN7_dx[:,i-6]
            self.Bu[:, i, 23] = self.dN8_dx[:,i-6]
        
        return
    
    def get_Jacobian(self):
        # Compute the element Jacobian.
        self.dx_dxi   = np.einsum('ik, k', self.dN_dxi, self.coordinates[:,0])
        self.dx_deta  = np.einsum('ik, k', self.dN_deta, self.coordinates[:,0])
        self.dx_dzeta = np.einsum('ik, k', self.dN_dzeta, self.coordinates[:,0])
        
        self.dy_dxi   = np.einsum('ik, k', self.dN_dxi, self.coordinates[:,1])
        self.dy_deta  = np.einsum('ik, k', self.dN_deta, self.coordinates[:,1])
        self.dy_dzeta = np.einsum('ik, k', self.dN_dzeta, self.coordinates[:,1])
        
        self.dz_dxi   = np.einsum('ik, k', self.dN_dxi, self.coordinates[:,2])
        self.dz_deta  = np.einsum('ik, k', self.dN_deta, self.coordinates[:,2])
        self.dz_dzeta = np.einsum('ik, k', self.dN_dzeta, self.coordinates[:,2])
                
        self.Je        = np.zeros((8,3,3),dtype=np.float64)
        self.Je[:,0,0] = self.dx_dxi
        self.Je[:,0,1] = self.dx_deta
        self.Je[:,0,2] = self.dx_dzeta
        self.Je[:,1,0] = self.dy_dxi
        self.Je[:,1,1] = self.dy_deta
        self.Je[:,1,2] = self.dy_dzeta
        self.Je[:,2,0] = self.dz_dxi
        self.Je[:,2,1] = self.dz_deta
        self.Je[:,2,2] = self.dz_dzeta
        
        self.j     = np.zeros(8, dtype=np.float64)
        self.Jeinv = np.zeros((8,3,3), dtype=np.float64)

        for i in range(8):
            self.j[i]          = np.linalg.det(self.Je[i,:,:])
            self.Jeinv[i,:,:,] = np.linalg.inv(self.Je[i,:,:])
        
        return

    def get_Global_DOF(self, a_LM):
        # Set the global degrees of freedom of this element.
        self.DOF    = a_LM[:,self.ID]
        self.numDOF = self.DOF.shape[0]
        return

    def set_Global_Solutions(self, a_D):
        # Set the local solution variables at the current time step.
        self.set_u_global(a_D[self.DOF[0:24]])
        return

    def apply_Local_BC(self, a_g):
        # Apply boundary conditions at the element scale.
        if np.any(self.DOF < 0):
            idxs = np.where((self.DOF == -1))[0]
            for idx in idxs:
                self.u_global[idx] = a_g[idx, self.ID]
        return

    def set_u_global(self, a_D):
        # Initialize the global solid displacement (at element level).
        self.u_global = a_D
        return
