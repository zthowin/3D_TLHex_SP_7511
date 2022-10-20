#----------------------------------------------------------------------------------------
# Main script to run 3D trilinear hex Python code.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 19, 2022
#----------------------------------------------------------------------------------------
import sys

try:
    import numpy as np
except ImportError:
    sys.exit("MODULE WARNING. NumPy not installed.")

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("MODULE WARNING. Matplotlib not installed.")

try:
    import solver_u
except ImportError:
    sys.exit("MODULE WARNING. solver_u not found, check present working directory.")

class Parameters:

    def __init__(self):
        self.float_dtype = np.float64 # single-precision float32 can lead to issues
        #-------------------------
        # Set material parameters.
        #-------------------------
        self.lambd = 2885
        self.mu    = 1923

        self.ns     = 0.01
        self.rhoS_0 = 1000
        self.rho_0  = self.ns*self.rhoS_0

        self.grav = 9.81
        #------------------------
        # Set constitutive model.
        #------------------------
        # self.constitutive_model = 'Saint Venant-Kirchhoff' # not working, removed
        self.constitutive_model = 'neo-Hookean'
        #-----------------------------------
        # Set boundary condition parameters.
        #-----------------------------------
        self.g_displ      = -0.05
        self.tractionLoad = 1e4
        self.theta        = np.pi/2
        self.g_da         = -0.4
        self.g_db         = -0.2
        self.t_ramp       = 1.0
        #---------------------------------
        # Set time integration parameters.
        #---------------------------------
        self.TStart   = 0.0
        self.TStop    = 1.0
        self.numSteps = 10 # set to low value for stability reasons
        self.dt       = (self.TStop - self.TStart)/self.numSteps
        self.t        = self.TStart
        self.n        = 0
        #-------------------------------
        # Set Newton-Raphson parameters.
        # - Note: loose tolerances gives
        #         better performance.
        #-------------------------------
        self.tolr = 1e-6
        self.tola = 1e-6
        self.kmax = 5
        #------------------------
        # Set element properties.
        #------------------------
        self.GaussOrder = 2
        self.numGauss   = 8
        self.numDim     = 3
        self.numElDOF   = int(self.numGauss*self.numDim)

params = Parameters()

#------------------
# You edit these...
#------------------
params.displacementProblem = False
params.tractionProblem     = True
params.rotationProblem     = False

params.finiteStrain = False
params.smallStrain  = True # if this comment is here, 
                            # then small strain is not yet implemented.
                            # check Canvas tomorrow by end of day 10/20/2022
                            # for update.
#------------------
#--------------------
# Call solver script.
#--------------------
solver_u.main(params,printTol=False)
