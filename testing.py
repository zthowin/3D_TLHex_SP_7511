import numpy as np
import classElement

class Parameters:

    def __init__(self):

        self.lambd = 2885
        self.mu = 1923

        self.ns = 0.01
        self.rhoS_0 = 1e3
        self.rho_0 = self.ns*self.rhoS_0

        self.grav = -9.81

        self.numDOF   = 4
        self.numEl    = 2
        self.numElDOF = 24

params = Parameters()

coordinates = np.zeros((2, 8, 3), dtype=np.float64)
coordinates[0,:,:] = np.array([[0.0,  0.0,  0.0],
                               [0.01, 0.0,  0.0],
                               [0.01, 0.01, 0.0],
                               [0.0,  0.01, 0.0],
                               [0.0,  0.0,  0.05],
                               [0.01, 0.0,  0.05],
                               [0.01, 0.01, 0.05],
                               [0.0,  0.01, 0.05]])
coordinates[1,:,:] = np.array([[0.0,  0.0,  0.05],
                               [0.01, 0.0,  0.05],
                               [0.01, 0.01, 0.05],
                               [0.0,  0.01, 0.05],
                               [0.0,  0.0,  0.1],
                               [0.01, 0.0,  0.1],
                               [0.01, 0.01, 0.1],
                               [0.0,  0.01, 0.1]])

LM       = np.ones((params.numElDOF, params.numEl), dtype=np.int32)
LM      *= -2
# Set the free DOFs
LM[1,1]  = 3
LM[5,1]  = 0
LM[8,1]  = 1
LM[11,1] = 2
LM[14,0] = 3
LM[17,0] = 0
LM[20,0] = 1
LM[23,0] = 2
# Set displacement BC DOFs
LM[14,1] = -1
LM[17,1] = -1
LM[20,1] = -1
LM[23,1] = -1

params.g_displ = -0.05

params.TStart = 0.0
params.TStop  = 1.0
params.dt     = 1/50
params.nsteps = np.round((params.TStop - params.TStart)/params.dt)
params.t      = params.TStart
params.n      = 0

params.t_ramp = 1.0

params.tolr = 1e-10
params.tola = 1e-8
params.kmax = 5

D      = np.zeros((params.numDOF), dtype=np.float64)
D_Last = np.zeros((params.numDOF), dtype=np.float64)

gd_n = 0
gd   = 0
g    = np.zeros((params.numElDOF, params.numEl), dtype=np.float64)
g_n  = np.zeros((params.numElDOF, params.numEl), dtype=np.float64)

while params.t < params.TStop:

    params.t += params.dt

    gd_n      = gd
    g_n[14,1] = gd_n
    g_n[17,1] = gd_n
    g_n[20,1] = gd_n
    g_n[23,1] = gd_n

    if params.t < params.t_ramp:
        gd = params.g_displ*(params.t/params.t_ramp)
    else:
        gd = params.g_displ

    g[14,1] = gd
    g[17,1] = gd
    g[20,1] = gd
    g[23,1] = gd

    Rtol = 1
    normR = 1
    k = 0
    D_Last[:] = D[:]

    while Rtol > params.tolr and normR > params.tola:

        k += 1
        if params.n == 0 and k == 1:
            del_d = np.zeros((params.numDOF), dtype=np.float64)
        else:
            del_d = np.linalg.solve(dR, -R)

        D      += del_d
        Delta_d = D - D_Last

        R  = np.zeros((params.numDOF), dtype=np.float64)
        dR = np.zeros((params.numDOF, params.numDOF), dtype=np.float64)

        for element_ID in range(params.numEl):

            element = classElement.Element(a_GaussOrder=2, a_ID=element_ID)
            element.set_Gauss_Points()
            element.set_Gauss_Weights()
            element.set_Coordinates(coordinates[element.ID,:,:])
            element.evaluate_Shape_Functions()
            element.get_Global_DOF(LM)
            element.set_Global_Solutions(D)
            element.apply_Local_BC(g)

            element.compute_variables(params)

            element.compute_forces(params)

            print(el.u_global)

            print(el.ID)