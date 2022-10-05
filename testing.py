import sys
import numpy as np
import classElement

import matplotlib.pyplot as plt

class Parameters:

    def __init__(self):

        self.lambd = 2885
        self.mu = 1923

        self.ns = 0.01
        self.rhoS_0 = 1e3
        self.rho_0 = self.ns*self.rhoS_0

        self.grav = 9.81
        # self.grav = 0

        self.numDOF   = 8
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
LM      *= -1
# Set the free DOFs
LM[2,1]  = 3
LM[5,1]  = 0
LM[8,1]  = 1
LM[11,1] = 2
LM[14,0] = 3
LM[17,0] = 0
LM[20,0] = 1
LM[23,0] = 2
# Uncomment for traction
LM[14,1] = 7
LM[17,1] = 4
LM[20,1] = 5
LM[23,1] = 6

params.g_displ = -0.0
params.traction = 1e4

params.TStart = 0.0
params.TStop  = 1.0
params.dt     = 1/20
params.nsteps = int(np.round((params.TStop - params.TStart)/params.dt))
params.t      = params.TStart
params.n      = 0

params.t_ramp = 1.0

params.tolr = 1e-8
params.tola = 1e-6
params.kmax = 5

D      = np.zeros((params.numDOF), dtype=np.float64)

DSolve = np.zeros((params.nsteps+1, params.numDOF), dtype=np.float64)
isv_solve = np.zeros((params.nsteps+1,params.numEl,8,3,3,2), dtype=np.float64)
stress_solve = np.zeros((params.nsteps+1,params.numEl,8,3,3,6), dtype=np.float64)

gd_n = 0
gd   = 0
g    = np.zeros((params.numElDOF, params.numEl), dtype=np.float64)
g_n  = np.zeros((params.numElDOF, params.numEl), dtype=np.float64)

print("Solving...")
while params.t < params.TStop:

    params.t += params.dt
    params.n += 1
    print("t = %.2f seconds" %params.t)
    print(params.n)

    gd_n      = gd
    g_n[14,1] = gd_n
    g_n[17,1] = gd_n
    g_n[20,1] = gd_n
    g_n[23,1] = gd_n

    if params.t < params.t_ramp:
        # gd = params.g_displ*(params.t/params.t_ramp)
        gd = 0
        params.tract = params.traction*(params.t/params.t_ramp)
    else:
        # gd = params.g_displ
        gd = 0
        params.tract = params.traction

    g[14,1] = gd
    g[17,1] = gd
    g[20,1] = gd
    g[23,1] = gd

    Rtol = 1
    normR = 1
    k = 0

    while Rtol > params.tolr and normR > params.tola:

        k += 1
        if params.n == 1 and k == 1:
            del_d = np.zeros((params.numDOF), dtype=np.float64)
        else:
            del_d = np.linalg.solve(dR, -R)

        D += del_d

        R  = np.zeros((params.numDOF), dtype=np.float64)
        dR = np.zeros((params.numDOF, params.numDOF), dtype=np.float64)

        for element_ID in range(params.numEl):
            #------------------------
            # Initialize the element.
            #------------------------
            element = classElement.Element(a_GaussOrder=2, a_ID=element_ID)
            element.set_Gauss_Points()
            element.set_Gauss_Weights()
            element.set_Coordinates(coordinates[element.ID,:,:])
            element.evaluate_Shape_Functions()
            element.get_Global_DOF(LM)
            element.set_Global_Solutions(D)
            element.apply_Local_BC(g)
            #------------------------------
            # Compute stresses and strains.
            #------------------------------
            element.compute_variables(params)
            #---------------------------
            # Save stresses and strains.
            #---------------------------
            stress_solve[params.n,element.ID,:,:,:,0] = element.SPK
            stress_solve[params.n,element.ID,:,:,:,1] = element.FPK
            stress_solve[params.n,element.ID,:,:,:,2] = element.sigma
            stress_solve[params.n,element.ID,:,:,:,3] = element.E
            stress_solve[params.n,element.ID,:,:,:,4] = element.e
            stress_solve[params.n,element.ID,:,:,:,5] = element.Hencky

            # isv_solve[n,:,:,0,:] = np.trace(element.sigma)/3
            # isv_solve[n,:,:,1,:] = np.sqrt(3/2)*np.linalg.norm(element.sigma - np.trace(element.sigma)/3*element.identity
            #--------------------------------
            # Compute internal force vectors.
            #--------------------------------
            element.compute_forces(params)
            #-----------------------------------
            # Compute the consistent tangent(s).
            #-----------------------------------
            element.compute_tangents(params)
            #--------------------------
            # Perform element assembly.
            #--------------------------
            for i in range(element.numDOF):
                I = element.DOF[i]

                if I > -1:
                    R[I] += element.G_int[i]
                    if element.ID == 1:
                        R[I] -= element.G_ext[i]

                    for j in range(element.numDOF):
                        J = element.DOF[j]

                        if J > -1:
                            dR[I,J] += element.G_Mtx[i,j]

        if k == 1:
            R0 = R
        Rtol = np.linalg.norm(R)/np.linalg.norm(R0)
        normR = np.linalg.norm(R)

        if k > params.kmax:
            print(Rtol)
            print(normR)
            sys.exit("ERROR. Reached max number of iterations.")

plt.figure(1)
plt.plot(-stress_solve[2:,0,0,2,2,3],-stress_solve[2:,0,0,2,2,0]*1e-3, label=r'-$S_{33}$ vs. -$E_{33}$')
plt.plot(-stress_solve[2:,0,0,2,2,4],-stress_solve[2:,0,0,2,2,2]*1e-3, label=r'-$s_{33}$ vs. -$e_{33}$')
plt.plot(-stress_solve[2:,0,0,2,2,5],-stress_solve[2:,0,0,2,2,2]*1e-3, label=r'-$s_{33}$ vs. -$h_{33}$')
plt.legend()
plt.show()
