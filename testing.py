import sys
import numpy as np
import classElement

import matplotlib.pyplot as plt

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
        # self.constitutive_model = 'Saint Venant-Kirchhoff'
        self.constitutive_model = 'neo-Hookean'
        #-----------------------------------
        # Set boundary condition parameters.
        #-----------------------------------
        self.g_d          = 0.0
        self.tractionLoad = 0.0
        self.theta        = np.pi/2
        self.g_da         = 0.0
        self.g_db         = 0.0
        self.t_ramp       = 1.0
        #---------------------------------
        # Set time integration parameters.
        #---------------------------------
        self.TStart   = 0.0
        self.TStop    = 1.0
        self.numSteps = 20 # set to low value for stability reasons
        self.dt       = (self.TStop - self.TStart)/self.numSteps
        self.t        = self.TStart
        self.n        = 0
        #-------------------------------
        # Set Newton-Raphson parameters.
        # - Note: loose tolerances gives
        #         better performance.
        #-------------------------------
        self.tolr = 1e-6
        self.tola = 1e-8
        self.kmax = 20
        #------------------------
        # Set element properties.
        #------------------------
        self.numGauss = 8
        self.numDim   = 3
        self.numElDOF = 24

params = Parameters()

params.displacementProblem = False
params.tractionProblem     = False
params.rotationProblem     = True

if params.displacementProblem or params.tractionProblem:
    #---------------------------------------
    # 2 element problem, 1D uniaxial strain.
    #---------------------------------------
    params.numEl = 2
    if params.displacementProblem:
        params.numDOF = 4
    elif params.tractionProblem:
        params.numDOF = 8
    #-----------------------------
    # Create global coordinates:
    # 0.01m x 0.01m x 0.1m column.
    #-----------------------------
    coordinates = np.zeros((params.numEl, params.numGauss, params.numDim), dtype=params.float_dtype)
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
    #------------------------------
    # Create the 'location matrix'.
    #------------------------------
    LM       = np.ones((params.numElDOF, params.numEl), dtype=np.int32)
    LM      *= -1 # (this would be 0 in MATLAB)
    #----------------------------------------------
    # Set the free DOFs:
    #   - For the displacement problem, only the 
    #     middle 4 nodes are unconstrained in the 
    #     z-direction. For two elements, this 
    #     amounts to 8 local DOFs.
    #   - For the traction problem, the top 4 nodes
    #     are unconstrained in the z-direction. For
    #     two elements, this amounts to 12 local
    #     DOFs.
    #----------------------------------------------
    LM[2,1]  = 3
    LM[5,1]  = 0
    LM[8,1]  = 1
    LM[11,1] = 2
    LM[14,0] = 3
    LM[17,0] = 0
    LM[20,0] = 1
    LM[23,0] = 2
    if params.tractionProblem:
        LM[14,1] = 7
        LM[17,1] = 4
        LM[20,1] = 5
        LM[23,1] = 6

elif params.rotationProblem:
    #-------------------
    # 1 element problem.
    #-------------------
    params.numEl  = 1
    params.numDOF = 4
    #---------------------------
    # Create global coordinates:
    # 1m x 1m x 1m cube.
    #---------------------------
    coordinates = np.zeros((params.numEl, params.numGauss, params.numDim), dtype=params.float_dtype)
    coordinates[0,:,:] = np.array([[0.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [1.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [1.0, 0.0, 1.0],
                                   [1.0, 1.0, 1.0],
                                   [0.0, 1.0, 1.0]])
    #------------------------------
    # Create the 'location matrix'.
    #------------------------------
    LM       = np.ones((params.numElDOF, params.numEl), dtype=np.int32)
    LM      *= -1 # (this would be 0 in MATLAB)
    #-------------------------------------------
    # Set the free DOFs:
    #   - Only the top 4 nodes are unconstrained
    #     in the z-direction.
    #-------------------------------------------
    LM[14] = 3
    LM[17] = 0
    LM[20] = 1
    LM[23] = 2

#----------------------------
# Set the initial conditions.
#----------------------------
D = np.zeros((params.numDOF), dtype=params.float_dtype)
g = np.zeros((params.numElDOF, params.numEl), dtype=params.float_dtype)
#-----------------------------
# Set storage arrays.
#--------------------
params.numStressStrain = 6 # S, P, sig, E, e, hencky
params.numISV          = 2 # mean sig, von Mises stress

DSolve       = np.zeros((params.numSteps+1, params.numDOF), dtype=params.float_dtype)
isv_solve    = np.zeros((params.numSteps+1,params.numEl,params.numGauss,params.numISV), dtype=params.float_dtype)
stress_solve = np.zeros((params.numSteps+1,params.numEl,params.numGauss,params.numDim,params.numDim,params.numStressStrain), dtype=params.float_dtype)
#------------------
# Begin simulation.
#------------------
print("Solving...")
while params.t < params.TStop:

    params.t += params.dt
    params.n += 1
    print("t = %.2f seconds" %params.t)
    print(params.n)

    if params.t < params.t_ramp:
        gd = params.g_displ*(params.t/params.t_ramp)
        params.tract = params.traction*(params.t/params.t_ramp)
    else:
        gd = params.g_displ
        params.tract = params.traction

    # g[14,1] = gd
    # g[17,1] = gd
    # g[20,1] = gd
    # g[23,1] = gd
    theta = np.pi/2
    da    = d_a*params.t
    db    = d_b*params.t

    # x_rot1 = -np.sin(theta)*np.tan(theta/2) + da*np.cos(theta)
    # y_rot1 = np.sin(theta) + da*np.sin(theta)
    # z_rot1
    g[3] = -np.sin(theta)*np.tan(theta/2) + da*np.cos(theta)
    g[4] = np.sin(theta) + da*np.sin(theta)
    g[6] = -(np.sqrt(2)*np.sin(theta)/np.cos(theta/2))*np.cos(np.pi/4 - theta/2) + da*np.cos(theta) - db*np.sin(theta)
    g[7] = (np.sqrt(2)*np.sin(theta)/np.cos(theta/2))*np.sin(np.pi/4 - theta/2)  + da*np.sin(theta) + db*np.cos(theta)
    g[9] = -np.sin(theta) - db*np.sin(theta)
    g[10] = -np.sin(theta)*np.tan(theta/2) + db*np.cos(theta)
    g[15] = -np.sin(theta)*np.tan(theta/2) + da*np.cos(theta)
    g[16] = np.sin(theta) + da*np.sin(theta)
    g[18] = (-np.sqrt(2)*np.sin(theta)/np.cos(theta/2))*np.cos(np.pi/4 - theta/2) + da*np.cos(theta) - db*np.sin(theta)
    g[19] = (np.sqrt(2)*np.sin(theta)/np.cos(theta/2))*np.sin(np.pi/4 - theta/2)  + da*np.sin(theta) + db*np.cos(theta)
    g[21] = -np.sin(theta) - db*np.sin(theta)
    g[22] = -np.sin(theta)*np.tan(theta/2) + db*np.cos(theta)

    Rtol = 1
    normR = 1
    k = 0

    while Rtol > params.tolr and normR > params.tola:

        k += 1
        if params.n == 1 and k == 1:
            del_d = np.zeros((params.numDOF), dtype=params.float_dtype)
        else:
            del_d = np.linalg.solve(dR, -R)

        D += del_d

        R  = np.zeros((params.numDOF), dtype=params.float_dtype)
        dR = np.zeros((params.numDOF, params.numDOF), dtype=params.float_dtype)

        for element_ID in range(params.numEl):
            #------------------------
            # Initialize the element.
            #------------------------
            element = classElement.Element(a_GaussOrder=2, a_ID=element_ID)
            element.set_Gauss_Points(params)
            element.set_Gauss_Weights(params)
            element.set_Coordinates(coordinates[element.ID,:,:])
            element.evaluate_Shape_Functions(params)
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

            isv_solve[params.n,element.ID,:,0] = element.sigma_mean
            isv_solve[params.n,element.ID,:,1] = element.von_mises
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

                    for j in range(element.numDOF):
                        J = element.DOF[j]

                        if J > -1:
                            dR[I,J] += element.G_Mtx[i,j]

        if k == 1:
            R0 = R
        # print(dR)
        # input()
        Rtol  = np.linalg.norm(R)/np.linalg.norm(R0)
        normR = np.linalg.norm(R)
        print("Rtol", Rtol)
        # print("normR", normR)
        if k > params.kmax:
            print(Rtol)
            print(normR)
            sys.exit("ERROR. Reached max number of iterations.")

plt.figure(1)
# plt.plot(-stress_solve[:,0,0,2,2,3],-stress_solve[:,0,0,2,2,0]*1e-3, 'k+-', label=r'-$S_{33}$ vs. -$E_{33}$', fillstyle='none')
# plt.plot(-stress_solve[:,0,0,2,2,4],-stress_solve[:,0,0,2,2,2]*1e-3, 'ko-', label=r'-$\sigma_{33}$ vs. -$e_{33}$', fillstyle='none')
# plt.plot(-stress_solve[:,0,0,2,2,5],-stress_solve[:,0,0,2,2,2]*1e-3, 'ks-', label=r'-$\sigma_{33}$ vs. -$h_{33}$', fillstyle='none')
stress_solve[1:,0,0,:,:,5][~np.isfinite(stress_solve[1:,0,0,:,:,5])] = 0
# print(stress_solve[1:,0,0,:,:,5])
minStress = np.min(np.linalg.eig(stress_solve[1:,0,0,:,:,2])[0], axis=1)
minStrain = np.min(np.linalg.eig(stress_solve[1:,0,0,:,:,5])[0], axis=1)
plt.plot(minStrain[0:], -minStress[0:,]*1e-3)
plt.legend()
# plt.xlim([0, 0.2])
# plt.ylim([0, 1.2])
plt.grid()
plt.ylabel('-Stress (kPa)')
plt.xlabel('-Strain (m/m)')
plt.show()
