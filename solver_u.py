#----------------------------------------------------------------------------------------
# Module housing 3D hexahedral element solver.
#
# Author:       Zachariah Irwin
# Institution:  University of Colorado Boulder
# Last Edit:    October 20, 2022
#----------------------------------------------------------------------------------------
import sys, os

try:
    import numpy as np
except ImportError:
    sys.exit("MODULE WARNING. NumPy not installed.")

try:
    import matplotlib.pyplot as plt
except ImportError:
    sys.exit("MODULE WARNING. Matplotlib not installed.")

try:
    import classElement
except ImportError:
    sys.exit("MODULE WARNING. classElement not found, check present working directory.")


def main(Parameters, printTol=False):

    if Parameters.displacementProblem or Parameters.tractionProblem:
        #---------------------------------------
        # 2 element problem, 1D uniaxial strain.
        #---------------------------------------
        Parameters.numEl = 2
        if Parameters.displacementProblem:
            Parameters.numDOF = 4
        elif Parameters.tractionProblem:
            Parameters.numDOF = 8
        #-----------------------------
        # Create global coordinates:
        # 0.01m x 0.01m x 0.1m column.
        #-----------------------------
        coordinates = np.zeros((Parameters.numEl, Parameters.numGauss, Parameters.numDim), dtype=Parameters.float_dtype)
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
        LM       = np.ones((Parameters.numElDOF, Parameters.numEl), dtype=np.int32)
        LM      *= -1 # (this would be 0 in MATLAB)
        #----------------------------------------------
        # Set the free DOFs:
        #   - For the displacement problem, only the 
        #     middle 4 nodes are unconstrained in the 
        #     z-direction. For two elements, this 
        #     amounts to 4 DOFs.
        #   - For the traction problem, the top 4 nodes
        #     are unconstrained in the z-direction. For
        #     two elements, this amounts to 8 DOFs.
        #----------------------------------------------
        if Parameters.displacementProblem or Parameters.tractionProblem:
            LM[2,1]  = 3
            LM[5,1]  = 0
            LM[8,1]  = 1
            LM[11,1] = 2
            LM[14,0] = 3
            LM[17,0] = 0
            LM[20,0] = 1
            LM[23,0] = 2
            if Parameters.tractionProblem:
                LM[14,1] = 7
                LM[17,1] = 4
                LM[20,1] = 5
                LM[23,1] = 6

    elif Parameters.rotationProblem:
    #-----------------------------------------------
    # 1 element problem, displacement with rotation.
    #-----------------------------------------------
        Parameters.numEl = 1
        Parameters.numDOF = 4
        #---------------------------
        # Create global coordinates:
        # 1m x 1m x 1m cube.
        #---------------------------
        coordinates = np.zeros((Parameters.numEl, Parameters.numGauss, Parameters.numDim), dtype=Parameters.float_dtype)
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
        LM       = np.ones((Parameters.numElDOF, Parameters.numEl), dtype=np.int32)
        LM      *= -1 # (this would be 0 in MATLAB)
        #------------------------------
        # Set the free DOFs (you code).
        #------------------------------
        LM[14] = 3
        # LM[?] = ???
        # .
        # ..
        # ...

    #----------------------------
    # Set the initial conditions.
    #----------------------------
    D = np.zeros((Parameters.numDOF),                     dtype=Parameters.float_dtype)
    g = np.zeros((Parameters.numElDOF, Parameters.numEl), dtype=Parameters.float_dtype)
    #-----------------------------
    # Set storage arrays.
    #--------------------
    Parameters.numStressStrain = 6 # S, P, sig, E, e, hencky
    Parameters.numISV          = 2 # mean sig, von Mises stress

    DSolve       = np.zeros((Parameters.numSteps+1, Parameters.numDOF), dtype=Parameters.float_dtype)
    isv_solve    = np.zeros((Parameters.numSteps+1,Parameters.numEl,Parameters.numGauss,Parameters.numISV), dtype=Parameters.float_dtype)
    stress_solve = np.zeros((Parameters.numSteps+1,Parameters.numEl,Parameters.numGauss,Parameters.numDim,Parameters.numDim,Parameters.numStressStrain), dtype=Parameters.float_dtype)
    #------------------
    # Begin simulation.
    #------------------
    print("Solving...")
    for n in range(Parameters.numSteps):

        Parameters.t += Parameters.dt
        Parameters.n += 1
        print("n = %i, t = %.2f seconds" %(Parameters.n, Parameters.t))

        if Parameters.displacementProblem:
            #-----------------------------------
            # Update the increment displacement.
            #-----------------------------------
            g_d_np1 = Parameters.g_displ*(Parameters.t/Parameters.t_ramp)
            #-------------------
            # Update global BCs.
            #-------------------
            g[14,1] = g_d_np1
            g[17,1] = g_d_np1
            g[20,1] = g_d_np1
            g[23,1] = g_d_np1

        elif Parameters.tractionProblem:
            #-------------------------------
            # Update the increment traction.
            #-------------------------------
            Parameters.traction = Parameters.tractionLoad*(Parameters.t/Parameters.t_ramp)

        elif Parameters.rotationProblem:
            #------------------------------------------
            # Update the increment rotation magnitudes.
            #------------------------------------------
            theta_np1 = Parameters.t*Parameters.theta
            g_da_np1  = Parameters.t*Parameters.g_da
            g_db_np1  = Parameters.t*Parameters.g_db
            #---------------------------------
            # Update the rotations (you code).
            #---------------------------------
            # x_rot1 = ??? some expression or function(theta_np1) + some expression or function(g_da_np1, theta_np1)
            # y_rot1 = ??? function(theta_np1) + function(g_da_np1, theta_np1)
            # x_rot2 = ??? function(theta_np1) + function(g_da_np1, theta_np1) + function(g_db_np1, theta_np1)
            # y_rot2 = ??? function(theta_np1) + function(g_da_np1, theta_np1) + function(g_db_np1, theta_np1)
            # x_rot3 = ??? function(theta_np1) + function(g_db_np1, theta_np1)
            # y_rot3 = ??? function(theta_np1) + function(g_db_np1, theta_np1)
            #------------------------------
            # Update global BCs (you code).
            #------------------------------
            g[3] = x_rot1
            # g[?] = ???
            # .
            # ..
            # ...

        #---------------------
        # Reset N-R variables.
        #---------------------
        Rtol  = 1
        normR = 1
        k     = 0
        #----------------------
        # Begin N-R iterations.
        #----------------------
        while Rtol > Parameters.tolr and normR > Parameters.tola:

            k += 1

            if Parameters.n == 1 and k == 1:
                del_d = np.zeros((Parameters.numDOF), dtype=Parameters.float_dtype)
            else:
                del_d = np.linalg.solve(dR, -R)

            D += del_d

            R  = np.zeros((Parameters.numDOF),                    dtype=Parameters.float_dtype)
            dR = np.zeros((Parameters.numDOF, Parameters.numDOF), dtype=Parameters.float_dtype)

            for element_ID in range(Parameters.numEl):
                #------------------------
                # Initialize the element.
                #------------------------
                element = classElement.Element(a_GaussOrder=Parameters.GaussOrder, a_ID=element_ID)
                element.set_Gauss_Points(Parameters)
                element.set_Gauss_Weights(Parameters)
                element.set_Coordinates(coordinates[element.ID,:,:])
                element.evaluate_Shape_Functions(Parameters)
                element.get_Global_DOF(LM)
                element.set_Global_Solutions(D)
                element.apply_Local_BC(g)
                #------------------------------
                # Compute stresses and strains.
                #------------------------------
                element.compute_variables(Parameters)
                #---------------------------
                # Save stresses and strains.
                #---------------------------
                if Parameters.finiteStrain:
                    stress_solve[Parameters.n,element.ID,:,:,:,0] = element.SPK
                    stress_solve[Parameters.n,element.ID,:,:,:,1] = element.FPK
                    stress_solve[Parameters.n,element.ID,:,:,:,3] = element.E
                    stress_solve[Parameters.n,element.ID,:,:,:,4] = element.e
                    stress_solve[Parameters.n,element.ID,:,:,:,5] = element.Hencky
                elif Parameters.smallStrain:
                    stress_solve[Parameters.n,element.ID,:,:,:,4] = element.eps

                stress_solve[Parameters.n,element.ID,:,:,:,2] = element.sigma

                isv_solve[Parameters.n,element.ID,:,0] = element.sigma_mean
                isv_solve[Parameters.n,element.ID,:,1] = element.von_mises
                #--------------------------------
                # Compute internal force vectors.
                #--------------------------------
                element.compute_forces(Parameters)
                #-----------------------------------
                # Compute the consistent tangent(s).
                #-----------------------------------
                element.compute_tangents(Parameters)
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

            Rtol  = np.linalg.norm(R)/np.linalg.norm(R0)
            normR = np.linalg.norm(R)

            if printTol:
                print("k = ", k)
                print("Relative tolerance = ", Rtol)
                print("Norm of tolerance = ", normR)

            if k > Parameters.kmax:
                print("Relative tolerance = ", Rtol)
                print("Norm of tolerance = ", normR)
                sys.exit("ERROR. Reached max number of iterations.")

        #-----------------------------------------
        # Save the converged displacement results.
        #-----------------------------------------
        # D_solve[Parameters.n,:] = D[:] # commented out since we don't plot it

    #-----------------------------------
    # Make plot for stress-strain curve.
    #-----------------------------------
    plt.figure(1)
    if Parameters.displacementProblem or Parameters.tractionProblem:
        if Parameters.finiteStrain:
            plt.plot(-stress_solve[:,0,0,2,2,3],-stress_solve[:,0,0,2,2,0]*1e-3, 'k+-', label=r'-$S_{33}$ vs. -$E_{33}$', fillstyle='none')
            plt.plot(-stress_solve[:,0,0,2,2,4],-stress_solve[:,0,0,2,2,2]*1e-3, 'ko-', label=r'-$\sigma_{33}$ vs. -$e_{33}$', fillstyle='none')
            plt.plot(-stress_solve[:,0,0,2,2,5],-stress_solve[:,0,0,2,2,2]*1e-3, 'ks-', label=r'-$\sigma_{33}$ vs. -$h_{33}$', fillstyle='none')
        elif Parameters.smallStrain:
            plt.plot(-stress_solve[:,0,0,2,2,4],-stress_solve[:,0,0,2,2,2]*1e-3, 'ko-', label=r'-$\sigma_{33}$ vs. -$\epsilon_{33}$', fillstyle='none')
    
    elif Parameters.rotationProblem:
        #-------------------------------
        # Calculate minimum eigenvalues.
        #-------------------------------
        minStress = np.min(np.linalg.eig(stress_solve[:,0,0,:,:,2])[0], axis=1)
        if Parameters.finiteStrain:
            minStrain = np.min(np.linalg.eig(stress_solve[:,0,0,:,:,5])[0], axis=1)
        elif Parameters.smallStrain:
            minStrain = np.min(np.linalg.eig(stress_solve[:,0,0,:,:,4])[0], axis=1)
        plt.plot(-minStrain,-minStress*1e-3, 'k-', label=r'minStress vs. minStrain')
    
    plt.ylabel('-Stress (kPa)')
    plt.xlabel('-Strain (m/m)')

    plt.legend()
    plt.grid()
    plt.show()

    return
