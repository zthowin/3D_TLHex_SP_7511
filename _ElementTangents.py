#----------------------------------------------------------------------------------------
# Module housing element object consistent tangents.
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
def compute_tangents(self, Parameters):
    # Compute element tangents.
    self.get_G_Tangents(Parameters)

@register_method
def get_G_Tangents(self, Parameters):
    # Assemble solid consistent tangents.
    self.G_Mtx = np.zeros((24,24))
    self.get_G_uu_1(Parameters)

    try:
        self.G_Mtx += self.G_uu_1
        # print(self.G_Mtx)
        # input()
    except FloatingPointError:
        print("ERROR. Encountered over/underflow error in G; occurred at element ID %i, t = %.2es and dt = %.2es." %(self.ID, Parameters.t, Parameters.dt))
        raise FloatingPointError
    return

@register_method
def get_G_uu_1(self, Parameters):
    # Compute G_uu_1.
    # debug1 = np.einsum('...ai, ...AI', self.identity, self.SPK)
    # debug2 = Parameters.lambd*np.einsum('...Aa,...Ii', self.F_inv, self.F_inv) 
    # debug3 = np.einsum('...Ai, ...Ia', self.F_inv, self.F_inv)
    # debug4 = np.einsum('...ai, ...AI', self.identity, self.C_inv)
    # debug5 = np.einsum('..., ...aiAI', Parameters.lambd*np.log(self.J) - Parameters.mu, debug3 + debug4)

    # print(Parameters.lambd*np.einsum('...Aa,...Ii', self.F_inv, self.F_inv))
    self.dPdF = np.einsum('...ai, ...AI', self.identity, self.SPK)\
                + Parameters.lambd*np.einsum('...ai,...AI', self.F_inv, self.F_inv)\
                - np.einsum('..., ...aAiI', Parameters.lambd*np.log(self.J) - Parameters.mu,\
                                            (np.einsum('...Ai, ...Ia', self.F_inv, self.F_inv)\
                                             + np.einsum('...ai, ...AI', self.identity, self.C_inv)))

    # print(self.dPdF)
    # input()
    self.dPdF_voigt = np.zeros((8,9,9), dtype=np.float64)
    for alpha in range(9):
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
        for beta in range(9):
            if beta == 0:
                a = 0
                A = 0
            elif beta == 1:
                a = 0
                A = 1
            elif beta == 2:
                a = 0
                A = 2
            elif beta == 3:
                a = 1
                A = 0
            elif beta == 4:
                a = 1
                A = 1
            elif beta == 5:
                a = 1
                A = 2
            elif beta == 6:
                a = 2
                A = 0
            elif beta == 7:
                a = 2
                A = 1
            elif beta == 8:
                a = 2
                A = 2

            self.dPdF_voigt[:,alpha,beta] = self.dPdF[:,i,I,a,A]

    # print(self.dPdF_voigt)
    # input()
    # 24 x 9 x 8
    # 9 x 24 x 8
    # 9 x 9 x 8
    # print(self.dPdF_voigt.shape)
    # print(self.Bu.shape, self.j)
    # print(self.Bu.T.shape)
    # weighted_dPdF = np.einsum('...ab, ...', self.dPdF_voigt, self.weights*self.j)
    weighted_dPdF = np.einsum('...ab, ...', self.dPdF_voigt, self.weights*self.j)
    # print(weighted_dPdF.shape)
    # print(self.Bu.shape)
    # input()
    one = np.einsum('ja..., ...bj', self.Bu, weighted_dPdF)
    # one = np.einsum('ja..., ...bj', self.Bu, weighted_dPdF)
    # print(one.shape)
    # print(one[0,:,:])
    # print(self.Bu[0,:,:].T.shape)
    # print(self.Bu[:,:,0].T)
    # input()
    print(one.shape, self.Bu.shape)
    # self.G_uu_1 = np.einsum('...kj, jn...->...nk', one, self.Bu)
    self.G_uu_1 = np.einsum('iI...,...IA,Aj...,...->...ij', self.Bu, self.dPdF_voigt, self.Bu, self.weights*self.j)
    print(self.G_uu_1[0,:,:])
    # print(self.G_uu_1.shape)
    input()
    # print(self.Bu.shape)
    # print(two.shape)
    # print(one.shape)
    # self.G_uu_1 = np.einsum('ijk, kjj, jik', self.Bu, weighted_dPdF, self.Bu, dtype=np.float64)
    # print(self.G_uu_1.shape)
    # sys.exit()
    return
