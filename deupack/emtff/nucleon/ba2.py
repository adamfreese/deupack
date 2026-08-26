import numpy as np

from ...constants import mN,hbar

from .nff import *




''' Nucleon EMT-FFs from the meson dominance model of:
        Masjuan, Ruiz Arriola and Broniowski
        Phys. Rev. D 87 (2013) 014005
        Masjuan:2012sk
    Editted to have separation between quarks and gluons
    quark gluon separated functions by Adam Freese
    For cbar form factor using the D2 scheme
'''



# mN    = 0.93891875569 # averaged nucleon mass [arithmetic mean] (GeV)
mN    = 0.970 # mass used for nucleon because of lattice pion mass difference from real mass (GeV)
mf0    = 0.98 # see text above Eq. (51)

mf0p= 1.250  #from PDG


# masses of mesons from set I
mf2    = 1.275 # from set I, see Eq. (51)
mf2p   = 1.517 # from set I, see Eq. (51)
mf2pp  = 1.565 # from set I, see Eq. (51)
mf2ppp = 1.936 # from set I, see Eq. (51)
msigma = 0.64 # central value for set I, see Eq. (52)

# mass of mesons from set II
# mf2    = 1.275 # from set II, see Eq. (51)
# mf2p   = 1.430 # from set II, see Eq. (51)
# mf2pp  = 1.517 # from set II, see Eq. (51)
# mf2ppp = 1.565 # from set II, see Eq. (51)

# msigma = 0.64 # central value for set II, see Eq. (52)


#mt is minus t so a positive number!!
def AN1(mt,A_0,cA,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    # t =-k**2
    t=-mt
    num = A_0 - cA*t + c2*t**2
    den = (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den

def JN1( mt,J_0,cJ,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    # t =-k**2
    t=-mt
    num = 2*J_0 - cJ*t + c2*t**2
    den = 2 * (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den



def ThetaP(mt,theta_p):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    # t = -k**2
    t=-mt
    num = mN*theta_p
    den = (1-t/mf0**2) * (1-t/msigma**2)
    return num/den

def newThetaP(mt,theta_p,c2theta):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    # t = -k**2
    t=-mt
    num = mN*theta_p+ c2theta*t
    den = (1-t/mf0**2) * (1-t/msigma**2)*(1-t/mf0p**2)
    return num/den





def cbar(mt,c_0):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    # t = -k**2
    t=-mt
    num = c_0
    den = (1-t/mf0**2) * (1-t/msigma**2)
    return num/den




def DN1(mt,A_0,J_0,cA,cJ,c2):
    # t = -k**2

    t=-mt
    return -4*mN*( -mN*AN1(mt,A_0,cA,c2) +ThetaP(mt,A_0) + (t/(8*mN))*(JN1(mt,J_0,cJ,c2) +AN1(mt,A_0,cA,c2)))/(3*(t))



def new_DN1(mt,A_0,J_0,cA,cJ,c2,c2theta):
    # t = -k**2

    t=-mt
    c0=0.12 #doesn't matter what this is it cancels anyway here
    return -4*mN*( -mN*AN1(mt,A_0,cA,c2) +newThetaP(mt,A_0+4*c0,c2theta) -4*mN*cbar(mt,c0) + (t/(8*mN))*(JN1(mt,J_0,cJ,c2) +AN1(mt,A_0,cA,c2)))/(3*(t))


# values found from BA previous fits to total form factors
cA     = 0.62 # central value for set I, see Eq. (52)
c2     = 0.15 # central value for set I, see Eq. (52)
cJ     = 0.87 # central value for set I, see Eq. (52)

# cA     = 0.83 # central value for set II, see Eq. (52)
# c2     = 0.25 # central value for set II, see Eq. (52)
# cJ     = 1.12 # central value for set II, see Eq. (52)



# fitting function


# depends on scheme (already divided by mass)
theta_q = 0.08
theta_g= 0.92




A0q= 0.575   
cAq=    0.351   
c2q=    0.152 
J0q =   0.275
cJq=    0.440 
c2thetaq =   -1.24 
c2thetag =  -0.835 



    #sum rules
A0g = 1.0 - A0q

J0g = 0.5 - J0q


    #constraints from BA previous fit for total


c2g = c2 -c2q
cAg = cA -cAq
cJg= cJ - cJq



#In D2 scheme
c_0q = (theta_q -A0q)/4.
c_0g = -c_0q


class nff_ba2(nff_with_SN):
    ''' 
    BA parametrization of EMT-FFs that were fit to lattice data
    '''

    def __init__(self):
        super().__init__()
        self.name = "ba2"
        return

    # Form factor overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN(self, k):
        ''' Form factor AN. Assumes k is in GeV. '''
        return self.AN_q(k) + self.AN_g(k)

    def JN(self, k):
        ''' Form factor JN. Assumes k is in GeV. '''
        return self.JN_q(k) + self.JN_g(k)

    def DN(self, k):
        ''' Form factor DN. Assumes k is in GeV. '''
        return self.DN_q(k) + self.DN_g(k)

    def cN(self, k):
        ''' Form factor cN. Assumes k is in GeV.
        Found using D2 scheme
        '''
        return self.cN_q(k) + self.cN_g(k)


    # Auxiliary functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_q(self, k):
        return AN1(k**2,A0q,cAq,c2q)

    def JN_q(self, k):
        return JN1(k**2,J0q,cJq,c2q)

    def DN_q(self, k):
        return DN1(k**2,A0q,J0q,cAq,cJq,c2q)

    def AN_g(self, k):
        return AN1(k**2,A0g,cAg,c2g)

    def JN_g(self, k):
        return JN1(k**2,J0g,cJg,c2g)

    def DN_g(self, k):
        return DN1(k**2,A0g,J0g,cAg,cJg,c2g)

    def cN_q(self, k):
        return cbar(k**2,c_0q)

    def cN_g(self, k):
        return cbar(k**2,c_0g)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_ba_quark(nff_ba2):
    ''' A quark-only variation on nff_ba2. '''

    def __init__(self):
        super().__init__()
        self.name = "baq"
        return

    # Overrides to eliminate gluons ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_g(self, k):
        return k*0

    def JN_g(self, k):
        return k*0

    def DN_g(self, k):
        return k*0

    def cN_g(self, k):
        return k*0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_ba_gluon(nff_ba2):
    ''' A gluon-variation on nff_ba2. '''

    def __init__(self):
        super().__init__()
        self.name = "bag"
        return

    # Overrides to eliminate quarks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_q(self, k):
        return k*0

    def JN_q(self, k):
        return k*0

    def DN_q(self, k):
        return k*0

    def cN_q(self, k):
        return k*0

    def SN(self, k):
        # Note that gluons cannot contribute to SN
        return k*0

