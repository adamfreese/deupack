# ba.py
# Created 2025.11.11 by Adam Freese

from ...constants import hbar

from .nff import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~



''' Nucleon EMT-FFs from the meson dominance model of:
        Masjuan, Ruiz Arriola and Broniowski
        Phys. Rev. D 87 (2013) 014005
        Masjuan:2012sk
    Editted to have separation between quarks and gluons
    quark gluon separated functions by Adam Freese
    For cbar form factor using different D schemes
'''


mN    = 0.970 # mass used for nucleon because of lattice pion mass difference from real mass (GeV)
mf0    = 0.98 # see text above Eq. (51)

mf0p= 1.250  #from PDG


# masses of mesons from set I
mf2    = 1.275 # from set I, see Eq. (51)
mf2p   = 1.517 # from set I, see Eq. (51)
mf2pp  = 1.565 # from set I, see Eq. (51)
mf2ppp = 1.936 # from set I, see Eq. (51)
# msigma = 0.64 # central value for set I, see Eq. (52)


msigma = 0.579 #from my fit!



def AN1(k,A_0,cA,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''

    t =-k**2
    num = A_0 - cA*t + c2*t**2
    den = (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den

def JN1(k,J_0,cJ,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    t =-k**2
    num = 2*J_0 - cJ*t + c2*t**2
    den = 2 * (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den



def ThetaP(k,theta_p):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    t = -k**2
    num = mN*theta_p
    den = (1-t/mf0**2) * (1-t/msigma**2)
    return num/den


def cbar(k,c_0):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    t = -k**2
    num = c_0
    den = (1-t/mf0**2) * (1-t/msigma**2)
    return num/den




def DN1(k,A_0,J_0,cA,cJ,c2):
    t = -k**2

    A = AN1(k,A_0,cA,c2)
    B = 2*JN1(k,J_0,cJ,c2) -A
    theta = ThetaP(k,A_0)
    return -1./(3.*t)*( 4*mN**2*(theta/mN-A) -t*B)





A0q= 0.571
cAq=    0.237
c2q=    0.206
cAg = 0.232
c2g = 0.015
J0q =   0.276
cJq=    0.422 
cJg= 0.396





    #sum rules
A0g = 1.0 - A0q

J0g = 0.5 - J0q


    #constraints from BA previous fit for total






class nff_ba2(nff_with_SN):
    ''' 
    BA parametrization of EMT-FFs that were fit to lattice data
    '''

    def __init__(self):
        super().__init__()
        self.name = "ba2"
        self.mN  = 0.970 # mass used for nucleon because of lattice pion mass difference from real mass (GeV)
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
        '''



        return self.cN_q(k) + self.cN_g(k)


    def AN_q(self, k):
        return AN1(k,A0q,cAq,c2q)

    def JN_q(self, k):
        return JN1(k,J0q,cJq,c2q)

    def DN_q(self, k):
        return DN1(k,A0q,J0q,cAq,cJq,c2q)

    def AN_g(self, k):
        return AN1(k,A0g,cAg,c2g)

    def JN_g(self, k):
        return JN1(k,J0g,cJg,c2g)

    def DN_g(self, k):
        return DN1(k,A0g,J0g,cAg,cJg,c2g)

    def cN_q(self, k):

        theta_q = 0.08

        c_0q = (theta_q -A0q)/4.
        # c_0g = -c_0q

        return cbar(k,c_0q)

    def cN_g(self, k):
    
        theta_q = 0.08
    
        c_0q = (theta_q -A0q)/4.
        c_0g = -c_0q
    
        return cbar(k,c_0g)

    
    # Auxiliary functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def ThetaN(self, k):
        ''' See Eq. (50) of Broniowski:2025ctl '''
        mf0    = 0.98 # see text above Eq. (51)
        msigma = 0.64 # central value for set I, see Eq. (52)
        t = -k**2
        mN = self.mN
        num = mN
        den = (1-t/mf0**2) * (1-t/msigma**2)
        return num/den

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_ba2_quarkD2(nff_ba2):
    ''' A quark-variation on nff_ba2 using D2 scheme at a scale mu=2 GeV^2'''

    def __init__(self):
        super().__init__()
        self.name = "ba2q_D2"
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

        
    def cN_q(self, k):

        theta_q = 0.08

        c_0q = (theta_q -A0q)/4.
        # c_0g = -c_0q

        return cbar(k,c_0q)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_ba2_gluonD2(nff_ba2):
    ''' A gluon-variation on nff_ba2 using D2 scheme at a scale mu=2 GeV^2'''

    def __init__(self):
        super().__init__()
        self.name = "ba2g_D2"
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

        
    def cN_g(self, k):

        theta_q = 0.08

        c_0q = (theta_q -A0q)/4.
        c_0g = -c_0q

        return cbar(k,c_0g)

    def SN(self, k):
        # Note that gluons cannot contribute to SN
        return k*0



class nff_ba2_quarkD3(nff_ba2):
    ''' A quark-variation on nff_ba2 using D3 scheme at a scale mu=2 GeV^2'''

    def __init__(self):
        super().__init__()
        self.name = "ba2q_D3"
        return

    # Overrides to eliminate gluons and change quark contribution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_g(self, k):
        return k*0

    def JN_g(self, k):
        return k*0

    def DN_g(self, k):
        return k*0

    def cN_g(self, k):
        return k*0

        
    def cN_q(self, k):

        theta_q = 1.0
        c_0q = (theta_q -A0q)/4.
        # c_0g = -c_0q


        return cbar(k,c_0q)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_ba2_gluonD3(nff_ba2):
    ''' A gluon-variation on nff_ba2 using D3 scheme at a scale mu=2 GeV^2'''

    def __init__(self):
        super().__init__()
        self.name = "ba2g_D3"
        return

    # Overrides to eliminate quarks and change gluon contribution ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_q(self, k):
        return k*0

    def JN_q(self, k):
        return k*0

    def DN_q(self, k):
        return k*0

    def cN_q(self, k):
        return k*0

        
    def cN_g(self, k):

        theta_q = 1.0
        c_0q = (theta_q -A0q)/4.
        c_0g = -c_0q

        return cbar(k,c_0g)

    def SN(self, k):
        # Note that gluons cannot contribute to SN
        return k*0


