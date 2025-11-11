# ba.py
# Created 2025.11.11 by Adam Freese
#
# This contains nucleon MFFs from:
#   Wojciech Broniowski and Enrique Ruiz Arriola
#   Physical Review D 112 (2025) 054028
#   Broniowski:2025ctl

from ...constants import mN

def AN(k):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    cA     = 0.62 # central value for set I, see Eq. (52)
    c2     = 0.15 # central value for set I, see Eq. (52)
    mf2    = 1.275 # from set I, see Eq. (51)
    mf2p   = 1.517 # from set I, see Eq. (51)
    mf2pp  = 1.565 # from set I, see Eq. (51)
    mf2ppp = 1.936 # from set I, see Eq. (51)
    t = -k**2
    num = 1 - cA*t + c2*t**2
    den = (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den

def JN(k):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    cJ     = 0.87 # central value for set I, see Eq. (52)
    c2     = 0.15 # central value for set I, see Eq. (52)
    mf2    = 1.275 # from set I, see Eq. (51)
    mf2p   = 1.517 # from set I, see Eq. (51)
    mf2pp  = 1.565 # from set I, see Eq. (51)
    mf2ppp = 1.936 # from set I, see Eq. (51)
    t = -k**2
    num = 1 - cJ*t + c2*t**2
    den = 2 * (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den

def ThetaN(k):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    mf0    = 0.98 # see text above Eq. (51)
    msigma = 0.64 # central value for set I, see Eq. (52)
    t = -k**2
    num = mN
    den = (1-t/mf0**2) * (1-t/msigma**2)
    return num/den

def DN(k):
    ''' See Eq. (15) of Broniowski:2025ctl '''
    t = -k**2
    P2 = mN**2 - t/4
    return ( (4*P2*AN(k) - 4*mN*ThetaN(k))/t + 2*JN(k) )/3
