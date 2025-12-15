# misc.py
# Created 2025.09.30 by Adam Freese
#
# This file contains simple estimates for the nucleon EMT form factors
# that are absent from the other modules.

def cN(k):
    ''' Zero if summed over parton flavors. '''
    return k*0

def SN(k):
    ''' Dipole form, assuming axial vector meson dominance,
    and using the JAM value for total quark spin.
    The s0 value is from:
        C. Cocuzza et al.
        Physical REview D 106 (2022) L031502
        Cocuzza:2022jye
    '''
    ma1 = 1.23  # PDG
    s0  = 0.204 # JAM22; Cocuzza et al., PRD 106 (2022) L031502
    return 0.204 / (1 + (k/ma1)**2)**2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def cN_q(k):
    ''' Dipole form, using parameters from:
        Lorcé, Moutarde and Trawiński
        European Physical Journal C 79 (2019) 89
        Lorce:2018egm
    '''
    c0 = -0.11 # EPJC 79 (2019) 89
    L  =  0.91 # EPJC 79 (2019) 89
    return c0 / (1 + (k/L)**2)**2

def cN_g(k):
    return -cN_q(k)

def SN_q(k):
    return SN(k)

def SN_g(k):
    return k*0
