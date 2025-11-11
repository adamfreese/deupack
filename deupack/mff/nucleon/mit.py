# hps.py
# Created 2025.09.30 by Adam Freese
#
# This file contains formulas for the mechanical form factors of a nucelon,
# as calculated by:
#   Daniel C. Hackett, Dimitra A. Pefkou, and Phiala E. Shanahan
#   Physical Review Letters 132 (2024) 241904
# which I will refer to as MIT after the authors' institution.
#
# MIT actually give two parametric fits to their lattice data:
# (1) a dipole fit, and (2) a z-expansion.
# Both fits achieve qualitatively similar results and similar chi2/dof.
# For simplicity, I use their dipole forms here.
#
# One minor tweak I make here is that I enforce the sum rules A(0)=1 and J(0)=1/2.
# I adjust the alpha parameters for Ag and Jg to achieve this.
# On the other hand, I use the central fit values for Aq and Jq.
# Since D(0) is not constrained by any conservation laws,
# I use the central values reported by MIT for both Dq(0) and Dg(0)
#
# Parameters of MIT are given in Table III of their supplemental material.

def AN(k):
    ''' Form factor AN. Assumes k is in GeV. '''
    return AN_q(k) + AN_g(k)

def JN(k):
    ''' Form factor JN. Assumes k is in GeV. '''
    return JN_q(k) + JN_g(k)

def DN(k):
    ''' Form factor DN. Assumes k is in GeV. '''
    return DN_q(k) + DN_g(k)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def AN_q(k):
    ''' Quark contribution to AN. Assumes k is in GeV. '''
    alpha  =  0.510 # reported 0.510(25)
    Lambda =  1.477 # reported 1.477(44)
    return dipole(k, alpha, Lambda)

def JN_q(k):
    ''' Quark contribution to JN. Assumes k is in GeV. '''
    alpha  =  0.251 # reported 0.251(21)
    Lambda =  1.62  # reported 1.62(13)
    return dipole(k, alpha, Lambda)

def DN_q(k):
    ''' Quark contribution to DN. Assumes k is in GeV. '''
    alpha  = -1.30 # reported -1.30(49)
    Lambda =  0.81 # reported 0.81(14)
    return dipole(k, alpha, Lambda)

def AN_g(k):
    ''' Gluon contribution to AN. Assumes k is in GeV. '''
    alpha  =  0.490 # reported 0.501(27); adjusted for sum rule
    Lambda =  1.262 # reported 1.262(18)
    return dipole(k, alpha, Lambda)

def JN_g(k):
    ''' Gluon contribution to JN. Assumes k is in GeV. '''
    alpha  =  0.249 # reported 0.255(13); adjusted for sum rule
    Lambda =  1.399 # reported 1.399(49)
    return dipole(k, alpha, Lambda)

def DN_g(k):
    ''' Gluon contribution to DN. Assumes k is in GeV. '''
    alpha  = -2.57  # reported -2.57(84)
    Lambda =  0.538 # reported 0.538(65)
    return dipole(k, alpha, Lambda)

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def dipole(k, alpha, Lambda):
    ''' Dipole form. '''
    return alpha / (1 + (k/Lambda)**2)**2
