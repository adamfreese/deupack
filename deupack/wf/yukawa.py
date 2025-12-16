# yukawa.py
# Created 2025.12.15 by Adam Freese to consolidate codes by Alan Sosa
#
# This parametrizes the deuteron wave function as a sum of Yukawa-type terms,
# in line with the suggestion in:
#   Lacombe, Loiseau, Vinh Mau, Cote, Pires, and de Tourreil,
#   Parametrization of the deuteron wave function of the Paris N-N potential
#   Phys.Lett.B 101 (1981) 139-140
#   Lacombe:1981eg

import numpy as np

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def u(r, C_J, m_j):
    s = C_J * np.exp(-m_j*r)
    s = np.sum(s)
    return s

def u1(r, C_J, m_j):
    s1 = -C_J*m_j * np.exp(-m_j*r)
    s1 = np.sum(s1)
    return s1

def u2(r, C_J, m_j):
    s2 = C_J*m_j**2 * np.exp(-m_j*r)
    s2 = np.sum(s2)
    return s2

def u3(r, C_J, m_j):
    s3 = -C_J*m_j**3 * np.exp(-m_j*r)
    s3 = np.sum(s3)
    return s3

def w(r, D_J, m_j):
    kr = m_j * r
    d = D_J * np.exp(-kr) * (1. + 3./kr + 3./ kr**2)
    d = np.sum(d)
    return d

def w1(r, D_J, m_j):
    kr = m_j * r
    d1 = -D_J*m_j * np.exp(-kr) * (
            1. + 3./kr + 6./ kr**2 + 6./ kr**3
        )
    d1 = np.sum(d1)
    return d1

def w2(r, D_J, m_j):
    kr = m_j * r
    d2 = D_J * m_j**2 * np.exp(-kr) * (
            1. + 3./kr + 9./ kr**2 + 
            18./ kr**3 + 18./ kr**4
        )
    d2 = np.sum(d2)
    return d2

def w3(r, D_J, m_j):
    kr = m_j * r
    d3 = -D_J * m_j**3 * np.exp(-kr) * (
            1. + 3./kr + 12./ kr**2 + 
            36./ kr**3 + 72./ kr**4 + 72./ kr**5
        )
    d3 = np.sum(d3)
    return d3
