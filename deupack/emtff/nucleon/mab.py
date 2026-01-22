# point.py
# Created 2025.11.13 by Alan Sosa
#
# This contains nucleon EMTFFs parametrization used in Freese Cosyn spatial density paper


def AN(k):
    ''' AN. Assumes k is in GeV. '''
    mf2_1270 = 1.27    # f2 1270 mass (GeV)
    mf2_1430 = 1.43    # f2 1430 mass (GeV)

    a= dipole(k, mf2_1270)*dipole(k, mf2_1430)
    return a


def JN(k):
    ''' JN. Assumes k is in GeV. '''
    mf2_1270 = 1.27    # f2 1270 mass (GeV)
    mf2_1430 = 1.43    # f2 1430 mass (GeV)

    a= 0.5*dipole(k, mf2_1270)*dipole(k, mf2_1430)
    return a

def DN(k):
    ''' DN. Assumes k is in GeV. '''
    mf2_1270 = 1.27    # f2 1270 mass (GeV)
    mf2_1430 = 1.43    # f2 1430 mass (GeV)
    msigma = 0.8    # sigma mass (GeV)

    a= -2.*dipole(k, mf2_1270)*dipole(k, mf2_1430)*dipole(k, msigma)
    return a


def dipole(k, Lambda):
    ''' Dipole form. '''
    return 1. / (1 + (k/Lambda)**2)**2
