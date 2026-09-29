# string.py
# Created 2026.05.08 by Adam Freese
#
# Contributions to EMT-FFs of a classical string.
# In progress. Currently unpolarized EMT-FFs only.
# TODO

import numpy as np
from scipy.special import spherical_jn as jn, sici
from scipy.integrate import quad_vec

from ..constants import hbar
from .impulse import regulate_zero # maybe put in a common utils.py file

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# The user interfaces for the string contributions to the form factors
# More friendly user interfaces are given in deuteron.py

def DU(k, dwf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_DU_integrand, 0, np.inf,
                        args=(k, dwf),
                        workers=8
                        )[0]
    return integral

def cU(k, dwf):
    k = regulate_zero(k) # avoid division by zero
    integral = quad_vec(_cU_integrand, 0, np.inf,
                        args=(k, dwf),
                        workers=8
                        )[0]
    return integral

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Under-the-hood implementation details for the EMTFFs: Integrands
# Parallelization of the integration requires the integrands to be defined
# as top-level (rather than nested) functions.

def _DU_integrand(r, k, dwf):
    kfm = k/hbar
    Si, _ = sici(kfm*r/2)
    intd = 8*dwf.mN*dwf.sigma/k**3*(
            Si - 3*jn(1,kfm*r/2)
            ) * dwf.u(r)**2
    return intd

def _cU_integrand(r, k, dwf):
    kfm = k/hbar
    intd = dwf.sigma / (dwf.mN*k) * jn(1,kfm*r/2) * dwf.u(r)**2
    return intd

