# deuteron.py
# Created 2026.05.08 by Adam Freese
#
# This file contains interfaces for the calculation of EMT-FFs of the deuteron.
# Implementations will be in different files.
# The impulse approximation used in Cosyn, Freese & Sosa (2602.18298)
# has been moved to the file impulse.py
# The impulse approximation can still be used
# through the interface functions found here.

import numpy as np
from numpy import sqrt

# Import wave function and nucleon form factor choosers
from ..wf.chooser import choose_wf
from .nucleon.chooser import choose_nff

# Default wave function is an AV18 instance
from ..wf.av18 import dwf_av18
wf_default = dwf_av18()

# Import impulse approximation routines
from . import impulse as onebody

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def AU(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor AU.
    ----------
    Input:
        - k : float or numpy.array
            float one-dimensional array of k values in GeV
        - wf : DWF or string
            Deuteron wave function to use
            See wf.chooser.choose_wf for available options
        - nff : string
            Nucleon EMT form factors to use
            Available: ba, mab, hz, point
            Default: ba
        - impulse: boolean
            True if the impulse contributions are to be included
            False to exclude them
        - interactions : list of booleans
            interaction types that should be included, e.g.,
            coulomb=True, string=True, etc.
            The list is allowed to be empty. Any interactions not listed
            will not be calculated.
            Recognized interactions:
            - None (TODO)
    Output:
        numpy.array with form factor values
    Notes:
        Some form factors have multiple options for formulas.
        These are meant for consistency checks.
        The default formula in each case is the fastest to evaluate.
        Form factors with a 'formula' option are:
            - cU
            - cT1
            - cT2
        The options for the formula are 'fast' (default) or 'paper'.
        The latter uses the formula explicitly given in the paper.
        The option is given only to demonstrate that the results are the same,
        but the 'paper' formula is significantly slower.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.AU(k, dwf=dwf, nff=_nff)

def AT(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor AT.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.AT(k, dwf=dwf, nff=_nff)

def DU(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor DU.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.DU(k, dwf=dwf, nff=_nff)

def DT1(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor DT1.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.DT1(k, dwf=dwf, nff=_nff)

def DT2(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor DT2.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.DT2(k, dwf=dwf, nff=_nff)

def cU(k,
       wf = wf_default,
       nff = 'ba',
       formula = 'fast',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor cU.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin = 1e-2
    return onebody.cU(k, dwf=dwf, nff=_nff, rmin=rmin, formula=formula)

def cT1(k,
       wf = wf_default,
       nff = 'ba',
       formula = 'fast',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor cT1.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.cT1(k, dwf=dwf, nff=_nff, formula=formula)

def cT2(k,
       wf = wf_default,
       nff = 'ba',
       formula = 'fast',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor cT2.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    _nff = choose_nff(nff)
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin =  1e-2
    return onebody.cT2(k, dwf=dwf, nff=_nff, rmin=rmin, formula=formula)

def J(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor J.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.J(k, dwf=dwf, nff=_nff)

def S(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor S.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.S(k, dwf=dwf, nff=_nff)

def sbar(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor S.
    See docstring of AU for more info.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    return onebody.sbar(k, dwf=dwf, nff=_nff)
