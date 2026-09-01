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

# Import impulse approximation and interaction contributions
from . import impulse as _impulse
from . import string as _string

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def AU(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       coulomb = False,
       string = False
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
            True to include one-body currents, False to exclude
            Default: True
        - coulomb: boolean
            True to include Coulomb-like stress, False to exclude
            This will fail unless the associated wf has an alpha member,
            which signifies the effective Coulomb constant
            (for one gluon exchange, alpha means alpha_s*CF)
            Default: False
        - string: boolean
            True to include string stress, False to exclude
            This will fail unless the associated wf has a sigma member,
            which specifies the string tension
            Default: False
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
        The latter uses the formula explicitly given in 2602.18298.
        The option is given only to demonstrate that the results are the same,
        but the 'paper' formula is significantly slower.
    '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.AU(k, dwf=dwf, nff=_nff)
    return result

def AT(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor AT. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.AT(k, dwf=dwf, nff=_nff)
    return result

def DU(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor DU. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.DU(k, dwf=dwf, nff=_nff)
    if(interactions.get('string', False)):
        result += _string.DU(k, dwf=dwf)
    return result

def DT1(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor DT1. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.DT1(k, dwf=dwf, nff=_nff)
    return result

def DT2(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor DT2. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.DT2(k, dwf=dwf, nff=_nff)
    return result

def cU(k,
       wf = wf_default,
       nff = 'ba',
       formula = 'fast',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor cU. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin = 1e-2
    result = k*0
    if(impulse):
        result += _impulse.cU(k, dwf=dwf, nff=_nff, rmin=rmin, formula=formula)
    if(interactions.get('string', False)):
        result += _string.cU(k, dwf=dwf)
    return result

def cT1(k,
       wf = wf_default,
       nff = 'ba',
       formula = 'fast',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor cT1. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.cT1(k, dwf=dwf, nff=_nff, formula=formula)
    return result

def cT2(k,
       wf = wf_default,
       nff = 'ba',
       formula = 'fast',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor cT2. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    # Need to change rmin from 0 to 1e-2 for Yukawa parametrizations,
    # because of an instability at small r
    _nff = choose_nff(nff)
    rmin = 0
    if(wf=='paris' or wf=='cdbonn'):
        rmin =  1e-2
    result = k*0
    if(impulse):
        result += _impulse.cT2(k, dwf=dwf, nff=_nff, rmin=rmin, formula=formula)
    return result

def J(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor J. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.J(k, dwf=dwf, nff=_nff)
    return result

def S(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor S. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.S(k, dwf=dwf, nff=_nff)
    return result

def sbar(k,
       wf = wf_default,
       nff = 'ba',
       impulse = True,
       **interactions
       ):
    ''' The EMT form factor S. See docstring of AU for more info. '''
    dwf = choose_wf(wf)
    _nff = choose_nff(nff)
    result = k*0
    if(impulse):
        result += _impulse.sbar(k, dwf=dwf, nff=_nff)
    return result
