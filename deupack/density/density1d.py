# density1d.py
# Created 2026.09.23 by Adam Freese
#
# Provides a modified version of the Density class that stores 1D arrays instead.
# This will construct densities much faster.

import numpy as np
import pandas as pd

from scipy.integrate import quad_vec
from scipy.interpolate import CubicSpline

from pathlib import Path

from ..wf.chooser import choose_wf
from ..emtff.nucleon.chooser import choose_nff
from .. import emtff

from .bessel import *
from .density3d import Density

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Density class

class Density1D(Density):
    ''' A class for the calculation of deuteron densities.

    Uses 1D arrays. This class is meant for faster calculations.

    This is implemented as a class so that lookup tables for mechanical form
    factors can be cached, and so that the user can create different objects
    with different EMT-FFs in their cache.
    '''

    def __init__(self,
                 wf='av18',
                 nff='ba',
                 nk=600,
                 kmin=1e-6, # GeV
                 kmax=20,   # GeV
                 **kwargs
                 ):
        self.wf     = choose_wf(wf)
        self.nff    = choose_nff(nff)
        self.mN     = self.wf.mN
        self.nk     = nk
        self.kmin   = kmin
        self.kmax   = kmax
        self.kwargs = kwargs
        # attempt to find a cached lookup table on disk
        path = self._cache_path()
        if(path.is_file()):
            self._load_emtff_table(path)
        else:
            # if not found, make one
            self._init_emtff_table(save_table=True)
        return

    # Density methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # TODO

    def mass_density(self, b, pol='U'):
        ''' Mass density in GeV/fm**3. '''
        if(pol=='U'):
            return self._bessel_transform(massU_integrand, b, self.AU, self.mN)
        elif(pol=='T'):
            # Needs to be multiplied by 3/2*cos(theta)**2 - 1/2
            return self._bessel_transform(massT_integrand, b, self.AT, self.mN)
        else:
            self._pol_error(pol)

    def momentum_density(self, b):
        ''' Phi component of the momentum density, in GeV/fm**3.
        Needs to be multiplied by sin(theta).
        '''
        return self._bessel_transform(momentum_integrand, b, self.J, self.S)

    def flux_density(self, b):
        ''' Phi component of the mass flux density, in GeV/fm**3.
        Needs to be multiplied by sin(theta).
        '''
        return self._bessel_transform(flux_integrand, b, self.J, self.S)

    def pressure(self, b, pol='U'):
        if(pol=='U'):
            return self._bessel_transform(pressureU_integrand, b, self.DU, self.cU, self.mN)
        elif(pol=='T'):
            return b*0 # TODO
        else:
            self._pol_error(pol)

    def shear(self, b, pol='U'):
        if(pol=='U'):
            return self._bessel_transform(shearU_integrand, b, self.DU, self.mN)
        elif(pol=='T'):
            return b*0 # TODO
        else:
            self._pol_error(pol)

    def radial_pressure(self, b, pol='U'):
        ''' Radial pressure in GeV/fm**3. '''
        return self.pressure(b, pol=pol) + 2/3*self.shear(b, pol=pol)

    def polar_pressure(self, b, pol='U'):
        ''' Polar pressure in GeV/fm**3. '''
        return self.pressure(b, pol=pol) - 1/3*self.shear(b, pol=pol)

    def azimuthal_pressure(self, b, pol='U'):
        ''' Azimuthal pressure in GeV/fm**3. '''
        return self.pressure(b, pol=pol) - 1/3*self.shear(b, pol=pol)

    def symmetric_shear(self, b, pol='U'):
        ''' Symmetric shear in the r-theta directions, in GeV/fm**3. '''
        return 0*b # TODO ... maybe can't do in 1D

    def torsion_shear(self, b, pol='U'):
        ''' Antisymmetric shear in the r-theta direction, in GeV/fm**3.
        Need to multiply by sin(theta)*cos(theta).
        Also by -2 for pol==0.
        '''
        return self._bessel_transform(shearA_integgrand, b, self.sbar, self.mN)

    def isoradial_pressure(self, b, pol='U'):
        ''' Principal stress closest to the radial direction, in GeV/fm**3. '''
        return 0*b # TODO ... maybe can't do in 1D

    def isopolar_pressure(self, b, pol='U'):
        ''' Principal stress closest to the polar direction, in GeV/fm**3. '''
        return 0*b # TODO ... maybe can't do in 1D

    def radial_force(self, b, pol='U'):
        ''' Radial force density, in GeV/fm**4. '''
        if(pol=='U'):
            return self._bessel_transform(f0_integrand, b, self.cU, self.mN)
        elif(pol=='T'):
            # Need to multiply by 3/2*cos(theta)**2 - 1/2
            return (
                    1/2*self._bessel_transform(f2_integrand, b, self.cT1, self.cT2, self.sbar, self.mN)
                    +
                    3/10*self._bessel_transform(f3_integrand, b, self.cT1, self.sbar, self.mN)
                    )
        else:
            self._pol_error(pol)

    def polar_force(self, b, pol='U'):
        ''' Polar force density, in GeV/fm**4.
        Need to multiply by sin(theta)*cos(theta).
        Also by -2 for pol==0.
        '''
        return (
                - 1/2*self._bessel_transform(f2_integrand, b, self.cT1, self.cT2, self.sbar, self.mN)
                + 1/5*self._bessel_transform(f3_integrand, b, self.cT1, self.sbar, self.mN)
                )

    # Internal methods ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def _bessel_transform(self, integrand, *args):
        result = quad_vec(integrand, self.kmin, self.kmax,
                          args=args,
                          workers=8)[0]
        return result
