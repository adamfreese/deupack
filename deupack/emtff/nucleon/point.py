# point.py
# Created 2025.11.11 by Alan Sosa

from .nff import *

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_point(NFF):
    ''' EMT-FFs for a pointlike fermion. '''

    def __init__(self):
        super().__init__()
        return

    # Form factor overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN(self, k):
        ''' AN for point like fermion '''
        return 1 + k*0

    def JN(self, k):
        ''' JN for point like fermion '''
        return 0.5 + k*0

    def SN(self, k):
        ''' SN for point like fermion '''
        return 0.5 + k*0

    def DN(self, k):
        ''' DN for point like fermion '''
        return k*0

    def mass_radius_squared():
        ''' Mass radius for point like fermion '''
        return 0.0
