# chooser.py
# Created 2025.11.11 by Adam Freese
#
# This is modelled after Alan's wf chooser, and is meant to choose the nucleon EMTFFs

# import nucleon EMT-FF modules
from .nff import NFF
from .ba import nff_ba
from .hz import nff_hz
from .mab import nff_mab
from .mit import nff_mit
from .point import nff_point

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# TODO: quark-only and gluon-only choices

def choose_nff(nff):
    ''' Return a tuple (AN,JN,DN,cbarN,SN) according to a choice
    of nucleon EMTFFs.
    '''
    if(isinstance(nff, NFF)):
        return nff
    elif(isinstance(nff, str)):
        if(nff=='ba'):
            _nff = nff_ba()
        elif(nff=='hz'):
            _nff = nff_hz()
        elif(nff=='mab'):
            _nff = nff_mab()
        elif(nff=='mit'):
            _nff = nff_mit()
        elif(nff=='point'):
            _nff = nff_point()
        else:
            raise ValueError("nff={} not recognized.".format(nff))
        return _nff
    raise TypeError("nff should be NFF or str.")
