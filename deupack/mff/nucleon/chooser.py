# chooser.py
# Created 2025.11.11 by Adam Freese
#
# This is modelled after Alan's wf chooser, and is meant to choose the nucleon MFFs

# import nucleon MFF modules
from . import ba as ba_mod
from . import hz as hz_mod
from . import misc as misc_mod
from . import mit as mit_mod
from . import point as point_mod

# Add mapping for convenient selection
NUCLEONFFS = {
        'mit' : (
            mit_mod.AN, mit_mod.JN, mit_mod.DN, misc_mod.cN, misc_mod.SN
            ),
        'hz' : (
            mit_mod.AN, mit_mod.JN, hz_mod.DN, misc_mod.cN, misc_mod.SN
            ),
        'ba' : (
            ba_mod.AN, ba_mod.JN, ba_mod.DN, misc_mod.cN, misc_mod.SN
            ),
        'point' : (
            point_mod.AN, point_mod.JN, point_mod.DN, misc_mod.cN, point_mod.SN
            )
        }

def choose_nff(nff):
    ''' Return a tuple (AN,JN,DN,cbarN,SN) according to a choice
    of nucleon MFFs.
    '''
    if isinstance(nff, str):
        key = nff.lower()
        if key in NUCLEONFFS:
            return NUCLEONFFS[key]
        raise ValueError(f"Unknown nff '{nff}', valid: {list(NUCLEONFFS.keys())}")
    # module-like object with attributes u,w,u1,...
    for attr in ('AN','JN','DN','cN','SN'):
        if not hasattr(nff, attr):
            raise ValueError("nff module must have attributes: AN,JN,DN,cN,SN")
    return (nff.AN,nff.JN,nff.DN,nff.cN,nff.SN)
