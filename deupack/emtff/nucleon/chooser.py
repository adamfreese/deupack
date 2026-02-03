# chooser.py
# Created 2025.11.11 by Adam Freese
#
# This is modelled after Alan's wf chooser, and is meant to choose the nucleon EMTFFs

# import nucleon EMT-FF modules
from .nff import NFF
from .ba import nff_ba
from .hz import nff_hz
from .mab import nff_mab
#from . import misc as misc_mod
from .mit import nff_mit
from .point import nff_point

# Add mapping for convenient selection
#NUCLEONFFS = {
#        'mitq' : (
#            mit_mod.AN_q, mit_mod.JN_q, mit_mod.DN_q, misc_mod.cN_q, misc_mod.SN_q
#            ),
#        'mitg' : (
#            mit_mod.AN_g, mit_mod.JN_g, mit_mod.DN_g, misc_mod.cN_g, misc_mod.SN_g
#            ),
#        'hzq' : (
#            mit_mod.AN_q, mit_mod.JN_q, hz_mod.DN_q, misc_mod.cN_q, misc_mod.SN_q
#            ),
#        'hzg' : (
#            mit_mod.AN_g, mit_mod.JN_g, hz_mod.DN_g, misc_mod.cN_g, misc_mod.SN_g
#            )
#        }

def choose_nff(nff):
    ''' Return a tuple (AN,JN,DN,cbarN,SN) according to a choice
    of nucleon EMTFFs.
    '''
    ## if isinstance(nff, str):
    ##     key = nff.lower()
    ##     if key in NUCLEONFFS:
    ##         return NUCLEONFFS[key]
    ##     raise ValueError(f"Unknown nff '{nff}', valid: {list(NUCLEONFFS.keys())}")
    ## # module-like object with attributes u,w,u1,...
    ## for attr in ('AN','JN','DN','cN','SN'):
    ##     if not hasattr(nff, attr):
    ##         raise ValueError("nff module must have attributes: AN,JN,DN,cN,SN")
    ## return (nff.AN,nff.JN,nff.DN,nff.cN,nff.SN)
    if(isinstance(nff, NFF)):
        return (nff.AN, nff.JN, nff.DN, nff.cN, nff.SN)
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
        return (_nff.AN, _nff.JN, _nff.DN, _nff.cN, _nff.SN)
    raise TypeError("nff should be NFF or str.")
