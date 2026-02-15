# chooser.py
# Created 2025.11.11 by Adam Freese
#
# This is modelled after Alan's wf chooser, and is meant to choose the nucleon EMTFFs

# import nucleon EMT-FF modules
from .nff import NFF
from .ba import nff_ba
from .hz import nff_hz, nff_hz_quark, nff_hz_gluon
from .mab import nff_mab
from .mit import nff_mit, nff_mit_quark, nff_mit_gluon
from .point import nff_point

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def choose_nff(nff):
    ''' Return a an NFF object based on a user choice (usually a string). '''
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
        # Quark-only and gluon-only choices
        elif(nff=='mitq'):
            _nff = nff_mit_quark()
        elif(nff=='mitg'):
            _nff = nff_mit_gluon()
        elif(nff=='hzq'):
            _nff = nff_hz_quark()
        elif(nff=='hzg'):
            _nff = nff_hz_gluon()
        else:
            raise ValueError("nff={} not recognized.".format(nff))
        return _nff
    raise TypeError("nff should be NFF or str.")
