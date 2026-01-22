# chooser.py
# Created 2025.11.10 (code by Alan Sosa, moved and modified by Adam Freese)
#
# Methods to choose the wave function, and return the appropriate functions.

from .dwf import DWF

from .av18   import dwf_av18, dwf_av18_s_only, dwf_av18_d_only
from .CDbonn import dwf_cdbonn
from .paris  import dwf_paris

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def choose_wf(wf):
    ''' Just give back wf if it's a DWF.
    If wf is a string, then return the a DWF object of the appropriate class.
    Allowed strings are:
        av18
        cdbonn
        paris
    '''
    if(isinstance(wf, DWF)):
        return wf
    elif(isinstance(wf, str)):
        if(wf=='av18'):
            dwf = dwf_av18()
        elif(wf=='av18-s-only'):
            dwf = dwf_av18_s_only()
        elif(wf=='av18-d-only'):
            dwf = dwf_av18_d_only()
        elif(wf=='paris'):
            dwf = dwf_paris()
        elif(wf=='cdbonn'):
            dwf = dwf_cdbonn()
        else:
            raise ValueError("wf={} not recognized.".format(wf))
        return dwf
    raise TypeError("wf should be DWF or str.")
