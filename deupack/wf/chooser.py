# chooser.py
# Created 2025.11.10 (code by Alan Sosa, moved and modified by Adam Freese)
#
# Methods to choose the wave function, and return the appropriate functions.

from .av18   import dwf_av18
from .CDbonn import dwf_cdbonn
from .paris  import dwf_paris

def choose_wf(wf):
    """Return a tuple (u,w,u1,w1,u2,w2,u3,w3) according to wf.
    wf can be a string 'av18'/'paris' or a module-like object.
    """
    if(wf=='av18'):
        dwf = dwf_av18()
    elif(wf=='paris'):
        dwf = dwf_paris()
    elif(wf=='cdbonn'):
        dwf = dwf_cdbonn()
    else:
        raise ValueError("wf={} not recognized.".format(wf))
    return (dwf.u, dwf.w, dwf.u1, dwf.w1, dwf.u2, dwf.w2, dwf.u3, dwf.w3)
