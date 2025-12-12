# chooser.py
# Created 2025.11.10 (code by Alan Sosa, moved and modified by Adam Freese)
#
# Methods to choose the wave function, and return the appropriate functions.

# import wavefunction modules
from . import av18 as av18_mod
from . import paris as paris_mod
from . import CDbonn as cdbonn_mod

# Add mapping for convenient selection
WAVEFUNCTIONS = {
        'av18': (av18_mod.u, av18_mod.w, av18_mod.u1, av18_mod.w1,
                 av18_mod.u2, av18_mod.w2, av18_mod.u3, av18_mod.w3),
        'av18-s-only': (av18_mod.u, av18_mod.zero, av18_mod.u1, av18_mod.zero,
                 av18_mod.u2, av18_mod.zero, av18_mod.u3, av18_mod.zero),
        'av18-d-only': (av18_mod.zero, av18_mod.w, av18_mod.zero, av18_mod.w1,
                 av18_mod.zero, av18_mod.w2, av18_mod.zero, av18_mod.w3),
        'paris': (paris_mod.u, paris_mod.w, paris_mod.u1, paris_mod.w1,
                  paris_mod.u2, paris_mod.w2, paris_mod.u3, paris_mod.w3),
        'cdbonn': (cdbonn_mod.u, cdbonn_mod.w, cdbonn_mod.u1, cdbonn_mod.w1,
                  cdbonn_mod.u2, cdbonn_mod.w2, cdbonn_mod.u3, cdbonn_mod.w3)
        }

def choose_wf(wf):
    """Return a tuple (u,w,u1,w1,u2,w2,u3,w3) according to wf.
    wf can be a string 'av18'/'paris' or a module-like object.
    """
    if isinstance(wf, str):
        key = wf.lower()
        if key in WAVEFUNCTIONS:
            return WAVEFUNCTIONS[key]
        raise ValueError(f"Unknown wf '{wf}', valid: {list(WAVEFUNCTIONS.keys())}")
    # module-like object with attributes u,w,u1,...
    for attr in ('u','w','u1','w1','u2','w2','u3','w3'):
        if not hasattr(wf, attr):
            raise ValueError("wf module must have attributes: u,w,u1,w1,u2,w2,u3,w3")
    return (wf.u, wf.w, wf.u1, wf.w1, wf.u2, wf.w2, wf.u3, wf.w3)
