# hz.py
# Created 2025.11.11 by Adam Freese (moved code from misc.py)

from ...constants import hbar

from .mit import nff_mit

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_hz(nff_mit):
    ''' Uses the MIT AN and JN, but overrides DN with a holographic result.
    The DN used here is from:
        Fangcheng He and Ismail Zahed
        Physical Review C109 (2024) 045209
        He:2023ogg
    '''

    def __init__(self):
        super().__init__()
        return

    # Form factor overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def DN(self, k):
        ''' The nucleon DN used by He and Zahed '''
        return self.DN_q(k) + self.DN_g(k)

    # Auxiliary functions ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def DN_q(self, k):
        ''' The nucleon DN used by He and Zahed, quark part '''
        D0 = -1.30
        Lambda = 0.81
        return D0 / (1 + (k/Lambda)**2)**2

    def DN_g(self, k):
        ''' The nucleon DN used by He and Zahed, gluon part '''
        D0 = -1.275
        Lambda = 0.963
        return D0 / (1 + (k/Lambda)**2)**2

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_hz_quark(nff_hz):
    ''' A quark-only variation on nff_hz. '''

    def __init__(self):
        super().__init__()
        return

    # Overrides to eliminate gluons ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_g(self, k):
        return k*0

    def JN_g(self, k):
        return k*0

    def DN_g(self, k):
        return k*0

    def cN_g(self, k):
        return k*0

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

class nff_hz_gluon(nff_hz):
    ''' A gluon-variation on nff_hz. '''

    def __init__(self):
        super().__init__()
        return

    # Overrides to eliminate quarks ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    def AN_q(self, k):
        return k*0

    def JN_q(self, k):
        return k*0

    def DN_q(self, k):
        return k*0

    def cN_q(self, k):
        return k*0

    def SN(self, k):
        # Note that gluons cannot contribute to SN
        return k*0
