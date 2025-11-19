# hz.py
# Created 2025.11.11 by Adam Freese (moved code from misc.py)
#
# This file contains formulas for the D form factor of a nucelon,
# as used by
#   Fangcheng He and Ismail Zahed
#   Physical Review C109 (2024) 045209
#   He:2023ogg

def DN(k):
    ''' The nucleon DN used by He and Zahed '''
    return DN_q(k) + DN_g(k)

def DN_q(k):
    ''' The nucleon DN used by He and Zahed, quark part '''
    D0 = -1.30
    Lambda = 0.81
    return D0 / (1 + (k/Lambda)**2)**2

def DN_g(k):
    ''' The nucleon DN used by He and Zahed, gluon part '''
    D0 = -1.275
    Lambda = 0.963
    return D0 / (1 + (k/Lambda)**2)**2
