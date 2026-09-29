import pickle
from pathlib import Path
import re
import lsqfit
import numpy as np


# from .nff import *

pickle_path = Path(__file__).with_name("nucleonGFFs.pickle")

with pickle_path.open("rb") as handle:
    my_dictionary = pickle.load(handle)

# print(my_dictionary.keys())

import gvar as gv


GFF = gv.gvar(my_dictionary["GFF_mean"],
              my_dictionary["GFF_cov"])
t = my_dictionary["minus_t_GeV2"]
order = my_dictionary["GFF_order"]



def extract_EMTff(prefix):
    """
    Pull out one form factor from the full data set.
    Returns the matching t and the corresponding form-factor values.
    From Lattice Data provided by Dimitra Pefkou
    """

    matching_indices = []
    matching_t_values = []

    for i, label in enumerate(order):
        if label.startswith(prefix + "_t"):

            index_in_t = int(re.search(r"\[(\d+)\]", label).group(1))

            matching_indices.append(i)
            matching_t_values.append(t[index_in_t])

    values = GFF[matching_indices]
    return matching_t_values, values


tAg, Ag = extract_EMTff("Ag")

tAu, Au = extract_EMTff("Au")
tAd, Ad = extract_EMTff("Ad")
tAs, As = extract_EMTff("As")

tJg, Jg = extract_EMTff("Jg")

tJu, Ju = extract_EMTff("Ju")
tJd, Jd = extract_EMTff("Jd")
tJs, Js = extract_EMTff("Js")

tDg, Dg = extract_EMTff("Dg")

tDu, Du = extract_EMTff("Du")
tDd, Dd = extract_EMTff("Dd")
tDs, Ds = extract_EMTff("Ds")


# quark contributions
Aq= Au+Ad+As
Jq= Ju+Jd+Js
Dq= Du+Dd+Ds





''' Nucleon EMT-FFs from the meson dominance model of:
        Masjuan, Ruiz Arriola and Broniowski
        Phys. Rev. D 87 (2013) 014005
        Masjuan:2012sk
    Editted to have separation between quarks and gluons
    quark gluon separated functions by Adam Freese
'''


mN    = 0.970 # mass used for nucleon because of lattice pion mass difference from real mass (GeV)
mf0    = 0.98 # see text above Eq. (51)

mf0p= 1.250  #from PDG


# masses of mesons from set I
mf2    = 1.275 # from set I, see Eq. (51)
mf2p   = 1.517 # from set I, see Eq. (51)
mf2pp  = 1.565 # from set I, see Eq. (51)
mf2ppp = 1.936 # from set I, see Eq. (51)
msigma = 0.64 # central value for set I, see Eq. (52)

# mass of mesons from set II
# mf2    = 1.275 # from set II, see Eq. (51)
# mf2p   = 1.430 # from set II, see Eq. (51)
# mf2pp  = 1.517 # from set II, see Eq. (51)
# mf2ppp = 1.565 # from set II, see Eq. (51)

# msigma = 0.64 # central value for set II, see Eq. (52)

def AN1(mt,A_0,cA,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    # t =-k**2
    t=-mt
    num = A_0 - cA*t + c2*t**2
    den = (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den

def JN1( mt,J_0,cJ,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    # t =-k**2
    t=-mt
    num = 2*J_0 - cJ*t + c2*t**2
    den = 2 * (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den



def ThetaP(mt,theta_p,m_sigma=msigma):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    t=-mt
    num = mN*theta_p
    den = (1-t/mf0**2) * (1-t/m_sigma**2)
    return num/den


def newThetaP(mt,theta_p,c2theta,m_sigma=msigma):
    '''new scalar form factor we do not use this for fits (only slightly improves fits)'''
    t=-mt
    num = mN*theta_p+ c2theta*t
    den = (1-t/mf0**2) * (1-t/m_sigma**2)*(1-t/mf0p**2)
    return num/den





def cbar(mt,c_0,m_sigma=msigma):
    t=-mt
    num = c_0
    den = (1-t/mf0**2) * (1-t/m_sigma**2)
    return num/den




def DN1(mt,A_0,J_0,cA,cJ,c2,m_sigma=msigma):
    t=-mt
    A = AN1(mt,A_0,cA,c2)
    B = 2*JN1(mt,J_0,cJ,c2) -A
    theta = ThetaP(mt,A_0,m_sigma)
    return -1./(3.*t)*( 4*mN**2*(theta/mN-A) -t*B)

# new D term based off of new scalar form factor (only slightly improves fits so we do not use it)
def new_DN1(mt,A_0,J_0,cA,cJ,c2,c2theta,m_sigma):
    t=-mt
    c0=0.12 #doesn't matter what this is it cancels anyway here
    return -4*mN*( -mN*AN1(mt,A_0,cA,c2) +newThetaP(mt,A_0+4*c0,c2theta,m_sigma) -4*mN*cbar(mt,c0,m_sigma) + (t/(4*mN))*(2*JN1(mt,J_0,cJ,c2) -AN1(mt,A_0,cA,c2)))/(3*(t))


# values found from BA previous fits to total form factors
cA     = 0.62 # central value for set I, see Eq. (52)
c2     = 0.15 # central value for set I, see Eq. (52)
cJ     = 0.87 # central value for set I, see Eq. (52)

# cA     = 0.83 # central value for set II, see Eq. (52)
# c2     = 0.25 # central value for set II, see Eq. (52)
# cJ     = 1.12 # central value for set II, see Eq. (52)



# fitting function


# depends on scheme (already divided by mass)
theta_q = 0.08
theta_g= 0.92




# priors for fitting to quark and gluon separated form factors with constraints from original BA fit
# this is so that when added together we get same as original BA fits
prior = gv.BufferDict()

prior["A0q"] = gv.gvar(0.2,0.50)
prior["cAq"] = gv.gvar(0,5)
prior["c2q"] = gv.gvar(0,5)


prior["J0q"] = gv.gvar(0.25,0.50)
prior["cJq"] = gv.gvar(0,5)



# priors for fitting separately to quark and gluon separated form factors
prior2 = gv.BufferDict()

prior2["A0q"] = gv.gvar(0.2,0.50)
prior2["cAq"] = gv.gvar(0,5)
prior2["c2q"] = gv.gvar(0,5)

prior2["cAg"] = gv.gvar(0,5)
prior2["c2g"] = gv.gvar(0,5)

prior2["J0q"] = gv.gvar(0.25,0.50)
prior2["cJq"] = gv.gvar(0,5)
prior2["cJg"] = gv.gvar(0,5)

prior2["mSigma"] = gv.gvar(0,5)


# new parameters if doing fit with new scalar form factor and new D term
# prior["c2thetaq"] = gv.gvar(-5,5)

# prior["c2thetag"] = gv.gvar(-5,5)


# fit function for fitting quark and gluon parts with BA constraints
def fcn(p):


    #sum rules
    A0g = 1.0 - p["A0q"]

    J0g = 0.5 - p["J0q"]


    #constraints from BA previous fit for total


    c2g = c2 -p["c2q"]
    cAg = cA -p["cAq"]
    cJg= cJ - p["cJq"]

    model = {}

    model["Aq"] = AN1(
        np.array(tAu),
        p["A0q"],
        p["cAq"],
        p["c2q"]
    )

    model["Ag"] = AN1(
        np.array(tAg),
        A0g,
        # p["cAg"],
        # p["c2g"]
        cAg,
        c2g
    )

    model["Jq"] = JN1(
        np.array(tJu),
        p["J0q"],
        p["cJq"],
        p["c2q"]
    )

    model["Jg"] = JN1(
        np.array(tJg),
        J0g,
        # p["cJg"],
        # p["c2g"]
        cJg,
        c2g
    )
    model["Dq"] = DN1(
        np.array(tDu),
        p["A0q"],
        p["J0q"],
        p["cAq"],
        p["cJq"],
        p["c2q"]
    )

    model["Dg"] = DN1(
        np.array(tDg),
        A0g,
        J0g,
        cAg,
        cJg,
        c2g
    )

    # model["Dq"] = new_DN1(
    #     np.array(tDu),
    #     p["A0q"],
    #     p["J0q"],
    #     p["cAq"],
    #     p["cJq"],
    #     p["c2q"],
    #     p["c2thetaq"]
    # )

    # model["Dg"] = new_DN1(
    #     np.array(tDg),
    #     A0g,
    #     J0g,
    #     cAg,
    #     cJg,
    #     c2g,
    #     p["c2thetag"]
    # )

    return model








# fit function for fitting quark and gluon parts separately
def fcn2(p):


    #sum rules
    A0g = 1.0 - p["A0q"]

    J0g = 0.5 - p["J0q"]


    #constraints from BA previous fit for total


    model = {}

    model["Aq"] = AN1(
        np.array(tAu),
        p["A0q"],
        p["cAq"],
        p["c2q"]
    )

    model["Ag"] = AN1(
        np.array(tAg),
        A0g,
        p["cAg"],
        p["c2g"]
    )

    model["Jq"] = JN1(
        np.array(tJu),
        p["J0q"],
        p["cJq"],
        p["c2q"]
    )

    model["Jg"] = JN1(
        np.array(tJg),
        J0g,
        p["cJg"],
        p["c2g"]
    )
    model["Dq"] = DN1(
        np.array(tDu),
        p["A0q"],
        p["J0q"],
        p["cAq"],
        p["cJq"],
        p["c2q"],
        p["mSigma"]
    )

    model["Dg"] = DN1(
        np.array(tDg),
        A0g,
        J0g,
        p["cAg"],
        p["cJg"],
        p["c2g"],
        p["mSigma"]
    )

    # model["Dq"] = new_DN1(
    #     np.array(tDu),
    #     p["A0q"],
    #     p["J0q"],
    #     p["cAq"],
    #     p["cJq"],
    #     p["c2q"],
    #     p["c2thetaq"]
    # )

    # model["Dg"] = new_DN1(
    #     np.array(tDg),
    #     A0g,
    #     J0g,
    #     cAg,
    #     cJg,
    #     c2g,
    #     p["c2thetag"]
    # )

    return model







data = {
    "Aq": Aq,
    "Ag": Ag,
    "Jq": Jq,
    "Jg": Jg,
    "Dq":Dq,
    "Dg":Dg
}

fit = lsqfit.nonlinear_fit(
    data=data,
    prior=prior,
    fcn=fcn
)




data2 = {
    "Aq": Aq,
    "Ag": Ag,
    "Jq": Jq,
    "Jg": Jg,
    "Dq":Dq,
    "Dg":Dg
}


fit2 = lsqfit.nonlinear_fit(
    data=data2,
    prior=prior2,
    fcn=fcn2
)


print(fit)



print(fit2)



# Fit 1 is result of keeping the form factors consistent with BA (q+g = total BA)


# Least Squares Fit:
#   chi2/dof [dof] = 0.92 [200]    Q = 0.8    logGBF = 419.44

# Parameters:
#             A0q   0.570 (14)     [  0.20 (50) ]  
#             cAq   0.334 (30)     [    0 ± 5.0 ]  
#             c2q   0.161 (21)     [    0 ± 5.0 ]  
#             J0q   0.277 (11)     [  0.25 (50) ]  
#             cJq   0.422 (38)     [    0 ± 5.0 ]  

# Settings:
#   svdcut/n = 1e-12/0    tol = (1e-08,1e-10*,1e-10)    (itns/time = 2/0.1s)
#   fitter = scipy_least_squares    method = trf

# Fit 2 is result of fitting the quark and gluon form factors separately
# Least Squares Fit:
#   chi2/dof [dof] = 0.89 [200]    Q = 0.87    logGBF = 403.29

# Parameters:
#             A0q   0.571 (14)     [  0.20 (50) ]  
#             cAq   0.237 (61)     [    0 ± 5.0 ]  
#             c2q   0.206 (50)     [    0 ± 5.0 ]  
#             cAg   0.232 (39)     [    0 ± 5.0 ]  
#             c2g   0.015 (25)     [    0 ± 5.0 ]  
#             J0q   0.276 (11)     [  0.25 (50) ]  
#             cJq   0.329 (72)     [    0 ± 5.0 ]  
#             cJg   0.396 (46)     [    0 ± 5.0 ]  
#          mSigma   0.579 (39)     [    0 ± 5.0 ]  

# Settings:
#   svdcut/n = 1e-12/0    tol = (1e-08,1e-10,1e-10*)    (itns/time = 7/0.1s)
#   fitter = scipy_least_squares    method = trf
