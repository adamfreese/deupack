import pickle
from pathlib import Path
import re
import matplotlib.pyplot as plt
import lsqfit
import numpy as np


# from ...constants import mN, hbar


mN    = 0.93891875569 # averaged nucleon mass [arithmetic mean] (GeV)
pickle_path = Path(__file__).with_name("nucleonGFFs.pickle")

with pickle_path.open("rb") as handle:
    my_dictionary = pickle.load(handle)

print(my_dictionary.keys())

import gvar as gv

GFF = gv.gvar(my_dictionary["GFF_mean"],
              my_dictionary["GFF_cov"])
t = my_dictionary["minus_t_GeV2"]
order = my_dictionary["GFF_order"]



def extract_EMTff(prefix):
    """
    Pull out one form factor from the full data set.
    Returns the matching t and the corresponding form-factor values.
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


    # Form factor overrides ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


def AN(mt,A_0,cA,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    mf2    = 1.275 # from set I, see Eq. (51)
    mf2p   = 1.517 # from set I, see Eq. (51)
    mf2pp  = 1.565 # from set I, see Eq. (51)
    mf2ppp = 1.936 # from set I, see Eq. (51)
    # t =-k**2
    t=-mt
    num = A_0 - cA*t + c2*t**2
    den = (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den

def JN( mt,J_0,cJ,c2):
    ''' See Eq. (49) of Broniowski:2025ctl '''
    mf2    = 1.275 # from set I, see Eq. (51)
    mf2p   = 1.517 # from set I, see Eq. (51)
    mf2pp  = 1.565 # from set I, see Eq. (51)
    mf2ppp = 1.936 # from set I, see Eq. (51)
    # t =-k**2
    t=-mt
    num = 2*J_0 - cJ*t + c2*t**2
    den = 2 * (1-t/mf2**2) * (1-t/mf2p**2) * (1-t/mf2pp**2) * (1-t/mf2ppp**2)
    return num/den

def DN(mt,A_0,J_0,cA,cJ,c2):
    # t = -k**2

    t=-mt
    return -4*mN*( -mN*AN(mt,A_0,cA,c2) +ThetaP(mt,A_0) + (t/(8*mN))*(JN(mt,J_0,cJ,c2) +AN(mt,A_0,cA,c2)))/(3*(t))


def ThetaP(mt,theta_p):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    mf0    = 0.98 # see text above Eq. (51)
    msigma = 0.64 # central value for set I, see Eq. (52)
    # t = -k**2
    t=-mt
    num = mN*theta_p
    den = (1-t/mf0**2) * (1-t/msigma**2)
    return num/den



def cbar(mt,c_0):
    ''' See Eq. (50) of Broniowski:2025ctl '''
    mf0    = 0.98 # see text above Eq. (51)
    msigma = 0.64 # central value for set I, see Eq. (52)
    # t = -k**2
    t=-mt
    num = c_0
    den = (1-t/mf0**2) * (1-t/msigma**2)
    return num/den


# fitting function


# depends on scheme (already divided by mass)
theta_q = 0.08
theta_g= 0.92




prior = gv.BufferDict()

prior["A0q"] = gv.gvar(0.2,0.50)
prior["cAq"] = gv.gvar(0,5)
prior["c2q"] = gv.gvar(0,5)

# prior["cAg"] = gv.gvar(0,5)
# prior["c2g"] = gv.gvar(0,5)

prior["J0q"] = gv.gvar(0.25,0.50)
prior["cJq"] = gv.gvar(0,5)



# prior["cJg"] = gv.gvar(0,5)


# values found from BA previous fits to total form factors
cA     = 0.62 # central value for set I, see Eq. (52)
c2     = 0.15 # central value for set I, see Eq. (52)


cJ     = 0.87 # central value for set I, see Eq. (52)


def fcn(p):


    #sum rules
    A0g = 1.0 - p["A0q"]

    J0g = 0.5 - p["J0q"]

    #constraints from BA previous fit for total


    c2g = c2 -p["c2q"]
    cAg = cA -p["cAq"]
    cJg= cJ - p["cJq"]


    model = {}

    model["Aq"] = AN(
        np.array(tAu),
        p["A0q"],
        p["cAq"],
        p["c2q"]
    )

    model["Ag"] = AN(
        np.array(tAg),
        A0g,
        # p["cAg"],
        # p["c2g"]
        cAg,
        c2g
    )

    model["Jq"] = JN(
        np.array(tJu),
        p["J0q"],
        p["cJq"],
        p["c2q"]
    )

    model["Jg"] = JN(
        np.array(tJg),
        J0g,
        # p["cJg"],
        # p["c2g"]
        cJg,
        c2g
    )
    model["Dq"] = DN(
        np.array(tDu),
        p["A0q"],
        p["J0q"],
        p["cAq"],
        p["cJq"],
        p["c2q"]
    )

    model["Dg"] = DN(
        np.array(tDg),
        A0g,
        J0g,
        cAg,
        cJg,
        c2g
    )

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

print(fit)

mt = np.linspace(1e-12,max(tAg),300)

Aq_fit = AN(
    mt,
    fit.p["A0q"],
    fit.p["cAq"],
    fit.p["c2q"]
)

Ag_fit = AN(
    mt,
    1-fit.p["A0q"],
    cA-fit.p["cAq"],
    c2-fit.p["c2q"]
)

Jq_fit = JN(
    mt,
    fit.p["J0q"],
    fit.p["cJq"],
    fit.p["c2q"]
)

Jg_fit = JN(
    mt,
    0.5-fit.p["J0q"],
    cJ-fit.p["cJq"],
    c2-fit.p["c2q"]
)

Dq_fit = DN(
    mt,
    fit.p["A0q"],
    fit.p["J0q"],
    fit.p["cAq"],
    fit.p["cJq"],
    fit.p["c2q"]
)

Dg_fit = DN(
    mt,
    1-fit.p["A0q"],
    0.5-fit.p["J0q"],
    cA-fit.p["cAq"],
    cJ-fit.p["cJq"],
    c2-fit.p["c2q"]
)


#In D2 scheme
c_0q = (theta_q -fit.p["A0q"])/4.
c_0g = -c_0q


cq_fit = cbar(
    mt,c_0q
)


cg_fit = cbar(
    mt,c_0g
)


plt.figure()

plt.plot(mt,gv.mean(Aq_fit))
plt.fill_between(
    mt,
    gv.mean(Aq_fit)-gv.sdev(Aq_fit),
    gv.mean(Aq_fit)+gv.sdev(Aq_fit),
    alpha=0.3
)


plt.errorbar(
    tAu,
    gv.mean(Aq),
    yerr=gv.sdev(Aq),
    fmt='o',ecolor='r',
    capsize=3
)

plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$A_q$")

plt.savefig("DeupackPlots/A_qFit.pdf")


plt.figure()
plt.plot(mt,gv.mean(Ag_fit))
plt.fill_between(
    mt,
    gv.mean(Ag_fit)-gv.sdev(Ag_fit),
    gv.mean(Ag_fit)+gv.sdev(Ag_fit),
    alpha=0.3
)



plt.errorbar(
    tAg,
    gv.mean(Ag),
    yerr=gv.sdev(Ag),
    fmt='o',
    capsize=3
)


plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$A_g$")


plt.savefig("DeupackPlots/A_gFit.pdf")







plt.figure()
plt.plot(mt,gv.mean(Jq_fit))
plt.fill_between(
    mt,
    gv.mean(Jq_fit)-gv.sdev(Jq_fit),
    gv.mean(Jq_fit)+gv.sdev(Jq_fit),
    alpha=0.3
)


plt.errorbar(
    tJu,
    gv.mean(Jq),
    yerr=gv.sdev(Jq),
    fmt='o',ecolor='r',
    capsize=3
)

plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$J_q$")

plt.savefig("DeupackPlots/J_qFit.pdf")


plt.figure()
plt.plot(mt,gv.mean(Jg_fit))
plt.fill_between(
    mt,
    gv.mean(Jg_fit)-gv.sdev(Jg_fit),
    gv.mean(Jg_fit)+gv.sdev(Jg_fit),
    alpha=0.3
)



plt.errorbar(
    tJg,
    gv.mean(Jg),
    yerr=gv.sdev(Jg),
    fmt='o',
    capsize=3
)


plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$J_g$")

plt.savefig("DeupackPlots/J_gFit.pdf")





plt.figure()
plt.plot(mt,gv.mean(Dq_fit))
plt.fill_between(
    mt,
    gv.mean(Dq_fit)-gv.sdev(Dq_fit),
    gv.mean(Dq_fit)+gv.sdev(Dq_fit),
    alpha=0.3
)


plt.errorbar(
    tDu,
    gv.mean(Dq),
    yerr=gv.sdev(Dq),
    fmt='o',ecolor='r',
    capsize=3
)

plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$D_q$")

plt.savefig("DeupackPlots/D_qFit.pdf")


plt.figure()
plt.plot(mt,gv.mean(Dg_fit))
plt.fill_between(
    mt,
    gv.mean(Dg_fit)-gv.sdev(Dg_fit),
    gv.mean(Dg_fit)+gv.sdev(Dg_fit),
    alpha=0.3
)



plt.errorbar(
    tDg,
    gv.mean(Dg),
    yerr=gv.sdev(Dg),
    fmt='o',
    capsize=3
)


plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$D_g$")

plt.savefig("DeupackPlots/D_gFit.pdf")






plt.figure()
plt.plot(mt,gv.mean(cq_fit))
plt.fill_between(
    mt,
    gv.mean(cq_fit)-gv.sdev(cq_fit),
    gv.mean(cq_fit)+gv.sdev(cq_fit),
    alpha=0.3
)


plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$\bar{c}_q$")

plt.savefig("DeupackPlots/c_qFit.pdf")



plt.figure()
plt.plot(mt,gv.mean(cg_fit))
plt.fill_between(
    mt,
    gv.mean(cg_fit)-gv.sdev(cg_fit),
    gv.mean(cg_fit)+gv.sdev(cg_fit),
    alpha=0.3
)




plt.xlabel(r"$-t$ (GeV$^2$)")
plt.ylabel(r"$\bar{c}_g$")
plt.savefig("DeupackPlots/c_gFit.pdf")

