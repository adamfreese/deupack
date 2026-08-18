import numpy as np

# Basicl physical constants
hbar     = 0.197326968     # GeV-fm
alphaQED = 0.0072973525643 # fine structure constant

# Masses
mpi_p = 0.13957039    # charged pion mass (GeV)
mpi_0 = 0.1349768     # neutral pion mass (GeV)
mpi   = 0.13803919    # average pion mass (GeV)
mp    = 0.93827208943 # proton mass (GeV)
mn    = 0.93956542194 # neutron mass (GeV)
me    = 0.00051099895 # electron mass (GeV)
mN    = 0.93891875569 # averaged nucleon mass [arithmetic mean] (GeV)
mr    = 0.46945915515 # reduced proton-nuetron mass [half harmonic mean] (GeV)
Md    = 1.87561294200 # deuteron mass (GeV)

# In fm
mNfm  = mN / hbar

# Electromagnetic properties
mu_p  =  2.79284734463 # proton magnetic moment (nuclear magnetons)
mu_n  = -1.91304276    # neutron magnetic moment (nuclear magnetons)
mu_d  =  0.8574382335  # deuteron magnetic moment (nuclear magnetons)

# Misc derived constants
Ed    = mp + mn - Md                 # deuteron binding energy (GeV)
kappa = np.sqrt((2*mN-Md)*mN) / hbar # asymptotic decay length (fm**-1)
a0    = hbar / (me*alphaQED)         # Bohr radius (fm)
