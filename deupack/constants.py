import numpy as np

# Basicl physical constants
hbar  = 0.197326968 # GeV-fm

# Masses
Md    = 1.87561294257 # deuteron mass (GeV)
mN    = 0.9389186795  # averaged nucleon mass (GeV)

# Misc constants
kappa = np.sqrt((2*mN-Md)*mN) / hbar # asymptotic decay length, fm**-1
