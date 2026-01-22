# deupack

deupack (short for **deu**teron **pack**age) is a Python package for calculating
the mechanical structure of the deuteron in non-relativistic quantum mechanics,
using state-of-the-art deuteron wave functions and lattice QCD results
for the quark-gluon structure of nucleons as input.
The calculations performed by this package include
the energy-momentum tensor form factors (EMT-FFs) of the deuteron
and spatial distributions of stresses.

The initial release calculates the one-body contributions to the
EMT-FFs and stress distributions.

## Authors

deupack was written by Adam Freese (Center for Nuclear Femtography)
and Alan Sosa (Florida International University).

### Data from other authors

Deuteron EMT form factor data were graciously provided by Wim Cosyn, Fangcheng He and Julia Panteleeva
from the following respective works:
- Freese and Cosyn, Phys Rev. D 106 (2022) 114013 [Freese:2022yur](https://inspirehep.net/literature/2121166)
- He and Zahed, Phys. Rev. C 110 (2024) 014312 [He:2024vzz](https://inspirehep.net/literature/2747749)
- Panteleeva *et al.*, Acta Phys. Polon. B 56 (2025) 3-A19 [Panteleeva:2024abz](https://inspirehep.net/literature/2121166)

Additionally, the AV18 deuteron wave function is constructed from a cubic spline
using the publicly-available wave function table on
[Robert Wiringa's website](https://www.phy.anl.gov/theory/research/av18/).

## Paper and citation

deupack was developed for the numerical calculations of:
- Wim Cosyn, Adam Freese and Alan Sosa,
  (in preparation)

Please cite the paper if you use this code in your research.
You can also cite the repository from GitHub,
using the "cite this repository" option on the right.

## Installation

The package can be installed using pip.
To install, navigate to the main package directory
(the one containing `pyproject.toml`)
after downloading the repository,
and run:
```
pip install .
```

### Dependencies

- Python
- The dependencies listed in `pyproject.toml`

The latter should be automatically installed by pip.

## Future plans

deupack will be expanded on in the future as part of ongoing research.
Future plans for deupack include:

- Energy and energy flux densities
- Pion exchange contributions to EMT-FFs
