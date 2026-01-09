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

## Paper and citation

deupack was developed for the numerical calculations of:

- Wim Cosyn, Adam Freese and Alan Sosa,
  (in preparation)

Please cite the paper if you use this code in your research.

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
