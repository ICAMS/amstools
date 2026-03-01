# amstools

**Atomistic Modelling and Simulation tools for material properties**

A Python library for computing material properties from atomistic simulations, developed by the [AMS department at ICAMS, Ruhr University Bochum](https://www.icams.de).
Built on top of [ASE](https://wiki.fysik.dtu.dk/ase/) (Atomic Simulation Environment).

[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)](https://www.python.org)
[![License](https://img.shields.io/badge/license-ASL-orange)](LICENSE)
[![Tests](https://github.com/ICAMS/amstools/actions/workflows/tests.yml/badge.svg)](https://github.com/ICAMS/amstools/actions/workflows/tests.yml)

---

## Features

- **Structural optimization** — stepwise, isotropic, and specialized relaxations
- **Equations of State** — Polynomial EOS fitting
- **Elastic constants** — full elastic matrix via energy methods
- **Phonons** — phonon dispersion and density of states via Phonopy
- **Surface properties** — slab-based surface energy and decohesion calculations
- **Generalized stacking faults** — gamma line and gamma surface calculators
- **Transformation paths** — Bain paths and other lattice transformations
- **Point defects** — vacancy and interstitial formation energies
- **Convex hull** — thermodynamic stability and hull distance
- **High-throughput pipeline** — chain calculators, serialize results to JSON, resume interrupted runs
- **DFT integration** — VASP and FHI-aims wrappers
- **Materials Project** — download and cache reference structures via `mp-api`

---

## Installation

```bash
pip install .
```

Or for development (editable install):

```bash
pip install -e .
```

Verify installation:

```python
import amstools
print(amstools.__path__)
```

### Conda environment

```bash
conda env create -f environment.yml
conda activate ams
pip install -e .
```

---

## Quick Example

```python
from ase.build import bulk
from ase.calculators.emt import EMT
from amstools import MurnaghanCalculator

atoms = bulk('Al', 'fcc', a=4.05)
atoms.calc = EMT()

calc = MurnaghanCalculator(atoms)
calc.calculate()
print(calc.value)  
```

### Pipeline example

```python
from amstools import StepwiseOptimizer, MurnaghanCalculator, ElasticMatrixCalculator

pipeline = StepwiseOptimizer() + MurnaghanCalculator() + ElasticMatrixCalculator()
pipeline.run(atoms, atoms.calc)
```

---

## Tutorials

Interactive Jupyter notebooks are in the [`tutorial/`](tutorial/) folder,
covering equations of state, phonons, elastic constants, pipelines, and high-throughput workflows.

---

## Running Tests

```bash
python -m pytest test/ -v
```

Tests use the EMT toy potential — no DFT setup required.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding new property calculators,
writing tests, and understanding the pipeline architecture.

---

## Citation

If you use **amstools** in your research, please cite:

> Lysogorskiy Y. et al., *amstools: Atomistic Modelling and Simulation tools*,
> ICAMS, Ruhr University Bochum (2026). <https://github.com/ICAMS/amstools>

---

## License

amstools is published under the **Academic Software License (ASL)**.
Free for academic non-commercial use. Contact [yury.lysogorskiy@aceworks.works](mailto:yury.lysogorskiy@aceworks.works) for commercial use.
See [LICENSE](LICENSE) for full terms.

---

## Acknowledgements

Developed at the [Interdisciplinary Centre for Advanced Materials Simulation (ICAMS)](https://www.icams.de),Ruhr University Bochum, Germany.
