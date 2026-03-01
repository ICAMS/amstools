# Copilot Instructions for amstools

## Project Overview

**amstools** — Atomistic Modelling and Simulation tools for computing material properties (elastic constants, phonons, defect energies, surface energies, equations of state, etc.). Built on top of ASE (Atomic Simulation Environment).

## Commands

```bash
pip install -e .                          # Dev install
python -m pytest test/                    # All tests
python -m pytest test/test_file.py        # Single file
python -m pytest test/test_file.py::func  # Single test
```

## Architecture

All property calculators inherit from `GeneralCalculator` (which inherits from `PipelineStep`):

```
PipelineStep (pipeline/pipelinestep.py)
  └─ GeneralCalculator (properties/generalcalculator.py)
       ├─ MurnaghanCalculator, ElasticMatrixCalculator, PhonopyCalculator, ...
```

### Adding a new calculator

1. Create file in `amstools/properties/`
2. Subclass `GeneralCalculator`
3. Set `property_name` (unique string) and `param_names` (list of custom `__init__` param names)
4. Implement three methods:
   - `generate_structures(verbose)` → `OrderedDict{name: Atoms}` — create structure variants from `self.basis_ref`
   - `get_structure_value(structure, name)` → `(result_dict, Atoms)` — calculate one structure
   - `analyse_structures(output_dict)` → writes to `self._value` — aggregate results
5. Export in `amstools/properties/__init__.py` and `amstools/__init__.py`
6. Write tests in `test/` using fixtures from `test/utils.py` (`atoms`, `calculator`)

**Simplest reference:** `amstools/properties/static.py` (~50 lines).

See `CONTRIBUTING.md` for a complete template with code examples.

### Key patterns

- Results stored in `self._value` (OrderedDict)
- `self.basis_ref` is a copy of the input atoms — never mutate it, always `.copy()`
- `param_names` drives serialization (`todict`/`fromdict`) — all custom `__init__` params must be listed
- Calculators accept both `(atoms, calculator)` and `(structure=, engine=)` styles
- Tests use EMT calculator (fast toy potential) — no DFT setup needed
- `Pipeline` orchestrates multiple calculators sequentially, serializes to `pipeline.json`

### Key directories

- `amstools/properties/` — all property calculator implementations
- `amstools/pipeline/` — pipeline framework (PipelineStep, Pipeline, scheduler)
- `amstools/calculators/dft/` — DFT engine wrappers (VASP, FHI-aims)
- `amstools/cli/` — CLI entry points
- `amstools/resources/` — atomic data, crystal structure prototypes
- `test/` — pytest suite; fixtures in `test/utils.py`
