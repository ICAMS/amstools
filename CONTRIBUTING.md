# Contributing to amstools

## Setup

```bash
# Option 1: conda (recommended)
conda env create -f environment.yml
conda activate ams
pip install -e .

# Option 2: pip only
pip install -e .
```

Verify installation:

```python
from amstools import MurnaghanCalculator
from ase.build import bulk
from ase.calculators.emt import EMT

atoms = bulk('Al', 'fcc')
atoms.calc = EMT()
m = MurnaghanCalculator(atoms)
m.calculate()
print("Setup works!", m.value)
```

## Running Tests

```bash
# All tests
python -m pytest test/

# Single test file
python -m pytest test/test_murnaghanCalculator.py

# Single test function
python -m pytest test/test_murnaghanCalculator.py::test_fit_murnaghan_run

# Verbose with printed output
python -m pytest test/ -v -s
```

Tests use EMT (a fast toy potential) so they run in seconds without any DFT setup.

## How to Add a New Property Calculator

This is the most common contribution: adding a calculator for a new material property.

### Step 1: Create the calculator file

Create a new file in `amstools/properties/`, e.g. `amstools/properties/my_property.py`.

Subclass `GeneralCalculator` and implement three methods:

```python
import logging
from collections import OrderedDict
from amstools.properties.generalcalculator import GeneralCalculator


class MyPropertyCalculator(GeneralCalculator):
    """Calculate some material property.

    :param atoms: ASE Atoms object with calculator attached
    :param my_param: Description of your parameter (default: 10)

    Usage:
        >>> atoms.calc = calculator
        >>> calc = MyPropertyCalculator(atoms, my_param=10)
        >>> calc.calculate()
        >>> print(calc.value)
    """

    property_name = "my_property"

    # List ALL custom __init__ parameter names here (not atoms/calculator).
    # This drives serialization and get_params_dict().
    param_names = ["my_param"]

    def __init__(self, atoms=None, my_param=10, **kwargs):
        GeneralCalculator.__init__(self, atoms, **kwargs)
        self.my_param = my_param

    def generate_structures(self, verbose=False):
        """Create the structures to be calculated.

        Returns an OrderedDict of {name: ASE Atoms}.
        self.basis_ref is the input structure (a copy of the original atoms).
        """
        structures = OrderedDict()
        # Example: generate scaled structures
        for i in range(self.my_param):
            s = self.basis_ref.copy()
            # ... modify structure ...
            structures[f"config_{i}"] = s
        return structures

    def get_structure_value(self, structure, name=None):
        """Run calculation on a single structure.

        Args:
            structure: ASE Atoms with calculator already attached
            name: the key from generate_structures()

        Returns:
            (result_dict, output_structure)
        """
        energy = structure.get_potential_energy()
        volume = structure.get_volume()
        return {"energy": energy, "volume": volume}, structure

    def analyse_structures(self, output_dict):
        """Aggregate results from all structures into self._value.

        Args:
            output_dict: {name: result_dict} from get_structure_value()
        """
        energies = [v["energy"] for v in output_dict.values()]
        self._value["energies"] = energies
        self._value["min_energy"] = min(energies)
```

**Reference:** `amstools/properties/static.py` is the simplest real example (~50 lines).

### Step 2: Export the calculator

Add your class to `amstools/properties/__init__.py`:

```python
from amstools.properties.my_property import MyPropertyCalculator
```

And to `amstools/__init__.py`'s `__all__` list:

```python
__all__ = [
    ...
    "MyPropertyCalculator",
]
```

### Step 3: Write tests

Create `test/test_myPropertyCalculator.py`:

```python
import pytest
from amstools import MyPropertyCalculator
from test.utils import atoms, calculator


@pytest.fixture
def my_calc(atoms, calculator):
    atoms.calc = calculator
    return MyPropertyCalculator(atoms)


def test_calculate(my_calc):
    my_calc.calculate()
    assert "min_energy" in my_calc.value


def test_generate_structures(my_calc):
    structures = my_calc.generate_structures()
    assert len(structures) == 10  # default my_param


def test_to_from_dict(my_calc):
    """Verify serialization roundtrip works."""
    my_calc.calculate()
    d = my_calc.todict()
    restored = MyPropertyCalculator.fromdict(d)
    assert restored.value == my_calc.value
```

Run your tests: `python -m pytest test/test_myPropertyCalculator.py -v`

### Step 4: Verify serialization

The `todict()`/`fromdict()` roundtrip is handled by `GeneralCalculator` automatically **if** you list all custom parameters in `param_names`. If your parameter contains non-JSON-serializable types (numpy arrays, custom objects), you may need to override `todict()`/`fromdict()`.

## Key Concepts for Contributors

### The three methods you implement

| Method | Purpose | Input | Output |
|--------|---------|-------|--------|
| `generate_structures()` | Create structure variants | `self.basis_ref` (the input atoms) | `OrderedDict{name: Atoms}` |
| `get_structure_value()` | Calculate one structure | single `Atoms` with `.calc` set | `(result_dict, Atoms)` |
| `analyse_structures()` | Aggregate all results | `{name: result_dict}` | writes to `self._value` |

### Where results live

- `self.value` (or `self._value`) — the final computed properties (OrderedDict)
- `self.output_structures_dict` — output structures keyed by name
- `self.basis_ref` — the input structure (copy of original atoms)
- `self.calculator` — the ASE calculator

### Using in a Pipeline

Your calculator works standalone and in pipelines with no extra code:

```python
from amstools import Pipeline, MyPropertyCalculator, MurnaghanCalculator

pipeline = Pipeline(
    steps=[MurnaghanCalculator(), MyPropertyCalculator(my_param=5)],
    init_structure=atoms,
    engine=atoms.calc,
)
pipeline.run()
```

### Test fixtures available in `test/utils.py`

- `calculator` — EMT calculator (fast, no DFT needed)
- `atoms` — Al FCC bulk (a=4.05) with EMT calculator attached
- `elements` — returns `"Al"`

## Common Pitfalls

- **Forgot `param_names`**: If you add `__init__` parameters but don't list them in `param_names`, serialization (`todict`/`fromdict`) will silently lose them.
- **Forgot `property_name`**: Each calculator needs a unique `property_name` string. It's used as the job directory name and in pipeline JSON.
- **Modifying `self.basis_ref` directly**: Always use `.copy()` when creating variants. `self.basis_ref` is the reference structure and should not be mutated.
- **Returning wrong tuple from `get_structure_value`**: Must return `(dict, Atoms)`. The dict goes into `output_dict`, the Atoms into `output_structures_dict`.
