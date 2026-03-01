import pytest
import numpy as np
from ase.calculators.emt import EMT
from amstools.properties.static import StaticCalculator
from amstools.properties.relaxation import StepwiseOptimizer
from amstools.pipeline.pipeline import Pipeline
from test.utils import atoms, calculator

def test_pipeline_getitem_int(atoms):
    step1 = StepwiseOptimizer()
    step2 = StaticCalculator()
    pipeline = step1 + step2
    
    # Test positive indexing
    assert pipeline[0] is step1
    assert pipeline[1] is step2
    
    # Test negative indexing
    assert pipeline[-1] is step2
    assert pipeline[-2] is step1
    
    # Test out of bounds
    with pytest.raises(IndexError):
        _ = pipeline[2]

def test_pipeline_get_final_structure(atoms):
    # Use real EMT calculator
    calc = EMT()
    
    # Create a 2-step pipeline
    # Using StaticCalculator which is essentially a static calculation
    pipeline = StepwiseOptimizer() + StaticCalculator()
    
    # Run the pipeline
    pipeline.run(init_structure=atoms, engine=calc, verbose=False)
    
    # Verify pipeline.get_final_structure() returns the structure from the last step
    final_sep = pipeline.get_final_structure()
    last_step_structure = pipeline[-1].get_final_structure()
    
    assert final_sep is not None
    assert np.allclose(final_sep.get_positions(), last_step_structure.get_positions())
    assert np.allclose(final_sep.get_cell(), last_step_structure.get_cell())
