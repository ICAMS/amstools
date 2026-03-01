import os
import shutil
import pytest
from amstools.highthroughput.generate_ht_pipeline_setup import generate_pipeline
from amstools.highthroughput.utils import initialize_pipe
from ase.build import bulk


def mock_calculate_r_estimation(structure):
    return 1.0  # Mocked value for r_estimation


def test_pipeline_naming():
    pipeline_steps = [
        "static",
        "murnaghan",
        "relax",
        "murnaghan",
        "static",
        "relax-atomic",
        "static",
    ]

    # Mocked relaxation type
    relaxation_type = "full"

    # Generate the pipeline
    pipeline = generate_pipeline(
        pipeline_steps, mock_calculate_r_estimation(None), relaxation_type
    )

    # Extract the step names from the pipeline
    step_names = list(pipeline.steps.keys())

    # Expected step names based on the rules
    expected_step_names = [
        "static",
        "murnaghan",
        "mrn_relax",
        "mrn_rlx_murnaghan",
        "mrn_rlx_mrn_static",
        "mrn_rlx_mrn_relax-atomic",
        "mrn_rlx_mrn_rla_static",
    ]

    assert step_names == expected_step_names


from ase.calculators.emt import EMT
import numpy as np


def emt_calculator():
    """EMT calculator with mocked get_stress (EMT stress is unreliable)."""
    calc = EMT()

    def get_stress(*args, **kwargs):
        return np.zeros(6)

    calc.get_stress = get_stress
    return calc


def test_pipeline_with_emt_calculator():
    pipeline_steps = [
        "static",
        "murnaghan",
        "relax",
        "murnaghan",
        "static",
        "relax-atomic",
        "static",
    ]

    expected_step_names = [
        "static",
        "murnaghan",
        "mrn_relax",
        "mrn_rlx_murnaghan",
        "mrn_rlx_mrn_static",
        "mrn_rlx_mrn_relax-atomic",
        "mrn_rlx_mrn_rla_static",
    ]

    # Mocked relaxation type
    relaxation_type = "full"

    # Create an aluminum FCC structure
    atoms = bulk("Al", "fcc", a=4.05, cubic=True)

    # Generate the pipeline
    pipeline = generate_pipeline(
        pipeline_steps, mock_calculate_r_estimation(atoms), relaxation_type
    )

    # Define output directory for saving steps and pipeline JSON
    output_dir = "test_pipeline_output"
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # Initialize the pipeline
    initialize_pipe(pipeline, output_dir, emt_calculator(), atoms)

    # assert that the step names are as expected
    step_names = list(pipeline.steps.keys())
    print("Pipeline step names:", step_names)
    assert (
        step_names == expected_step_names
    ), f"Expected step names {expected_step_names}"

    # Save the pipeline JSON
    pipeline_json_filename = os.path.join(output_dir, "pipeline.json")
    pipeline.to_json(pipeline_json_filename)

    # Run the pipeline
    pipeline.run(verbose=True)

    # Check that the output directory and files exist
    assert os.path.exists(output_dir), "Output directory was not created."
    assert os.path.exists(pipeline_json_filename), "Pipeline JSON was not saved."
    for step_name in pipeline.steps.keys():
        step_dir = os.path.join(output_dir, step_name)
        assert os.path.exists(step_dir), f"Step directory {step_dir} was not created."

    # Clean up after the test
    # shutil.rmtree(output_dir)
