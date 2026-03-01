import os
import numpy as np
import pytest

from amstools import StepwiseOptimizer, NearestNeighboursExpansionCalculator
from amstools.pipeline.generalstructure import GeneralStructure
from amstools.pipeline.pipeline import Pipeline
from amstools.pipeline.pipelinestep import PipelineEngine, PARTIALLY_FINISHED, FINISHED
from test.utils import atoms, calculator
from ase.calculators.emt import EMT

from amstools.properties import *


@pytest.fixture
def pipeline_calculator(calculator, monkeypatch):
    monkeypatch.setattr(calculator, "get_stress", lambda *args: np.zeros(6))
    return calculator


def test_general_structure(atoms):
    general_structure = GeneralStructure(atoms)
    assert general_structure.atoms == atoms


def test_pipeline_list(atoms, pipeline_calculator):
    pipeline = (
        StepwiseOptimizer()
        + NearestNeighboursExpansionCalculator()
        + MurnaghanCalculator(volume_range=0.05)
        + ElasticMatrixCalculator()
        + DefectFormationCalculator()
        + PhonopyCalculator()
        + TransformationPathCalculator()
    )

    expected_num_steps = 7
    assert len(pipeline.steps) == expected_num_steps

    assert not pipeline.is_finished()

    pipeline.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)

    assert pipeline.is_finished()
    assert pipeline.status == "finished"
    assert pipeline.steps_statuses == ["finished"] * expected_num_steps
    assert pipeline.named_steps_statuses == {
        "optimization": "finished",
        "energy_nn_distance": "finished",
        "murnaghan": "finished",
        "elastic_matrix": "finished",
        "defect": "finished",
        "phonons": "finished",
        "transformation_path": "finished",
    }
    assert isinstance(pipeline.steps["optimization"], StepwiseOptimizer)
    assert isinstance(
        pipeline.steps["transformation_path"], TransformationPathCalculator
    )


def test_pipeline_dict(atoms, pipeline_calculator):
    pipeline = Pipeline(
        steps={
            "Full relxation": StepwiseOptimizer(),
            "NNExp": NearestNeighboursExpansionCalculator(),
            "Murnaghan": MurnaghanCalculator(
                volume_range=0.05, num_of_point=7
            ),
            "Elastic": ElasticMatrixCalculator(),
            "Defect": DefectFormationCalculator(),
            "Phonons": PhonopyCalculator(),
            "Transformation_trigonal": TransformationPathCalculator(
                transformation_type="trigonal"
            ),
        }
    )
    expected_num_steps = 7
    assert len(pipeline.steps) == expected_num_steps

    assert not pipeline.is_finished()

    pipeline.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)

    assert pipeline.is_finished()

    murn_job = pipeline["Murnaghan"]
    assert len(murn_job.value["energy"]) == 7

    trig_job = pipeline["Transformation_trigonal"]
    assert trig_job.transformation_type == "trigonal"
    assert pipeline.status == "finished"
    assert pipeline.steps_statuses == ["finished"] * expected_num_steps
    assert pipeline.named_steps_statuses == {
        "Full relxation": "finished",
        "NNExp": "finished",
        "Murnaghan": "finished",
        "Elastic": "finished",
        "Defect": "finished",
        "Phonons": "finished",
        "Transformation_trigonal": "finished",
    }
    dict_repr = pipeline.todict()
    assert dict_repr is not None


def test_pipeline_surfaceenergy_step(atoms, pipeline_calculator):
    pipeline = Pipeline(
        steps=[
            StepwiseOptimizer(),
            MurnaghanCalculator(volume_range=0.05),
            SurfaceEnergyCalculator(
                surface_orientation="100",
                surface_name="X100_Y010_Z001_6at",
                fmax=0.5,
            ),
        ]
    )
    expected_num_steps = 3
    assert len(pipeline.steps) == expected_num_steps

    assert not pipeline.is_finished()

    pipeline.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)

    assert pipeline.is_finished()
    assert pipeline.status == "finished"
    assert pipeline.steps_statuses == ["finished"] * expected_num_steps
    assert pipeline.named_steps_statuses == {
        "optimization": "finished",
        "murnaghan": "finished",
        "surface_energy": "finished",
    }
    assert isinstance(pipeline.steps["optimization"], StepwiseOptimizer)
    assert isinstance(pipeline.steps["surface_energy"], SurfaceEnergyCalculator)


def test_pipeline_stackingfault_step(atoms, pipeline_calculator):
    pipeline = Pipeline(
        steps=[
            StepwiseOptimizer(),
            MurnaghanCalculator(volume_range=0.05),
            StackingFaultCalculator(fmax=0.5),
        ]
    )
    expected_num_steps = 3
    assert len(pipeline.steps) == expected_num_steps

    assert not pipeline.is_finished()

    pipeline.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)

    assert pipeline.is_finished()
    assert pipeline.status == "finished"
    assert pipeline.steps_statuses == ["finished"] * expected_num_steps
    assert pipeline.named_steps_statuses == {
        "optimization": "finished",
        "murnaghan": "finished",
        "stacking_fault": "finished",
    }
    assert isinstance(pipeline.steps["optimization"], StepwiseOptimizer)
    assert isinstance(pipeline.steps["stacking_fault"], StackingFaultCalculator)


def test_pipeline_tqha_step(atoms, pipeline_calculator):
    pipeline = Pipeline(
        steps=[
            StepwiseOptimizer(),
            MurnaghanCalculator(
                volume_range=0.05,
                optimize_deformed_structure=True,
            ),
            ThermodynamicQHACalculator(
                supercell_range=2,
                num_of_point=4,
                fit_order=2,
                q_space_sample=25,
                optimize_deformed_structure=True,
                fmax=0.1,
            ),
        ]
    )
    expected_num_steps = 3
    assert len(pipeline.steps) == expected_num_steps

    assert not pipeline.is_finished()
    pipeline.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)

    assert pipeline.is_finished()
    assert pipeline.status == "finished"
    assert pipeline.steps_statuses == ["finished"] * expected_num_steps
    assert pipeline.named_steps_statuses == {
        "optimization": "finished",
        "murnaghan": "finished",
        "qha": "finished",
    }
    assert isinstance(pipeline.steps["optimization"], StepwiseOptimizer)
    assert isinstance(
        pipeline.steps["qha"], ThermodynamicQHACalculator
    )


def test_restore_pipeline_engine_ignore_import_errors():
    engine = PipelineEngine(EMT())
    engine_dct = engine.todict()
    restored_engine = PipelineEngine.fromdict(engine_dct)
    assert restored_engine is not None

    engine_dct["calculator"]["__cls__"] = "SomeNoneExtistingClass"

    with pytest.raises(Exception):
        PipelineEngine.fromdict(engine_dct)

    broken_restored_engine_2 = PipelineEngine.fromdict(
        engine_dct, ignore_import_errors=True
    )
    assert broken_restored_engine_2 is None


def test_store_static_calculation_step_in_pipeline_json(atoms, pipeline_calculator):
    pipeline = Pipeline(
        steps={
            "static": StaticCalculator(),
            "opt": StepwiseOptimizer(),
        }
    )
    expected_num_steps = 2
    assert len(pipeline.steps) == expected_num_steps

    assert not pipeline.is_finished()

    pipeline.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)

    assert pipeline.is_finished()

    tmp_json = "tmp.json"
    if os.path.isfile(tmp_json):
        os.remove(tmp_json)
    assert not os.path.isfile(tmp_json)
    pipeline.to_json(tmp_json)
    assert os.path.isfile(tmp_json)
    pipeline_restores = Pipeline.read_json(tmp_json)
    os.remove(tmp_json)


def test_partially_finished_state(atoms, pipeline_calculator):
    pipeline = Pipeline(
        steps={
            "static": StaticCalculator(),
            "opt": StepwiseOptimizer(),
        }
    )
    pipeline.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)
    assert pipeline.status == FINISHED
    pipeline.steps["opt"].status = PARTIALLY_FINISHED
    assert pipeline.status == PARTIALLY_FINISHED


def test_pipeline_rerun_for(atoms, pipeline_calculator):
    pipeline_orig = Pipeline(
        steps={
            "Full relxation": StepwiseOptimizer(),
            "NNExp": NearestNeighboursExpansionCalculator(),
            "Murnaghan": MurnaghanCalculator(
                volume_range=0.05, num_of_point=7
            ),
            "Elastic": ElasticMatrixCalculator(),
            "Defect": DefectFormationCalculator(),
            "Transformation_trigonal": TransformationPathCalculator(
                transformation_type="trigonal"
            ),
        }
    )
    expected_num_steps = 6
    assert len(pipeline_orig.steps) == expected_num_steps

    assert not pipeline_orig.is_finished()

    pipeline_orig.run(init_structure=atoms, engine=pipeline_calculator, verbose=True)

    assert pipeline_orig.is_finished()

    new_calc = EMT()
    pipeline = pipeline_orig.rerun_for(engine=new_calc)

    assert pipeline.is_finished()

    for name, step in pipeline.steps.items():
        val = step.value
        print("{}: {}".format(name, val))

    murn_job = pipeline["Murnaghan"]
    print("murn_VALUE=", murn_job._value)
    assert len(murn_job._value["energy"]) == 7

    trig_job = pipeline["Transformation_trigonal"]
    print("trig_job VALUE=", trig_job._value)
    assert trig_job.transformation_type == "trigonal"
    assert pipeline.status == "finished"
    assert pipeline.steps_statuses == ["finished"] * expected_num_steps


def test_pipeline_add_operator(atoms, pipeline_calculator):
    step1 = StepwiseOptimizer()
    step2 = StaticCalculator()
    step3 = MurnaghanCalculator(volume_range=0.05)

    # Test step + step
    pipeline = step1 + step2
    assert isinstance(pipeline, Pipeline)
    assert len(pipeline.steps) == 2
    assert isinstance(pipeline.steps["optimization"], StepwiseOptimizer)
    assert isinstance(pipeline.steps["static"], StaticCalculator)

    # Test pipeline + step
    pipeline += step3
    assert len(pipeline.steps) == 3
    assert isinstance(pipeline.steps["murnaghan"], MurnaghanCalculator)

    # Test pipeline + pipeline
    pipeline2 = ElasticMatrixCalculator() + DefectFormationCalculator()
    combined_pipeline = pipeline + pipeline2
    assert len(combined_pipeline.steps) == 5
    assert isinstance(combined_pipeline.steps["elastic_matrix"], ElasticMatrixCalculator)
    assert isinstance(combined_pipeline.steps["defect"], DefectFormationCalculator)

    # Test step + pipeline
    step_start = NearestNeighboursExpansionCalculator()
    pipeline_start = step_start + pipeline2
    assert len(pipeline_start.steps) == 3
    assert isinstance(pipeline_start.steps["energy_nn_distance"], NearestNeighboursExpansionCalculator)
    assert isinstance(pipeline_start.steps["elastic_matrix"], ElasticMatrixCalculator)
    assert isinstance(pipeline_start.steps["defect"], DefectFormationCalculator)

    # Run the combined pipeline
    combined_pipeline.run(
        init_structure=atoms, engine=pipeline_calculator, verbose=True
    )
    assert combined_pipeline.is_finished()
