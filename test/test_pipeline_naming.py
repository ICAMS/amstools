import pytest
from amstools.pipeline.pipeline import Pipeline
from amstools.properties.relaxation import StepwiseOptimizer
from amstools.properties.murnaghan import MurnaghanCalculator
from amstools.properties.transformationpath import TransformationPathCalculator

def test_pipeline_automatic_naming():
    # Test naming from list
    pipeline = Pipeline(steps=[
        StepwiseOptimizer(),
        MurnaghanCalculator(),
        TransformationPathCalculator(transformation_type="tetragonal"),
        TransformationPathCalculator(transformation_type="orthogonal")
    ])

    names = list(pipeline.steps.keys())
    assert names == [
        "optimization",
        "murnaghan",
        "transformation_path",
        "transformation_path_1"
    ]

def test_pipeline_add_naming():
    # Test naming from + operator
    p = StepwiseOptimizer() + MurnaghanCalculator()
    assert list(p.steps.keys()) == ["optimization", "murnaghan"]

    p += StepwiseOptimizer()
    assert list(p.steps.keys()) == ["optimization", "murnaghan", "optimization_1"]

def test_pipeline_explicit_naming():
    # Test explicit naming
    pipeline = Pipeline(steps=[
        StepwiseOptimizer(name="Initial_Relax"),
        StepwiseOptimizer(name="Final_Relax"),
    ])

    assert list(pipeline.steps.keys()) == ["Initial_Relax", "Final_Relax"]
    assert pipeline.steps["Initial_Relax"].name == "Initial_Relax"
    assert pipeline.steps["Final_Relax"].name == "Final_Relax"

def test_pipeline_mixed_naming():
    # Test mixed explicit and automatic naming
    p = StepwiseOptimizer(name="custom") + StepwiseOptimizer()
    assert list(p.steps.keys()) == ["custom", "optimization"]

    p += StepwiseOptimizer(name="collision")
    assert list(p.steps.keys()) == ["custom", "optimization", "collision"]

    p += StepwiseOptimizer(name="collision")
    assert list(p.steps.keys()) == ["custom", "optimization", "collision", "collision_1"]
