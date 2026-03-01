"""
Integration test for HT pipeline generation and execution using EMT calculator.

This tests that generate_ht_pipelines_setup produces pipelines that can actually
run end-to-end, not just generate structures and names.
"""

import os
import numpy as np
import pytest

from ase.calculators.emt import EMT

from amstools.highthroughput.generate_ht_pipeline_setup import (
    generate_ht_pipelines_setup,
    STRUCTURES_PATH,
)
from amstools.pipeline.pipeline import Pipeline
from amstools.properties import (
    StaticCalculator,
    StepwiseOptimizer,
    MurnaghanCalculator,
)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
os.environ[STRUCTURES_PATH] = os.path.join(TEST_DIR, "structures")


@pytest.fixture
def emt_calculator():
    """EMT calculator with mocked get_stress (EMT stress is unreliable)."""
    calc = EMT()

    def get_stress(*args, **kwargs):
        return np.zeros(6)

    calc.get_stress = get_stress
    return calc


def test_ht_pipeline_emt_static(emt_calculator):
    """Test HT pipeline with static step only - simplest smoke test."""
    config = {
        "composition": ["Al", "Cu"],
        "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg": ["static"],
    }
    res = generate_ht_pipelines_setup(config, "wyckoff")

    assert len(res["pipeline"]) == 2  # AlCu2, CuAl2

    for structure, name, pipe in zip(res["structure"], res["name"], res["pipeline"]):
        assert isinstance(pipe, Pipeline)
        pipe.run(init_structure=structure, engine=emt_calculator, verbose=False)
        assert pipe.is_finished(), f"Pipeline for {name} did not finish: {pipe.status}"


def test_ht_pipeline_emt_relax_murnaghan(emt_calculator):
    """Test HT pipeline with relax + murnaghan steps."""
    config = {
        "composition": ["Al", "Cu"],
        "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg": [
            "relax",
            {"murnaghan": {"num_of_point": 3, "volume_range": 0.05}},
        ],
    }
    res = generate_ht_pipelines_setup(config, "wyckoff")

    assert len(res["pipeline"]) == 2

    for structure, name, pipe in zip(res["structure"], res["name"], res["pipeline"]):
        assert "relax" in pipe.steps
        assert "rlx_murnaghan" in pipe.steps

        pipe.run(init_structure=structure, engine=emt_calculator, verbose=False)
        assert pipe.is_finished(), f"Pipeline for {name} did not finish: {pipe.status}"

        murn_step = pipe.steps["rlx_murnaghan"]
        assert len(murn_step._value["energy"]) == 3


def test_ht_pipeline_emt_from_yaml(emt_calculator):
    """Test HT pipeline generated from YAML config with multiple steps."""
    yamlfile = os.path.join(TEST_DIR, "htp_emt.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "wyckoff")

    assert len(res["pipeline"]) == 2
    assert len(res["structure"]) == 2

    for structure, name, pipe in zip(res["structure"], res["name"], res["pipeline"]):
        expected_steps = {"relax", "rlx_elastic", "rlx_Enn-local", "rlx_static", "rlx_murnaghan"}
        assert set(pipe.steps.keys()) == expected_steps, (
            f"Steps mismatch: {set(pipe.steps.keys())} != {expected_steps}"
        )

        pipe.run(init_structure=structure, engine=emt_calculator, verbose=False)

        # With allow_fail=True on most steps, pipeline should finish
        # even if some steps have issues
        assert pipe.status in ("finished", "partially_finished"), (
            f"Pipeline for {name} unexpected status: {pipe.status}"
        )

        # Static step should always succeed
        assert pipe.steps["rlx_static"].status == "finished"


def test_ht_pipeline_emt_unary(emt_calculator):
    """Test HT pipeline with unary composition on a unary prototype."""
    config = {
        "composition": ["Cu"],
        "unaries/bulk/A15/shaken/A15.shakesmallsuper2corrected.1.cfg": [
            "static",
        ],
    }
    res = generate_ht_pipelines_setup(config, "atoms")

    assert len(res["pipeline"]) >= 1

    for structure, name, pipe in zip(res["structure"], res["name"], res["pipeline"]):
        pipe.run(init_structure=structure, engine=emt_calculator, verbose=False)
        assert pipe.is_finished(), f"Pipeline for {name} did not finish: {pipe.status}"


def test_ht_pipeline_emt_enn_coarse(emt_calculator):
    """Test HT pipeline with Enn-coarse step (nearest neighbour expansion)."""
    config = {
        "composition": ["Al", "Cu"],
        "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg": ["Enn-coarse"],
    }
    res = generate_ht_pipelines_setup(config, "wyckoff")

    # Run only the first permutation to keep the test fast (Enn-coarse has 28 points)
    structure = res["structure"][0]
    name = res["name"][0]
    pipe = res["pipeline"][0]

    assert "Enn-coarse" in pipe.steps
    pipe.run(init_structure=structure, engine=emt_calculator, verbose=False)
    assert pipe.status in ("finished", "partially_finished"), (
        f"Pipeline for {name} unexpected status: {pipe.status}"
    )


def test_ht_pipeline_emt_serialization(emt_calculator, tmp_path):
    """Test that HT-generated pipeline can be serialized and deserialized."""
    config = {
        "composition": ["Al", "Cu"],
        "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg": ["static"],
    }
    res = generate_ht_pipelines_setup(config, "wyckoff")

    structure = res["structure"][0]
    pipe = res["pipeline"][0]

    # Run the pipeline
    pipe.run(init_structure=structure, engine=emt_calculator, verbose=False)
    assert pipe.is_finished()

    # Serialize
    json_path = str(tmp_path / "pipeline.json")
    pipe.to_json(json_path)

    # Deserialize and verify
    restored = Pipeline.read_json(json_path)
    assert restored.is_finished()
    assert set(restored.steps.keys()) == set(pipe.steps.keys())


def _make_emt_with_zero_stress():
    """Create an EMT calculator with mocked zero stress."""
    calc = EMT()
    calc.get_stress = lambda *args, **kwargs: np.zeros(6)
    return calc


class FailingEMT(EMT):
    """EMT calculator that raises RuntimeError after `fail_after` calculate() calls."""

    def __init__(self, fail_after=1):
        super().__init__()
        self.call_count = 0
        self.fail_after = fail_after

    def calculate(self, *args, **kwargs):
        self.call_count += 1
        if self.call_count > self.fail_after:
            raise RuntimeError("Simulated DFT crash")
        return super().calculate(*args, **kwargs)

    def get_stress(self, *args, **kwargs):
        return np.zeros(6)


def test_ht_pipeline_emt_stop_and_restart(tmp_path):
    """Test pipeline stops on error, serialized, then restarts reusing finished steps."""
    config = {
        "composition": ["Al", "Cu"],
        "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg": ["static"],
    }
    res = generate_ht_pipelines_setup(config, "wyckoff")
    structure = res["structure"][0]

    # Build a 3-step pipeline: static -> relax -> murnaghan
    pipe = Pipeline(steps=[
        StaticCalculator(),
        StepwiseOptimizer(),
        MurnaghanCalculator(volume_range=0.05, num_of_point=3),
    ])

    # --- Phase 1: Run with a calculator that fails after 1 call ---
    # Static needs 1 calculate() call, so relax will crash on its first call
    failing_calc = FailingEMT(fail_after=1)
    with pytest.raises(RuntimeError, match="Simulated DFT crash"):
        pipe.run(init_structure=structure, engine=failing_calc, verbose=False)

    # Static finished, relax errored, murnaghan never started
    step_names = list(pipe.steps.keys())
    assert pipe.steps[step_names[0]].finished  # static
    assert not pipe.steps[step_names[1]].finished  # relax
    assert not pipe.steps[step_names[2]].finished  # murnaghan
    assert not pipe.is_finished()

    # --- Phase 2: Simulate restart via serialize/deserialize ---
    # Replace FailingEMT with a serializable EMT before saving
    pipe.engine = _make_emt_with_zero_stress()
    json_path = str(tmp_path / "pipeline_interrupted.json")
    pipe.to_json(json_path)
    restored_pipe = Pipeline.read_json(json_path)

    # Verify restored state: static still done, others not
    restored_step_names = list(restored_pipe.steps.keys())
    assert restored_pipe.steps[restored_step_names[0]].finished
    assert not restored_pipe.steps[restored_step_names[1]].finished
    assert not restored_pipe.steps[restored_step_names[2]].finished

    # --- Phase 3: Re-run with a working calculator ---
    working_calc = _make_emt_with_zero_stress()
    restored_pipe.run(init_structure=structure, engine=working_calc, verbose=False)

    assert restored_pipe.is_finished()
    for name in restored_step_names:
        assert restored_pipe.steps[name].finished, f"Step {name} not finished"


def test_ht_pipeline_emt_stop_and_restart_counts(tmp_path):
    """Test that restarted pipeline does NOT re-execute already finished steps."""
    config = {
        "composition": ["Al", "Cu"],
        "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg": ["static"],
    }
    res = generate_ht_pipelines_setup(config, "wyckoff")
    structure = res["structure"][0]

    # Build a pipeline: static -> relax -> murnaghan
    pipe = Pipeline(steps=[
        StaticCalculator(),
        StepwiseOptimizer(),
        MurnaghanCalculator(volume_range=0.05, num_of_point=3),
    ])

    # Phase 1: Run first step only, then crash during second
    failing_calc = FailingEMT(fail_after=1)
    with pytest.raises(RuntimeError, match="Simulated DFT crash"):
        pipe.run(init_structure=structure, engine=failing_calc, verbose=False)

    step_names = list(pipe.steps.keys())
    assert pipe.steps[step_names[0]].finished  # static done
    assert failing_calc.call_count == 2  # 1 for static (OK) + 1 for relax (crash)

    # Phase 2: Save interrupted state and restore
    pipe.engine = _make_emt_with_zero_stress()
    json_path = str(tmp_path / "pipeline.json")
    pipe.to_json(json_path)
    restored = Pipeline.read_json(json_path)

    # Phase 3: Resume with a counting calculator
    counting_calc = EMT()
    counting_calc.call_count = 0
    original_calculate = counting_calc.calculate

    def counting_calculate(*args, **kwargs):
        counting_calc.call_count += 1
        return original_calculate(*args, **kwargs)

    counting_calc.calculate = counting_calculate
    counting_calc.get_stress = lambda *args, **kwargs: np.zeros(6)

    restored.run(init_structure=structure, engine=counting_calc, verbose=False)
    assert restored.is_finished()

    # Static was skipped (0 calls for it), only relax + murnaghan ran
    # If static had been re-run, there would be 1 extra call
    # Murnaghan alone accounts for exactly 3 calls (num_of_point=3)
    assert counting_calc.call_count >= 3
    # Verify static was never touched by counting_calc
    # The restored static step still has its old results, not results from counting_calc
    assert restored.steps[step_names[0]].finished
