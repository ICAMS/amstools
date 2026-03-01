import os
import shutil

from ase.build import bulk

from amstools import *
from amstools.calculators.dft.vasp import AMSVasp

test_vasp_calc_dir = "test_vasp_calc_dir"
dft_test_dirname = os.path.dirname(__file__)
test_vasp_calc_dir = os.path.join(dft_test_dirname, test_vasp_calc_dir)
test_pipeline_dir = os.path.join(dft_test_dirname, "test_pipeline")
test_pipeline_json_dir = os.path.join(dft_test_dirname, "test_pipeline_json")
os.environ["ASE_VASP_COMMAND"] = os.path.join(
    dft_test_dirname, "mock_bin/mock_vasp_std"
)
os.environ["VASP_PP_PATH"] = os.path.join(dft_test_dirname, "mock_pp")


# todo test autosave, to_json


def test_pipeline_dft():
    if os.path.isdir(test_pipeline_dir):
        shutil.rmtree(test_pipeline_dir)

    calc = AMSVasp(xc="pbe", setups="recommended", ispin=1)
    atoms = bulk("Mo")
    atoms.calc = calc

    pipeline = Pipeline(
        steps=[
            StaticCalculator(),
            MurnaghanCalculator(
                volume_range=0.05,
                num_of_point=3,
                optimize_deformed_structure=False,
            ),
        ],
        init_structure=atoms,
        engine=calc,
        autosave=True,
        path=test_pipeline_dir,
    )

    pipeline.run()
    print(pipeline.status)
    print(pipeline.steps_statuses)

    assert pipeline.status == "finished"
    assert pipeline.steps_statuses == ["finished", "finished"]

    if os.path.isdir(test_pipeline_dir):
        shutil.rmtree(test_pipeline_dir)


def test_pipeline_dft_write_input_only():
    if os.path.isdir(test_pipeline_dir):
        shutil.rmtree(test_pipeline_dir)

    calc = AMSVasp(xc="pbe", setups="recommended", ispin=1)
    atoms = bulk("Mo")
    atoms.calc = calc

    pipeline = Pipeline(
        steps=[
            MurnaghanCalculator(
                volume_range=0.05,
                num_of_point=3,
                optimize_deformed_structure=False,
            ),
        ],
        init_structure=atoms,
        engine=calc,
        write_input_only=True,
        path=test_pipeline_dir,
    )

    pipeline.run()
    print(calc.paused_calculations_dirs)
    assert len(calc.paused_calculations_dirs) == 3

    if os.path.isdir(test_pipeline_dir):
        shutil.rmtree(test_pipeline_dir)


def test_pipeline_from_json():
    pipe = Pipeline.read_json(test_pipeline_json_dir + "/pipeline.json")
    print(pipe.status)
    print(pipe.steps_statuses)

    assert pipe.status == "finished"
    assert pipe.steps_statuses == ["finished", "finished"]
