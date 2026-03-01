import os

from tqdm.auto import tqdm
import glob
import re
import pandas as pd


import getpass  # for getpass.getuser()
import socket  # for socket.gethostname()
from amstools.utils import load_yaml, logger
from amstools.pipeline import Pipeline

from ase.io import jsonio


def identify_environment():
    hostname = socket.gethostname()
    username = getpass.getuser()
    pid = str(os.getpid())
    return hostname, username, pid


def create_calc(calc_fname, if_enforce_pbc=False):
    from amstools.calculators.dft.aims import AMSFHIaims
    from amstools.calculators.dft.vasp import AMSVasp

    if not os.path.isfile(calc_fname):
        raise RuntimeError("Could not find calculator filename: {}".format(calc_fname))
    calc_setup = load_yaml(calc_fname)
    class_name = calc_setup.pop("calculator")
    if "name" in calc_setup:
        calc_setup.pop("name")
    if class_name.lower() in ["amsvasp", "vasp"]:
        calc_class = AMSVasp
        enforce_pbc = True
    elif class_name.lower() in ["amsfhiaims", "fhiaims"]:
        calc_class = AMSFHIaims
        enforce_pbc = False
    else:
        raise ValueError("Unsupported class option: {}".format(class_name))
    calc = calc_class(**calc_setup)
    if if_enforce_pbc:
        return calc, enforce_pbc
    else:
        return calc


def initialize_pipe(pipe, cur_work_dir, calc, structure):
    pipe.autosave = True
    pipe.path = cur_work_dir
    pipe.init_structure = structure
    if hasattr(structure, "enforce_pbc") and structure.enforce_pbc:
        calc = calc.copy()
        calc.kmesh_spacing = None
        calc.set_kmesh([1, 1, 1])
    pipe.engine = calc


def discover_pipeline_jobs(location_root="", force=False):
    location_path_search = os.path.join(location_root, "**", "pipeline.json")
    locations_suffix = location_root.replace("/", "__")
    locations_db = os.path.join(f"pipeline_locations_db__{locations_suffix}.csv")
    # if os.path.exists(locations_db) and ( force is False ):
    #    with open(locations_db, 'r') as f:
    #        elastic_jsons = f.read().splitlines()
    #    elastic_jsons = [line.strip() for line in elastic_jsons]
    #    return elastic_jsons
    logger.info("Finding the property jsons locations")
    elastic_jsons = glob.glob(location_path_search, recursive=True)
    with open(locations_db, "w") as f:
        f.writelines("\n".join(elastic_jsons))
    return elastic_jsons


def collect_pipelines(location_root=None, return_df=False, force_search=False):
    jsons = discover_pipeline_jobs(location_root=location_root, force=force_search)
    dest_pkl = os.path.relpath(location_root).replace("/", "__") + ".pkl.gzip"
    pipelines = []
    progress = tqdm(jsons, desc="reading pipelines")
    for file in progress:
        pipeline: Pipeline = Pipeline.read_json(file)
        try:
            name = os.path.relpath(pipeline.path)
            result_dict = {"name": name, "status": pipeline.status}
            result_dict.update(pipeline.named_steps_statuses)
            for step_name, pipeline_step in pipeline.steps.items():
                if not pipeline_step.finished:
                    logger.info(
                        f"Pipeline step {step_name} for {name} not finished",
                        exc_info=ValueError,
                    )
                    continue
                property_json_path = os.path.join(
                    pipeline_step.parent_path, step_name, "property.json"
                )
                if not os.path.exists(property_json_path):
                    continue
                property_dict = jsonio.read_json(property_json_path)
                if step_name == "relax":
                    if "optimized_structure" in property_dict:
                        result_dict.update(
                            {
                                "relax_optimized_structure": property_dict[
                                    "optimized_structure"
                                ]
                            }
                        )
                result_dict.update(
                    {
                        f"{step_name}_{value_name}": value
                        for value_name, value in property_dict["_VALUE"].items()
                    }
                )
            pipelines.append(result_dict)
        except Exception as E:
            logger.error(f"Error at reading {file}", exc_info=E)

    df: pd.core.frame.DataFrame = pd.DataFrame(pipelines)
    df.to_pickle(dest_pkl, compression="gzip")
    if return_df:
        return df
