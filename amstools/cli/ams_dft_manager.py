#!/usr/bin/env python

import os
import sys
import logging
import pandas as pd

NAME = "name"
FORMULA = "formula"
ASE_ATOMS = "ase_atoms"
UNIQUE_PATH = "UNIQUE_PATH"

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger()

import argparse
import time

from ase.db import connect

import amstools
from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.utils import (
    collect_raw_data,
    load_amstools_queues_setup,
    load_state_dict,
    save_state_dict,
    lock_file,
)
from amstools.pipeline.scheduler import get_current_running_job_ids
from amstools.highthroughput.utils import create_calc

# predefined constants

SLEEP_INTERVAL = 30  # seconds


def aggregate_calculations_status(state_dict):
    is_any_submitted_calculations = False
    is_all_finished = True
    for cur_dir, cur_state in state_dict.items():
        status = cur_state.get("status")
        if status != "finished":
            is_all_finished = False
        if status == "submitted":
            is_any_submitted_calculations = True
    return is_all_finished, is_any_submitted_calculations


def generate_path(ind, row, name_column=NAME):
    if name_column in row:
        cur_structure_path = str(row[name_column])
    elif FORMULA in row:
        cur_structure_path = "{}__{}".format(ind, row[FORMULA])
    else:
        formula = row[ASE_ATOMS].get_chemical_formula()
        cur_structure_path = "{}__{}".format(ind, formula)
    return cur_structure_path


def convert_ase_db_to_dataframe(db_or_filename):
    if isinstance(db_or_filename, str):
        db = connect(db_or_filename)
    else:
        db = db_or_filename
    db_rows = list(db.select())

    ids = []
    formulas = []
    ase_atoms_list = []

    for row in db_rows:
        ids.append(row.id)
        formulas.append(row.formula)
        ase_atoms_list.append(row.toatoms())

    df = pd.DataFrame({ASE_ATOMS: ase_atoms_list, FORMULA: formulas}, index=ids)
    return df


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="ams_dft_manager",
        description="AMS DFT custody utility.\n"
        + "\tversion: {}".format(amstools.__version__),
    )

    parser.add_argument(
        "-c",
        "--calculator",
        help="calculator JSON file",
        dest="calc_filename",
        default="calculator.json",
    )

    parser.add_argument(
        "-db",
        "--database",
        help="ASE SQlite3 DB file name (.db) or pandas DataFrame .pckl.gzip (with unique `name` column)",
        dest="db_filename",
    )

    parser.add_argument(
        "-d",
        "--daemon",
        help="Running infinitely until all pipelines are finished or failed",
        action="store_true",
        dest="daemon_mode",
    )

    parser.add_argument(
        "-p",
        "--path",
        help="Working dir for DFT calculations",
        dest="dft_dir",
        default=None,
    )

    parser.add_argument(
        "-r",
        "--reset-errors",
        help="force to reset error status to submitted and try to recollect the data",
        dest="reset_errors",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-n",
        "--name-column",
        type=str,
        help="Unique name column. Default: {}".format(NAME),
        default=NAME,
    )

    parser.add_argument(
        "-l",
        "--log",
        help="log filename, default: log.txt",
        type=str,
        default="log.txt",
    )

    parser.add_argument(
        "-o",
        "--output",
        help="file name of collected pandas DataFrame (pckl.gzip)",
        type=str,
        default="collected_df.pckl.gzip",
    )

    parser.add_argument(
        "-dr",
        "--dry-run",
        help="Dry run, i.e. do not submit actual calculations to the queue_name,"
        " but update state_dict.json file",
        dest="dry_run",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-q",
        "--queue-name",
        help="submit calculations to the queue_name. Options: <QUEUE_NAME> (~/.amstools::queues[QUEUE_NAME])",
        type=str,
        default=None,
        dest="queue_name",
    )

    parser.add_argument(
        "-s",
        "--static",
        help="Static DFT calculation (default)",
        dest="static",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "-a",
        "--atomic-opt",
        help="Atomic-only relaxation",
        dest="atomic",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "-f",
        "--full-opt",
        help="Full (atomic and cell) relaxation",
        dest="full",
        action="store_true",
        default=False,
    )

    args_parse = parser.parse_args(args)

    # default or input parameter
    calc_filename = args_parse.calc_filename
    # input parameter
    db_filename = args_parse.db_filename
    dft_dir = args_parse.dft_dir
    collected_file = args_parse.output
    dry_run = args_parse.dry_run
    name_column = args_parse.name_column
    queue_name = args_parse.queue_name
    daemon_mode = args_parse.daemon_mode
    static = args_parse.static
    atomic = args_parse.atomic
    full = args_parse.full

    if dft_dir is None:
        dft_dir = os.path.dirname(os.path.abspath(db_filename))
    reset_errors = args_parse.reset_errors

    if "log" in args_parse:
        log_file_name = args_parse.log
        logger.info("Redirecting log into file {}".format(log_file_name))
        fileh = logging.FileHandler(log_file_name, "a")
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fileh.setFormatter(formatter)
        logger.addHandler(fileh)

    msg = """Running AMS DFT manager:
    calculator : {}
    database   : {}
    working dir: {}
    queue_name : {}
""".format(
        calc_filename, db_filename, dft_dir, queue_name
    )
    logger.info(msg)

    try:
        # load settings from standard location
        queues = load_amstools_queues_setup()
        if len(queues) == 0:
            raise RuntimeError("No $HOME/.amstools::queues configured")
        if queue_name is None or queue_name == "DEFAULT":
            default_queue_name = next(iter(queues))
            logger.info("Default queue_name settings: {}".format(default_queue_name))
            queue_options = queues[default_queue_name]
        else:
            queue_options = queues[queue_name]
    except Exception as e:
        if dry_run:
            queue_options = {}
        else:
            logger.error("Can't load queue settings: {}".format(str(e)))
            raise e

    # load calculator
    logger.info("Try to load calculator from {}".format(calc_filename))
    if calc_filename.endswith(".json"):
        calc = AMSDFTBaseCalculator.load(calc_filename)
    elif calc_filename.endswith(".yaml"):
        calc = create_calc(calc_filename)
    logger.info("DFT calculator type: {}".format(type(calc)))

    # setup calculator: static, atomic, full
    if full:
        calc.optimize_full(max_steps=100, ediffg=-0.005)
        logger.info("Full relaxation (cell, atomic)")
    elif atomic:
        calc.optimize_atoms_only(max_steps=100, ediffg=-0.005)
        logger.info("Atomic-only relaxation ")
    elif static:
        calc.static_calc()
        logger.info("Static calculation")

    logger.info("Try to load ASE DB from {}".format(db_filename))
    try:
        df = pd.read_pickle(db_filename, compression="gzip")
        logger.info("Loaded as pandas DataFrame pckl.gzip")
    except Exception as e:
        try:
            df = convert_ase_db_to_dataframe(db_filename)
            logger.info("Loaded as ASE DB")
        except Exception as ee:
            logger.error(
                "Couldn't load {} - neither pandas DataFrame in pckl.gzip format ({}) nor ASE DB file ({})".format(
                    db_filename, str(e), str(ee)
                )
            )

    logger.info("Structures found: {}".format(len(df)))
    df[UNIQUE_PATH] = [
        generate_path(ind, row, name_column=name_column) for ind, row in df.iterrows()
    ]
    n_unique = len(df[UNIQUE_PATH].unique())
    if n_unique < len(df):
        raise ValueError(
            "Non-unique paths were found in dataset: dataset size = {}, number of unique paths = {}".format(
                len(df), n_unique
            )
        )

    # reset error statuses
    if reset_errors:
        logger.info("Try to reset 'error' status")
        state_dict = load_state_dict()
        for k, cur_state in state_dict.items():
            if cur_state.get("status") == "error":
                cur_state["status"] = "reset"
        save_state_dict(state_dict)
    #### MAIN LOOP ####
    if dry_run:
        logger.info("==== DRY RUN ====")
    # repeat this periodically
    logger.info("Entering into main loop, sleep interval = {} s".format(SLEEP_INTERVAL))
    while True:
        try:
            current_job_ids = get_current_running_job_ids(queue_options)
        except Exception as e:
            if dry_run:
                logger.info(
                    "Error ({}) while trying to get current job_ids, ignore because in dry-run mode".format(
                        str(e)
                    )
                )
                current_job_ids = {}
            else:
                raise e

        state_dict = load_state_dict()
        for ind, row in df.iterrows():

            cur_atoms = row[ASE_ATOMS]
            cur_structure_path = row[UNIQUE_PATH]

            cur_dir = os.path.join(dft_dir, cur_structure_path)
            cur_state = state_dict.get(cur_dir, {})
            if cur_state and cur_state.get("status") != "reset":
                # work with a status
                if cur_state.get("status") == "finished":
                    if dry_run:
                        logger.info("{}: finished".format(cur_structure_path))
                elif cur_state.get("status") == "submitted":
                    # check if running in the queue_name
                    job_id = cur_state["job_id"]
                    if job_id in current_job_ids:
                        if dry_run:
                            logger.info(
                                "{}: already submitted with job_id={}".format(
                                    cur_structure_path, job_id
                                )
                            )
                    else:
                        # check if calculation is finished
                        calc.directory = cur_dir
                        if calc.is_finished():
                            logger.info(
                                "{}: calculation is done, set status to finished".format(
                                    cur_structure_path
                                )
                            )
                            state_dict[cur_dir]["status"] = "finished"
                        else:
                            logger.info(
                                "{}: calculation is neither in the queue_name nor finished, set status to error".format(
                                    cur_structure_path
                                )
                            )
                            state_dict[cur_dir]["status"] = "error"
            else:  # new or reset
                calc.atoms = cur_atoms
                calc.directory = cur_dir
                if not calc.is_finished():
                    if dry_run:
                        if cur_state.get("status") != "reset":
                            logger.info(
                                "{}: neither finished, nor in the queue_name".format(
                                    cur_structure_path
                                )
                            )
                        else:
                            logger.info("{}: reset status".format(cur_structure_path))
                    else:
                        # if not finished and not previously stored
                        job_id = calc.submit_to_scheduler(queue_options)
                        if job_id is not None:
                            if cur_state.get("status") != "reset":
                                logger.info(
                                    "{}: submit to the queue_name with job id {}".format(
                                        cur_structure_path, job_id
                                    )
                                )
                            else:
                                logger.info(
                                    "{}: RESET, submit to the queue_name with job id {}".format(
                                        cur_structure_path, job_id
                                    )
                                )
                            state_dict[cur_dir] = {
                                "status": "submitted",
                                "job_id": job_id,
                            }
                        else:
                            if cur_state.get("status") != "reset":
                                logger.info(
                                    "{}: Error in job submission, set status to error".format(
                                        cur_structure_path
                                    )
                                )
                            else:
                                logger.info(
                                    "{}: Try to reset/resubmit: Error in job submission, set status to error".format(
                                        cur_structure_path
                                    )
                                )
                            state_dict[cur_dir] = {"status": "error"}
                else:
                    logger.info(
                        "{}: calculation is finished".format(cur_structure_path)
                    )
                    state_dict[cur_dir] = {"status": "finished"}

        # save state_dict to local file
        save_state_dict(state_dict)
        if dry_run:
            is_any_submitted_calculations = False
            is_all_finished = False
            logger.info("Stop main loop, in dry-run mode")
            break

        # check if all calculations are finished
        is_all_finished, is_any_submitted_calculations = aggregate_calculations_status(
            state_dict
        )

        if not is_any_submitted_calculations:
            logger.info("No any submitted/running calculations, stop main loop")
            break

        if is_all_finished:
            logger.info("All calculations are finished, stop main loop")
            break

        if daemon_mode:
            time.sleep(SLEEP_INTERVAL)
        else:
            break

    # after main loop

    # TODO: works only for VASP, generalize
    if is_all_finished and not dry_run:
        # collect raw data
        collect_raw_data(os.path.join(dft_dir, collected_file), dft_dir)
        # write finished flag
        with open(os.path.join(dft_dir, "finished"), "w") as f:
            print("True", file=f)
    elif not is_any_submitted_calculations and not dry_run:
        # write error flag
        with open(os.path.join(dft_dir, "error"), "w") as f:
            print("True", file=f)


if __name__ == "__main__":
    with lock_file(lock_filename="ams_dft_manager.lock"):
        main(sys.argv[1:])
