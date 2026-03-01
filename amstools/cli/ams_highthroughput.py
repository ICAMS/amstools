#!/usr/bin/env python

import logging
import os

import argparse
import sys
import time
from datetime import datetime

import pkg_resources
import amstools
from shutil import copyfile
from amstools import Pipeline
from amstools.utils import load_yaml, CALC_LOCK
from amstools.highthroughput.generate_ht_pipeline_setup import (
    generate_ht_pipelines_setup,
    STRUCTURES_PATH,
    PIPELINE_STEPS_NAMES,
)
from amstools.highthroughput.utils import (
    create_calc,
    initialize_pipe,
    identify_environment,
)
from amstools.pipeline.scheduler import get_current_running_job_ids, cancel_queue_jobs
from amstools.utils import (
    load_amstools_queues_setup,
    standardize_pipe_name,
    SQLiteStateDict,
)
from amstools.pipeline import scheduler as pq
from amstools.pipeline.pipeline import PIPELINE_LOCK
from amstools.pipeline.pipelinestep import FINISHED, PARTIALLY_FINISHED, ERROR, PAUSED

REBUILD_STATE_DICT_DB = "rebuild_state_dict.db"

hostname, username, pid = identify_environment()
LOG_FMT = "%(asctime)s ({}[{}]): %(message)s".format(hostname, pid)
logging.basicConfig(level=logging.INFO, format=LOG_FMT, datefmt="%Y/%m/%d %H:%M:%S")
logger = logging.getLogger()

default_state_dict_fname = "state_dict.db"


# DONE: 1. Store  timestamp in state_dict
# TODO 2. Store in state_dict composition of initial structure
# TODO 3. webinterface: show job-id in table
# DONE 4. prettify JSON of state dict and VALUE
# DONE: 5. navigate through raw files and folders
# TODO 6. pipeline page: calculator config (from pipeline.json)
# DONE 7. Pipeline steps table: split out calculator folder
# TODO 9. bug for test_normal with more n_elements


def watchqueue(options):
    """Watch queue"""
    sleep_time = options.get("sleep_time", 1000)
    queuename = options["queuename"]
    max_jobs = options.get("max_jobs_in_queue")
    # do not wait if no `max_jobs_in_queue` option is provided
    if max_jobs is None:
        return

    if options["scheduler"] == "slurm":
        cmd = (
            "squeue --user="
            + username
            + " -O Partition | grep "
            + queuename
            + " | wc -l"
        )
    elif options["scheduler"] == "sge":
        cmd = "qstat -s p -u " + username + " | wc -l"
    else:
        raise ValueError("Unknown scheduler")

    while True:
        out = os.popen(cmd).read()
        num_jobs = int(out)
        if num_jobs < max_jobs:
            break
        logger.info(
            f"{num_jobs} jobs in queue (max {max_jobs}) ...waiting {sleep_time} s... "
        )
        time.sleep(sleep_time)


def generate_templates(calc_name):
    if calc_name.lower() == "vasp":
        template_calculator_yaml_filename = pkg_resources.resource_filename(
            "amstools.highthroughput.data", "calculator_template_vasp.yaml"
        )
    elif calc_name.lower() == "aims":
        template_calculator_yaml_filename = pkg_resources.resource_filename(
            "amstools.highthroughput.data", "calculator_template_aims.yaml"
        )
    else:
        raise RuntimeError("Invalid calculator name: {}".format(calc_name))
    copyfile(template_calculator_yaml_filename, "calculator.yaml")

    template_ht_yaml_filename = pkg_resources.resource_filename(
        "amstools.highthroughput.data", "ht_template.yaml"
    )
    copyfile(template_ht_yaml_filename, "ht.yaml")


def run_pipeline_queue_mode(
    name, structure, pipe, calc, queue_name, queue_options, state_dict, cur_work_dir
):
    """
    Run pipeline in queue mode, submit to the queue.
    """
    watchqueue(queue_options)
    logger.info("Submitting pipeline to the queue: {}".format(queue_name))
    initialize_pipe(pipe, cur_work_dir, calc, structure)
    os.makedirs(cur_work_dir, exist_ok=True)
    cur_state = state_dict[name]
    pipeline_json_filename = os.path.join(cur_work_dir, pipe.json_filename)
    # write
    pipe.to_json(pipeline_json_filename)
    job_id = pipe.submit_to_scheduler(queue_options)
    # when submit to the queue - keep track of the job-id in the queue, check its status
    if job_id is not None:
        cur_state.update(
            {
                "status": "submitted",
                "job_id": job_id,
                "queue_name": queue_name,
                "hostname": hostname,
            }
        )
        logger.info("Submitted {} to the queue with job id {}".format(name, job_id))
    else:
        cur_state["status"] = ERROR
        logger.info("Error in job submission, set status to error for {}".format(name))
    state_dict.save_row(name, cur_state)


def run_pipeline_locally(
    name, structure, pipe, calc, state_dict, cur_work_dir, worker_mode=False
):
    """
    Run pipeline locally. If worker_mode is True, then write pipeline.lock file
    :param name: name of the pipeline
    :param structure: structure
    :param pipe: pipeline object
    :param calc: calculator object
    :param state_dict: state dict object
    :param cur_work_dir: current working directory
    :param worker_mode: if True, then write pipeline.lock file
    """
    cur_state = state_dict[name]
    pipeline_lock_filename = os.path.join(cur_work_dir, PIPELINE_LOCK)
    pipeline_json_filename = os.path.join(cur_work_dir, pipe.json_filename)
    initialize_pipe(pipe, cur_work_dir, calc, structure)
    os.makedirs(cur_work_dir, exist_ok=True)
    if worker_mode:
        #  (worker mode): submit the whole ams_highthroughput into the queue, use pipeline.lock files
        #  acquire pipeline.lock file and run locally, otherwise skip
        # os.makedirs(cur_work_dir, exist_ok=True)
        with open(pipeline_lock_filename, "w") as f:
            print("{}\n{}".format(datetime.now(), hostname), file=f)  # create and leave
    # normal run locally
    logger.info("Running pipeline for {}".format(name))
    # write
    pipe.to_json(pipeline_json_filename)
    update_statuses_hash_in_cur_state(cur_state, pipe)
    cur_state.update({"status": "running", "hostname": hostname, "pid": pid})
    state_dict.save_row(name, cur_state)
    try:
        pipe.run()
        update_statuses_hash_in_cur_state(cur_state, pipe)
        state_dict.save_row(name, cur_state)
    except Exception as e:
        update_statuses_hash_in_cur_state(cur_state, pipe)
        cur_state["status"] = ERROR
        cur_state["error"] = str(e)
        state_dict.save_row(name, cur_state)
        # TODO: optionally raise error
        # raise e


def write_submission_script_worker_mode(
    queue_options, working_dir, calc_fname, ht_fname
):
    """
    Write submission script for ams_highthroughput in worker mode
    :param queue_options: queue options
    :param working_dir: working directory
    :param calc_fname: calculator filename
    :param ht_fname: highthroughput filename
    :return: scheduler object
    """
    ht_fname = os.path.abspath(ht_fname)
    scheduler_type = queue_options["scheduler"]
    if scheduler_type == "slurm":
        scheduler = pq.SLURM(
            queue_options, cores=queue_options["cores"], directory=working_dir
        )
    elif scheduler_type == "sge":
        scheduler = pq.SGE(
            queue_options, cores=queue_options["cores"], directory=working_dir
        )
    else:
        raise ValueError("Unknown scheduler: {}".format(scheduler_type))

    scheduler.maincommand = "ams_highthroughput {} -c {} --worker-mode".format(
        ht_fname, calc_fname
    )
    ams_worker_job_fname = "ams_worker_job.sh"
    submission_script = os.path.join(working_dir, ams_worker_job_fname)
    scheduler.write_script(submission_script)
    logger.info("Prepare submission script: {}".format(ams_worker_job_fname))
    return scheduler


def cancel_all_queue_jobs(queue_options):
    """Cancel all jobs in the queue"""
    job_ids_dict = get_current_running_job_ids(queue_options)
    job_ids_list = [j for j, v in job_ids_dict.items()]
    logging.info("Canceling {} jobs".format(len(job_ids_list)))
    cancel_queue_jobs(job_ids_list, queue_options)
    # TODO: reset calc.lock
    # TODO: reset pipeline.lock
    # TODO: update state dict
    # job_ids_set = set(job_ids_list)
    # state_dict = create_state_dict(state_dict_fname)


def configure_logger(args_parse):
    """
    Configure logger
    """
    log_file_name = args_parse.log
    split_worker_output_files = args_parse.split_worker_output_files
    worker_mode = args_parse.worker_mode
    if worker_mode and split_worker_output_files:
        log_file_name = log_file_name.replace(".", "__" + hostname + ".")
    logger.info("Redirecting log into file {}".format(log_file_name))
    fh = logging.FileHandler(log_file_name, "a")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter(LOG_FMT)
    fh.setFormatter(formatter)
    logger.addHandler(fh)


def create_state_dict(state_dict_fname):
    """
    Create state dict object from the state_dict_fname.
    Only SQLiteStateDict is supported for now.
    """
    # if state_dict_fname.endswith(".json"):
    #     state_dict = JSONStateDict(state_dict_fname)
    if state_dict_fname.endswith(".db"):
        state_dict = SQLiteStateDict(state_dict_fname)
    else:
        raise ValueError(
            "Unknown state dict file extension: {}".format(state_dict_fname)
        )
    return state_dict


def do_rebuild_state_dict(
    working_dir, state_dict_fname, include_dirs=None, exclude_dirs=None
):
    """
    Rebuild state dict from the working_dir.

    :param working_dir: working directory
    :param state_dict_fname: state dict file name
    :param include_dirs: list of strings, include only pipelines with these strings in the path. If None, include all.
    :param exclude_dirs: list of strings, exclude pipelines with these strings in the path. If None, exclude none
    :return: None
    """

    def is_in_include_dirs(pipe_name):
        if include_dirs is None:
            return True
        else:
            for include_dir in include_dirs:
                if include_dir in pipe_name:
                    return True
            return False

    def is_in_exclude_dirs(pipe_name):
        if exclude_dirs is None:
            return False
        else:
            for exclude_dir in exclude_dirs:
                if exclude_dir in pipe_name:
                    return True
            return False

    def print_stat():
        logger.info(
            "{} folder(s) visited / {} folder(s) filtered / {} pipeline(s) found".format(
                visited_folders_counter, selected_folders_counter, pipeline_file_counter
            )
        )

    PIPELINE_JSON = "pipeline.json"
    rewrite_db = False
    if os.path.isfile(state_dict_fname):
        logger.warning(f"WARNING!!! {state_dict_fname} file already exists!")
        answer = input(
            f" {state_dict_fname} file already exists. Do you want to append new entries to it (Y) or overwrite this file (N) [Y=append/N=overwrite]?:"
        ).lower()
        if answer == "y" or answer == "":
            rewrite_db = False
        elif answer == "n":
            answer2 = input(
                f"{state_dict_fname} will be overwritten, are you sure? [Y/N, default=N]?:"
            ).lower()
            if answer2 == "Y":
                rewrite_db = True
        if rewrite_db:
            logger.warning(f"{state_dict_fname} file will be overwritten!")

    # use TMP filename, delete it if exists
    if rewrite_db:
        # overwrite DB file
        tmp_state_dict_fname = state_dict_fname.replace(".db", ".TMP.db")
        if os.path.isfile(tmp_state_dict_fname):
            os.remove(tmp_state_dict_fname)
        state_dict = create_state_dict(tmp_state_dict_fname)
    else:
        # append/update mode
        state_dict = create_state_dict(state_dict_fname)

    # walk recursively through the working_dir and find all pipeline.json files
    # for each pipeline.json load it and update state_dict
    logger.info("Rebuilding state dict from {}".format(working_dir))
    if include_dirs is not None:
        logger.info("Include dirs: {}".format(include_dirs))
    if exclude_dirs is not None:
        logger.info("Exclude dirs: {}".format(exclude_dirs))

    logger.info("Building list of pipeline JSON files...")

    root_folders = [working_dir]
    if include_dirs is not None:
        # if include_dir is real folder - add it to root path, otherwise - as part of path
        new_root_folders = []
        for include_dir in include_dirs:
            ext_include_dir = os.path.join(working_dir, include_dir)
            if os.path.isdir(ext_include_dir):
                new_root_folders.append(ext_include_dir)
                logger.info(f"{ext_include_dir} ({include_dir}) folder will be scanned")
        if len(new_root_folders) > 0:
            logger.info(
                "Only following root folders will be scanned: {}".format(
                    new_root_folders
                )
            )
            root_folders = new_root_folders

    # build list of files in separate process
    pipeline_files_dict = {}
    pipeline_file_counter = 0
    visited_folders_counter = 0
    selected_folders_counter = 0

    for root_folder in root_folders:
        for i, (root, dirs, files) in enumerate(os.walk(root_folder)):
            if PIPELINE_JSON in files:
                std_pipe_name = standardize_pipe_name(root.split(working_dir)[-1])
                visited_folders_counter += 1
                if visited_folders_counter % 500 == 0:
                    print_stat()
                if not is_in_include_dirs(std_pipe_name) or is_in_exclude_dirs(
                    std_pipe_name
                ):
                    continue
                selected_folders_counter += 1

                full_path = os.path.join(root, PIPELINE_JSON)
                pipeline_files_dict[std_pipe_name] = full_path
                pipeline_file_counter = len(pipeline_files_dict)
                if pipeline_file_counter % 100 == 0:
                    print_stat()

    num_of_pipelines = len(pipeline_files_dict)
    logger.info("Found {} pipeline JSON files".format(num_of_pipelines))
    logger.info("Rebuilding state dict...")
    for i, (std_pipe_name, full_path) in enumerate(pipeline_files_dict.items()):
        try:
            if std_pipe_name not in state_dict:
                existing_pipe = Pipeline.read_json(full_path)
                cur_state = state_dict[std_pipe_name]
                update_statuses_hash_in_cur_state(cur_state, existing_pipe)
                state_dict.save_row(std_pipe_name, cur_state)
            logger.info(" {}/{}: {}".format(i + 1, num_of_pipelines, std_pipe_name))
        except Exception as e:
            logger.error(
                "Error while rebuilding state dict for {}: {}".format(std_pipe_name, e)
            )
    if rewrite_db:
        os.rename(tmp_state_dict_fname, state_dict_fname)


def update_statuses_hash_in_cur_state(cur_state, pipe):
    """
    Update statuses and hash in cur_state from pipe
    :param cur_state: current state dict
    :param pipe: pipeline
    :return: None
    """
    cur_state["status"] = pipe.status
    if "named_steps_statuses" not in cur_state:
        cur_state["named_steps_statuses"] = pipe.named_steps_statuses
    else:
        cur_state["named_steps_statuses"].update(pipe.named_steps_statuses)
    cur_state["hash"] = pipe.hash()


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    steps_description = "Possible pipeline steps names: " + ",".join(
        PIPELINE_STEPS_NAMES
    )
    parser = argparse.ArgumentParser(
        prog="ams_highthroughput",
        description="AMS high-throughput (HT) utility.\n"
        + "\tversion: {}.\n".format(amstools.__version__)
        + steps_description,
    )
    parser.add_argument(
        "ht_yaml_filename",
        help="HT YAML filename.",
        type=str,
        nargs="?",
        default="ht.yaml",
    )

    parser.add_argument(
        "-c",
        "--calculator",
        help="YAML file with calculator setup",
        type=str,
        dest="calculator_setup_fname",
    )

    parser.add_argument(
        "-wd",
        "--working-dir",
        help="top directory where keep and execute HT calculations",
        type=str,
        default=".",
        dest="working_dir",
    )

    parser.add_argument(
        "-q",
        "--queue-name",
        help="submit calculations to the queue. Options: <QUEUE_NAME> (~/.amstools::queues[QUEUE_NAME])."
        " Default is first queue from ~/.amstools::queues (if exists)",
        type=str,
        default=None,
        const="DEFAULT",
        nargs="?",
        dest="queue_name",
    )
    parser.add_argument(
        "-i",
        "--write-input-only",
        help="Write input files only. Default is False.",
        dest="write_input_only",
        action="store_true",
        default=False,
    )
    ############# GENERATE #############
    parser.add_argument(
        "-p",
        "--permutation-type",
        help="Global permutation type: atoms, mixed, wyckoff, cfg, constrained. Default: cfg",
        type=str,
        default="cfg",
        dest="permutation_type",
    )

    parser.add_argument(
        "--legacy-step-naming",
        help="Use legacy (flat) step naming: step names are their plain names with no "
             "structure-modifier prefix (e.g., 'murnaghan' instead of 'rlx_murnaghan'). "
             "Default is False.",
        dest="legacy_step_naming",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-r",
        "--repository-path",
        help="structure repository path. Default is env variable {}={}".format(
            STRUCTURES_PATH, os.environ.get(STRUCTURES_PATH, "<NONE>")
        ),
        type=str,
        dest="repository_path",
    )

    ############ RUN and RERUN ############
    parser.add_argument(
        "-dr",
        "--dry-run",
        help="Dry run: generate combinations but do not run calculations. Default is False.",
        dest="dry_run",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-u",
        "--update-pipeline-statuses",
        help="(Force) Update all pipeline statuses. Default is False.",
        dest="update_pipeline_statuses",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-rep",
        "--rerun-error-pipelines",
        help="Rerun pipelines in ERROR state. Default is False.",
        action="store_true",
        default=False,
        dest="rerun_error_pipelines",
    )

    parser.add_argument(
        "-rpfp",
        "--rerun-partially-finished-pipelines",
        help="Rerun pipelines in ERROR state. Default is False.",
        action="store_true",
        default=False,
        dest="rerun_partially_finished_pipelines",
    )

    parser.add_argument(
        "--cancel-all-jobs",
        help="(DANGER!) will cancel all jobs in the given queue. Default is False.",
        dest="cancel_all_jobs",
        action="store_true",
        default=False,
    )

    ############ WORKER MODE ############
    parser.add_argument(
        "--worker-mode",
        help="run in worker mode. "
        "Synchronization is done via pipeline.lock file for each pipeline. Default is False.",
        action="store_true",
        dest="worker_mode",
        default=False,
    )

    parser.add_argument(
        "-w",
        "--workers",
        help="Submit several workers mode. Use -q option to submit them to the queue. Default is 0.",
        type=int,
        default=0,
        dest="workers",
    )

    parser.add_argument(
        "--split-worker-output-files",
        help="Option shows if different workers should write log and state dict "
        "in files with unique names of worker host. Default is False.",
        dest="split_worker_output_files",
        default=False,
        action="store_true",
    )

    ############ DAEMON MODE ############
    parser.add_argument(
        "-d",
        "--daemon",
        help="Running infinitely until all pipelines are finished or failed. Default is False.",
        action="store_true",
        dest="daemon_mode",
        default=False,
    )

    parser.add_argument(
        "--time",
        help="Sleeping time in seconds for daemon mode. Default is 10",
        dest="sleeping_time",
        default=10,
        type=int,
    )

    ############ PIPELINE AND CALC LOCKS  ############
    parser.add_argument(
        "--reset-pipeline-lock",
        dest="reset_pipeline_lock",
        action="store_true",
        default=False,
        help="Force to remove {} file in 'worker' mode. Default is False".format(
            PIPELINE_LOCK
        ),
    )

    parser.add_argument(
        "--reset-calculator-lock",
        dest="reset_calculator_lock",
        action="store_true",
        default=False,
        help="Force to remove calculator lock files ({}). Default is False".format(
            CALC_LOCK
        ),
    )

    ############ STATE DICT ############
    parser.add_argument(
        "--state-dict-file",
        help="file with calculations state: .db for SQLite3 (default: {})".format(
            default_state_dict_fname
        ),
        type=str,
        dest="state_dict_fname",
        default=default_state_dict_fname,
    )

    parser.add_argument(
        "--rebuild-state-dict",
        help="Rebuild state dict from scratch. If provided, then default is '{}'".format(
            REBUILD_STATE_DICT_DB
        ),
        nargs="?",
        dest="rebuild_state_dict",
        default="SKIP",
    )

    parser.add_argument(
        "--rebuild-include",
        help="List of folders (space separated) to include when rebuilding state dict. "
        + "If folder ends with '/', then only this subfolder will be scanned",
        nargs="+",
    )
    parser.add_argument(
        "--rebuild-exclude",
        help="List of folders to exclude when rebuilding state dict",
        nargs="+",
    )

    ############ LOGGING ############
    parser.add_argument(
        "-l",
        "--log",
        help="log filename, Default is log.txt",
        type=str,
        default="log.txt",
    )

    ############ OTHER ############
    parser.add_argument(
        "-t",
        "--template",
        help="Generate template for ht.yaml and calculator.yaml,"
        + " provide calculator type (vasp or aims)",
        type=str,
        default=None,
        const="vasp",
        nargs="?",
    )

    parser.add_argument(
        "--verbose",
        help="Verbose detailed output. Default is None",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-v",
        "--version",
        help="Show version info",
        dest="version",
        action="store_true",
        default=False,
    )

    args_parse = parser.parse_args(args)

    daemon_mode = args_parse.daemon_mode
    state_dict_fname = args_parse.state_dict_fname
    worker_mode = args_parse.worker_mode
    reset_pipeline_lock = args_parse.reset_pipeline_lock
    permutation_type = args_parse.permutation_type
    legacy_step_naming = args_parse.legacy_step_naming
    structure_repo_path = args_parse.repository_path
    rerun_error_pipelines = args_parse.rerun_error_pipelines
    calculator_setup_fname = args_parse.calculator_setup_fname
    split_worker_output_files = args_parse.split_worker_output_files
    working_dir = os.path.abspath(args_parse.working_dir)
    update_pipeline_statuses = args_parse.update_pipeline_statuses
    rerun_partially_finished_pipelines = args_parse.rerun_partially_finished_pipelines
    ##################################################

    if args_parse.version:
        print("ams_highthroughput version: {}".format(amstools.__version__))
        sys.exit(0)

    if "log" in args_parse:
        configure_logger(args_parse)

    if args_parse.template:
        print(
            "A template for ht.yaml and calculator.yaml (calculator: {}) being generated...".format(
                args_parse.template
            ),
            end="",
        )
        generate_templates(args_parse.template)
        print("done")
        sys.exit(0)

    if structure_repo_path is None:
        # structure path not provided, try to extract from ENV
        if STRUCTURES_PATH in os.environ:
            structure_repo_path = os.environ[STRUCTURES_PATH]
        else:
            logger.error(
                "Could not find structure repository path, use either -r option or {} env variable".format(
                    STRUCTURES_PATH
                )
            )
            sys.exit(1)

    logger.info("=" * 50)
    logger.info("Running AMS-HighThroughput:")
    logger.info("amstools version: {}".format(amstools.__version__))
    logger.info("Hostname: {}".format(hostname))
    logger.info("PID: {}".format(pid))
    logger.info("Username: {}".format(username))
    logger.info("\tdefault permutation type: {}".format(permutation_type))
    logger.info("\tstructure repo path: {}".format(structure_repo_path))
    logger.info("\tworking dir: {}".format(working_dir))

    if args_parse.rebuild_state_dict != "SKIP":
        if args_parse.rebuild_state_dict is None:
            rebuild_state_dict_fname = REBUILD_STATE_DICT_DB
        else:
            rebuild_state_dict_fname = args_parse.rebuild_state_dict
        logger.info(
            "State dict will be rebuild and stored into {}".format(
                rebuild_state_dict_fname
            )
        )
        do_rebuild_state_dict(
            working_dir,
            rebuild_state_dict_fname,
            include_dirs=args_parse.rebuild_include,
            exclude_dirs=args_parse.rebuild_exclude,
        )
        logger.info(
            "Rebuild state dict is done and stored into {}".format(
                rebuild_state_dict_fname
            )
        )
        sys.exit(0)

    queue_options = {}
    queue_name = args_parse.queue_name
    queue_mode = queue_name is not None
    if queue_mode or args_parse.workers > 0:
        # load queue settings
        queues = load_amstools_queues_setup()
        if len(queues) == 0:
            raise RuntimeError("No $HOME/.amstools::queues configured")
        if queue_name == "DEFAULT":
            # take first config from .amstools::queues
            queue_name = next(iter(queues))
            logger.info("Default queue settings: {}".format(queue_name))
        queue_options = queues[queue_name]

    if args_parse.cancel_all_jobs:
        if not queue_options:
            logger.error("Please provide queue name")
            sys.exit(1)
        response = input(
            "CAUTION! This will cancel all jobs in the queue `{}`. Do you want to proceed?[Y/N]".format(
                queue_name
            )
        )
        if response.upper() == "Y":
            cancel_all_queue_jobs(queue_options)
        sys.exit(0)

    # create calculator from setup
    if calculator_setup_fname is not None:
        logger.info("Loading calculator setup from {}".format(calculator_setup_fname))
        calc, enforce_pbc = create_calc(calculator_setup_fname, if_enforce_pbc=True)
        calc_setup = load_yaml(calculator_setup_fname)
    else:
        raise ValueError(
            "Calculator setup file is not provided, please add -c or --calculator option"
        )

    scheduler_worker_mode = None
    if args_parse.workers > 0:
        scheduler_worker_mode = write_submission_script_worker_mode(
            queue_options,
            working_dir,
            calculator_setup_fname,
            args_parse.ht_yaml_filename,
        )

    if worker_mode and split_worker_output_files:
        state_dict_fname = state_dict_fname.replace(".", "__" + hostname + ".")
        logger.info(
            "Running in worker mode, update state dict file name to {}".format(
                state_dict_fname
            )
        )

    logger.info("Calculator:")
    logger.info("\t{}".format(calc_setup))

    if "name" in calc_setup:
        calc_name = calc_setup["name"]
        logger.info(
            "Calculator name: {}, working dir is {}".format(calc_name, working_dir)
        )
    else:
        raise ValueError(
            "Calculator name is not provided, please add `name: CALCULATOR_NAME` into {} file".format(
                calculator_setup_fname
            )
        )

    ht_yaml_filename = os.path.abspath(args_parse.ht_yaml_filename)
    if not os.path.isfile(ht_yaml_filename):
        logger.error("Could not find HT YAML-file {}".format(ht_yaml_filename))
        sys.exit(1)
    logger.info("Loading HT setup from {}".format(ht_yaml_filename))

    ht_setup = generate_ht_pipelines_setup(
        ht_yaml_filename,
        permutation_type=permutation_type,
        structures_repository_path=structure_repo_path,
        name_prefix=calc_name + "/",
        enforce_pbc=enforce_pbc,
        verbose=args_parse.verbose,
        legacy_step_naming=legacy_step_naming,
    )
    # count number of total steps
    total_structure = len(ht_setup["name"])
    total_steps = sum(len(pipe.steps) for pipe in ht_setup["pipeline"])
    logger.info(
        "Total number of pipelines (steps): {} ({})".format(
            total_structure, total_steps
        )
    )
    if args_parse.verbose:
        for name, structure, pipe in zip(
            ht_setup["name"], ht_setup["structure"], ht_setup["pipeline"]
        ):
            logger.info(
                "\t{}: {} atom(s), {} step(s)".format(
                    name, len(structure), len(pipe.steps)
                )
            )
    else:
        logger.info("Use --verbose flag to show all pipelines")

    logger.info("Loading state dictionary from {}".format(state_dict_fname))
    state_dict = create_state_dict(state_dict_fname)
    logger.info("{} total state(s) found".format(len(state_dict)))

    ht_names = set(ht_setup["name"])
    state_counter = state_dict.analyze_stats(ht_names)
    logger.info(
        "States statistics for {} current jobs: {}".format(len(ht_names), state_counter)
    )

    # TODO: add option for analysing existing pipeline.lock files
    # TODO: add possibility ot remove pipeline.lock older than ...

    if reset_pipeline_lock:
        do_reset_pipeline_lock(
            ht_names,
            ht_setup,
            rerun_error_pipelines,
            state_dict,
            working_dir,
            dry_run=args_parse.dry_run,
        )
        sys.exit(0)

    if args_parse.reset_calculator_lock:
        do_reset_calculator_lock(
            ht_names, state_dict, working_dir, dry_run=args_parse.dry_run
        )
        sys.exit(0)

    if rerun_error_pipelines:
        logger.info("Rerun error pipelines: pipelines with error state will be re-run")
    if rerun_partially_finished_pipelines:
        logger.info(
            "Rerun partially finished pipelines: pipelines with partially finished state will be re-run"
        )

    if args_parse.workers > 0:
        logger.info("Submitting {} workers to the queue".format(args_parse.workers))
        if args_parse.dry_run:
            logger.info("DRY-RUN, no submission of workers")
        else:
            for worker in range(args_parse.workers):
                res = scheduler_worker_mode.submit()
                lines = res.stdout.readlines()
                job_id = scheduler_worker_mode.get_job_id(lines)
                logger.info(
                    "Submitting worker number {} to the queue with job id {}".format(
                        worker, job_id
                    )
                )
        sys.exit(0)

    if args_parse.dry_run:
        logger.info("DRY-RUN, finishing")
        sys.exit(0)

    if daemon_mode:
        logger.info("Running in infinite-loop (daemon) mode")
    current_job_ids = []
    locked_pipelines = []
    ##########################################################################
    #                  MAIN LOOP                                             #
    ##########################################################################
    logger.info("Entering main loop")
    num_of_pipelines = len(ht_setup["pipeline"])
    while True:
        number_of_locked_pipelines = 0
        for ind, (name, structure, pipe) in enumerate(
            zip(ht_setup["name"], ht_setup["structure"], ht_setup["pipeline"])
        ):
            if not daemon_mode:
                if (ind + 1) % 100 == 0:
                    logging.info("Pipe {}/{}".format(ind + 1, num_of_pipelines))
            # if queue_mode or worker_mode:
            # TODO: get job_ids for worker_mode also
            if queue_mode:
                current_job_ids = get_current_running_job_ids(
                    queue_options
                )  # check the status of each of them
            if args_parse.write_input_only:
                pipe.write_input_only = True
            cur_state = state_dict[name]
            job_id = cur_state.get("job_id")
            cur_status = cur_state.get("status")
            # skip error pipelines if not rerun_error_pipelines or update_pipeline_statuses
            if cur_status in [ERROR] and not (
                update_pipeline_statuses or rerun_error_pipelines
            ):
                continue
            # skip partially finished pipelines if not rerun_partially_finished_pipelines or update_pipeline_statuses
            if cur_status in [PARTIALLY_FINISHED] and not (
                update_pipeline_statuses or rerun_partially_finished_pipelines
            ):
                continue
            # skip finished pipelines if not update_pipeline_statuses
            if cur_status in [FINISHED] and not update_pipeline_statuses:
                continue

            cur_work_dir = os.path.abspath(os.path.join(working_dir, name))

            if worker_mode:
                #  (worker mode): submit the whole ams_highthroughput into the queue, use pipeline.lock files
                #  acquire pipeline.lock file and run locally, otherwise skip
                pipeline_lock_filename = os.path.join(cur_work_dir, PIPELINE_LOCK)
                if os.path.isfile(pipeline_lock_filename):
                    number_of_locked_pipelines += 1
                    locked_pipelines.append(name)
                    continue

            pipeline_json_filename = os.path.join(cur_work_dir, pipe.json_filename)
            # before submission to the queue we try check the status from pipeline.json
            if os.path.isfile(pipeline_json_filename):
                try:
                    existing_pipe = Pipeline.read_json(pipeline_json_filename)
                    # TODO more carefully comparison (by hash: include steps names + job options(maybe?))
                    if (
                        pipe.hash() != existing_pipe.hash()
                    ):  # if pipeline setup is changed
                        if update_pipeline_statuses:
                            cur_state["named_steps_statuses"] = (
                                pipe.named_steps_statuses
                            )
                            cur_state["named_steps_statuses"].update(
                                existing_pipe.named_steps_statuses
                            )
                            cur_state["status"] = "outdated/" + existing_pipe.status
                            state_dict.save_row(name, cur_state)
                            logger.info(
                                "Pipeline {}: set status to outdated".format(name)
                            )
                        # then rerun
                        elif queue_mode:
                            run_pipeline_queue_mode(
                                name,
                                structure,
                                pipe,
                                calc,
                                queue_name,
                                queue_options,
                                state_dict,
                                cur_work_dir,
                            )
                        else:  # if normal or write-input-only mode
                            run_pipeline_locally(
                                name,
                                structure,
                                pipe,
                                calc,
                                state_dict,
                                cur_work_dir,
                                worker_mode,
                            )
                    elif existing_pipe.status in [
                        PAUSED
                    ]:  # if pipeline paused, then try to run it locally
                        run_pipeline_locally(
                            name,
                            structure,
                            pipe,
                            calc,
                            state_dict,
                            cur_work_dir,
                            worker_mode,
                        )
                    else:  # else if pipe is the same  then check statuses
                        # check if pipeline finished ?
                        update_statuses_hash_in_cur_state(cur_state, existing_pipe)
                        if existing_pipe.status in [FINISHED]:
                            logger.info(
                                "Pipeline {} is {}".format(name, existing_pipe.status)
                            )
                        elif (
                            existing_pipe.status in [PARTIALLY_FINISHED]
                            and not rerun_partially_finished_pipelines
                        ):
                            logger.info(
                                "Pipeline {} is {}".format(name, existing_pipe.status)
                            )
                        # elif (queue_mode or worker_mode) and job_id in current_job_ids:
                        # TODO: check the status also for worker_mode
                        elif (
                            queue_mode and job_id in current_job_ids
                        ):  # not finished, but is in a queue
                            queue_state = current_job_ids[job_id]
                            st = queue_state.get("state")
                            if st.lower() in ["r"]:  # statuses for SLURM or SGE
                                cur_state["status"] = "running"
                            elif st.lower() in [
                                "pd",
                                "qw",
                            ]:  # statuses for SLURM or SGE
                                cur_state["status"] = "pending"
                            else:
                                cur_state["status"] = "inqueue"
                            cur_state.update(queue_state)
                        else:  # not finished, not partially finished, not in queue
                            if (
                                rerun_error_pipelines
                                or rerun_partially_finished_pipelines
                            ):
                                logger.info("Force rerun {}".format(name))
                                if queue_mode:
                                    run_pipeline_queue_mode(
                                        name,
                                        structure,
                                        pipe,
                                        calc,
                                        queue_name,
                                        queue_options,
                                        state_dict,
                                        cur_work_dir,
                                    )
                                else:
                                    run_pipeline_locally(
                                        name,
                                        structure,
                                        pipe,
                                        calc,
                                        state_dict,
                                        cur_work_dir,
                                        worker_mode,
                                    )
                            else:
                                if not daemon_mode:
                                    logger.info(
                                        "{}: calculation is neither running"
                                        " nor finished, set status to error".format(
                                            name
                                        )
                                    )
                                update_statuses_hash_in_cur_state(
                                    cur_state, existing_pipe
                                )
                                # override status to error
                                cur_state["status"] = ERROR
                        state_dict.save_row(name, cur_state)
                except Exception as e:
                    state_dict.save_row(name, cur_state)
                    raise e
            else:  # new run TODO: AND NOT COLLECT STATUSES
                if update_pipeline_statuses:
                    cur_state["status"] = "notexisting"
                    state_dict.save_row(name, cur_state)
                elif queue_mode:
                    run_pipeline_queue_mode(
                        name,
                        structure,
                        pipe,
                        calc,
                        queue_name,
                        queue_options,
                        state_dict,
                        cur_work_dir,
                    )
                else:
                    run_pipeline_locally(
                        name,
                        structure,
                        pipe,
                        calc,
                        state_dict,
                        cur_work_dir,
                        worker_mode,
                    )

        if daemon_mode:
            # avoid re-run calculation on every loop, switch rerun_error_pipelines=False
            rerun_error_pipelines = False
            rerun_partially_finished_pipelines = False
        else:  # if not daemon_mode
            logging.info("Pipe {}/{}".format(num_of_pipelines, num_of_pipelines))

        # if states are finished or error - break loop
        has_running = state_dict.has_not_finished_or_error_states(ht_names)

        if not daemon_mode:
            break  # infinite while True loop
        elif not has_running:
            state_counter = state_dict.analyze_stats(ht_names)
            logger.info(
                "Stopping infinite-loop because no running jobs: {}".format(
                    state_counter
                )
            )
            break
        time.sleep(args_parse.sleeping_time)

    #  end while-True loop
    logger.info("Main loop is finished")
    if worker_mode:
        logger.info(
            "Number of locked/skipped pipelines: {}".format(number_of_locked_pipelines)
        )
        running_states = state_dict.get_running_states(ht_names) or locked_pipelines
        if len(running_states):
            logger.info(
                "Currently {} jobs has running state in the following folders:".format(
                    len(running_states)
                )
            )
            for name in running_states:
                logger.info(" - {}".format(name))
            logger.info(
                "Check/remove the files {} manually or use '--reset-pipeline-lock' option to remove all".format(
                    PIPELINE_LOCK
                )
            )

    state_counter = state_dict.analyze_stats(ht_names)
    logger.info(
        "States statistics for {} current jobs: {}".format(len(ht_names), state_counter)
    )
    logger.info("Finishing ams_highthroughput")


def do_reset_pipeline_lock(
    ht_names, ht_setup, rerun_error_pipelines, state_dict, working_dir, dry_run
):
    logger.info("Reset pipeline lock files and set status to `error`")
    if rerun_error_pipelines:
        running_states = ht_setup["name"]
    else:
        running_states = state_dict.get_running_states(ht_names)
        logger.info(
            "Look through all {}  running/paused pipelines".format(len(running_states))
        )
    logger.info("Removing {} files in:".format(PIPELINE_LOCK))
    if dry_run:
        logger.info("DRY-RUN, no removing of files")
    for name in running_states:
        cur_work_dir = os.path.abspath(os.path.join(working_dir, name))
        pipeline_lock_filename = os.path.join(cur_work_dir, PIPELINE_LOCK)
        if os.path.isfile(pipeline_lock_filename):
            logger.info(" - " + name)
            if not dry_run:
                os.remove(pipeline_lock_filename)
                cur_state = state_dict[name]
                if cur_state["status"] not in [FINISHED, PARTIALLY_FINISHED]:
                    cur_state["status"] = ERROR
                    cur_state["error"] = "{} file automatically removed".format(
                        PIPELINE_LOCK
                    )
                    state_dict.save_row(name, cur_state)
    logger.info("Resetting pipeline locks is done, finishing")


def do_reset_calculator_lock(ht_names, state_dict, working_dir, dry_run):
    logger.info("Reset calculator lock files")

    all_states = state_dict.get_all_states(ht_names)  # dict (name -> status)
    logger.info("Currently {} pipelines is found ".format(len(all_states)))
    logger.info("Removing {} files in:".format(CALC_LOCK))
    if dry_run:
        logger.info("DRY-RUN, no removing of files")
    count = 0
    for name, status in all_states.items():
        if status in [FINISHED, PARTIALLY_FINISHED, ERROR]:
            cur_work_dir = os.path.abspath(os.path.join(working_dir, name))
            # use os.walk to scan through all files
            for root, dirs, files in os.walk(cur_work_dir):
                if CALC_LOCK in files:
                    lock_filename = os.path.join(root, CALC_LOCK)
                    logger.info(" - " + lock_filename)
                    if not dry_run:
                        os.remove(lock_filename)
                    count += 1
    if not dry_run:
        logger.info(f"Resetting calc locks is done, {count} files processed")
    else:
        logger.info(f"Scanning calc locks is done, {count} files processed")


if __name__ == "__main__":
    main(sys.argv[1:])
