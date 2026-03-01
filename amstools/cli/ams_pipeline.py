#!/usr/bin/env python

# ams_pipeline pipeline.json
import argparse
import getpass  # for getpass.getuser()
import os
import sys
import logging
import socket

from amstools import Pipeline
from amstools.pipeline.pipelinestep import CREATED

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger()

import amstools


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="ams_pipeline",
        description="AMS pipeline execution utility.\n"
        + "\tversion: {}".format(amstools.__version__),
    )
    parser.add_argument(
        "json_filename",
        help="pipeline json filename, default: pipe.json",
        nargs="?",
        type=str,
        default="pipe.json",
    )

    parser.add_argument(
        "-v",
        "--version",
        help="Show version info",
        dest="version",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-l",
        "--log",
        help="log filename, default: log.txt",
        type=str,
        default="log.txt",
    )

    parser.add_argument(
        "-r",
        "--reset-status",
        help="reset steps' statuses and rerun (recollect) the pipeline",
        dest="reset_status",
        default=False,
        action="store_true",
    )

    args_parse = parser.parse_args(args)

    json_filename = args_parse.json_filename

    reset_status = args_parse.reset_status

    if "log" in args_parse:
        log_file_name = args_parse.log
        logger.info("Redirecting log into file {}".format(log_file_name))
        fh = logging.FileHandler(log_file_name, "a")
        fh.setLevel(logging.DEBUG)
        formatter = logging.Formatter(LOG_FMT)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if args_parse.version:
        print("ams_pipeline version: {}".format(amstools.__version__))
        sys.exit(0)
    logger.info("*" * 50)
    logger.info("Start ams_pipeline")
    logger.info("Hostname: {}".format(socket.gethostname()))
    logger.info("Username: {}".format(getpass.getuser()))
    logger.info("Loading pipeline from {}".format(json_filename))
    pipe = Pipeline.read_json(json_filename)
    if reset_status:
        logger.info("Reset steps statuses")
        for step_name, step in pipe.steps.items():
            step.finished = False
            step.status = CREATED

    logger.info("Loaded pipeline status: {}".format(pipe.status))
    logger.info("        steps statuses:")
    for step_name, step in pipe.steps.items():
        logger.info("{0: >15} : {1}".format(step_name, step.status))

    cwd = os.getcwd()
    logger.info("Setting working directory to {}".format(cwd))
    pipe.path = cwd
    pipe.autosave = True
    logger.info("Running pipeline")
    pipe.run()
    logger.info("Storing pipeline to {}".format(json_filename))
    pipe.to_json(json_filename)


if __name__ == "__main__":
    main(sys.argv[1:])
