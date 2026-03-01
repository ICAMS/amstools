#!/usr/bin/env python

# ams_calculation calcuation.json -s -d working_dir
# ams_calculation calcuation.json -d working_dir

import argparse
import logging
import os
import socket
import sys

from ase.io import jsonio

import amstools
from amstools.calculators.dft.base import AMSDFTBaseCalculator
from amstools.pipeline import scheduler as pq

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger()


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="ams_calculation",
        description="AMS calculations execution utility.\n"
        + "\tversion: {}".format(amstools.__version__),
    )
    parser.add_argument(
        "json_filename",
        help="calculation json filename, default: calculation.json",
        nargs="?",
        type=str,
        default="calculation.json",
    )

    parser.add_argument(
        "-s",
        "--submit",
        help="submit to the queue",
        dest="submit",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-d",
        "--directory",
        help="directory for calculation (default - empty)",
        type=str,
        default="",
    )

    args_parse = parser.parse_args(args)

    json_filename = args_parse.json_filename
    submit = args_parse.submit  # flag, whether to submit the calculation to the queue
    working_dir = args_parse.directory
    if working_dir == "":
        working_dir = os.getcwd()
    logging.debug("Working dir: {}".format(working_dir))
    if submit:
        logging.debug("Job submission")
        # submit to the queue
        config_file = os.path.expanduser("~/.amstools")  # yaml config
        logging.debug("Try to load config from {}".format(config_file))
        import yaml

        with open(config_file, "r") as f:
            config = yaml.load(f)
        logging.debug("Config loaded")
        hostname = socket.gethostname()
        logging.info("Hostname: {}".format(hostname))
        queues_options = config.get("queues", {})
        if len(queues_options) == 0:
            raise RuntimeError(
                "Please specify the queues options in {}::queues::[hostname1,hostname2,...]".format(
                    config_file
                )
            )

        options = queues_options.get(hostname) or queues_options

        logging.info("Queue options: {}".format(options))

        working_dir = os.path.abspath(working_dir)

        if not os.path.isdir(working_dir):
            os.mkdir(working_dir)

        if options["scheduler"] == "slurm":
            scheduler = pq.SLURM(options, cores=options["cores"], directory=working_dir)
        elif options["scheduler"] == "sge":
            scheduler = pq.SGE(options, cores=options["cores"], directory=working_dir)
        else:
            raise ValueError("Unknown scheduler")

        scheduler.maincommand = "ams_calculation {} -d {}".format(
            json_filename, working_dir
        )
        submission_script = os.path.join(working_dir, "ams_calculation.sh")
        scheduler.write_script(submission_script)

        res = scheduler.submit()
        lines = res.stdout.readlines()
        logging.info("Submission result: {}".format(lines[0].decode()))
    else:
        # run
        dct_loaded = jsonio.read_json(json_filename)
        ams_dft_calculator = AMSDFTBaseCalculator.fromdict(dct_loaded)
        ams_dft_calculator.directory = working_dir
        # run calculation
        ams_dft_calculator.get_potential_energy()


if __name__ == "__main__":
    main(sys.argv[1:])
