#!/usr/bin/env python

# ams_pipeline pipeline.json
import argparse
import logging
import os
import sys

from amstools.utils import collect_raw_data,plot_df
from amstools.highthroughput.utils import discover_pipeline_jobs, collect_pipelines

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger()

import amstools


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser(
        prog="ams_collect",
        description="AMS pipeline and DFT data collection utility.\n"
        + "\tversion: {}".format(amstools.__version__),
    )

    parser.add_argument(
        "path", help="root path for collecting", nargs="?", type=str, default="."
    )

    parser.add_argument(
        "-r",
        "--raw-data",
        help="collect raw DFT data (data.json files)",
        dest="collect_raw_data",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-p",
        "--pipelines",
        help="collect pipeline status and results",
        dest="collect_pipelines_data",
        action="store_true",
        default=False,
    )

    parser.add_argument(
        "-o",
        "--output",
        help="output file name, default: collected_df.pckl.gzip",
        type=str,
        default="collected_df.pkl.gz",
    )

    parser.add_argument("--plot",
                        help="whether to plot the dataset or not",
                        dest="plot_df",
                        action="store_true",
                        default=False)


    args_parse = parser.parse_args(args)

    is_collect_raw_data = args_parse.collect_raw_data
    is_collect_pipelines = args_parse.collect_pipelines_data
    path = os.path.abspath(args_parse.path)
    output_name = args_parse.output

    if is_collect_raw_data:
        logger.info(
            "`-r` option has no effect and is deprecated. You can just run without this option"
        )


    if is_collect_pipelines:
        collect_pipelines(location_root=path)
        return

    collect_raw_data(output_name, path)

    is_plot_df = args_parse.plot_df

    if is_plot_df:
        plot_df(output_name)


if __name__ == "__main__":
    main(sys.argv[1:])
