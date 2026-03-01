#!/usr/bin/env python
import argparse
import os
import pkg_resources
import socket
import sys

from amstools.highthroughput.webinterface.app import app
from amstools.utils import SQLiteStateDict


def main(args=None):
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser(
        prog="ams_ht_webinterface", description="AMS HT webinterface"
    )
    parser.add_argument(
        "-p", "--port", help="port", type=int, dest="port", default=8000
    )

    parser.add_argument(
        "--host", help="hostname", type=str, dest="host", default="0.0.0.0"
    )

    parser.add_argument(
        "-wd",
        "--working-dir",
        help="working dir",
        type=str,
        dest="working_dir",
        default=".",
    )

    args_parse = parser.parse_args(args)

    template_folder = pkg_resources.resource_filename(
        "amstools.highthroughput.webinterface.resources", "templates"
    )
    static_folder = pkg_resources.resource_filename(
        "amstools.highthroughput.webinterface.resources", "static"
    )
    app.template_folder = template_folder  # hope that works
    app.static_folder = static_folder

    listen_host = args_parse.host
    port = args_parse.port
    working_dir = args_parse.working_dir

    app.ROOT_PATH = os.path.abspath(working_dir)
    state_db_fname = os.path.join(app.ROOT_PATH, "state_dict.db")
    app.state_dict = SQLiteStateDict(state_db_fname)

    self_hostname = socket.gethostname()

    print(
        "Starting ams-highthroughput webinterface at {self_hostname}:  http://{listen_host}:{port}".format(
            self_hostname=self_hostname, listen_host=listen_host, port=port
        )
    )
    print("Root path: ", app.ROOT_PATH)
    app.run(host=listen_host, port=port)


if __name__ == "__main__":
    main(sys.argv[1:])
