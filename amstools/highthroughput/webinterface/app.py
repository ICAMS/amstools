import argparse
import json
import os.path
import time
import socket
import sys

from flask import Flask, request, render_template, abort, send_file

from amstools.pipeline import *
from amstools.utils import JsonNumpyEncoder

app = Flask(__name__)


@app.route("/", methods=["GET"])
def index():
    start_time = time.time()
    states_info = app.state_dict.get_all_info()
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    return render_template(
        "overview.html", states_info=states_info, elapsed_time=elapsed_time, str=str
    )


@app.route("/pipeline", methods=["GET"])
def pipeline():
    name = request.args.get("name")
    start_time = time.time()
    pipe_state = app.state_dict[name]
    pipe_data = []
    pipeline_fname = os.path.join(app.ROOT_PATH, name, "pipeline.json")
    init_atoms = None
    if os.path.isfile(pipeline_fname):
        pipe = Pipeline.read_json(pipeline_fname)
        init_atoms = pipe.init_structure.atoms
        for (step_name, step), status in zip(pipe.steps.items(), pipe.steps_statuses):
            pipe_data.append(
                {
                    "name": step_name,
                    "type": str(step.CALCULATOR.property_name),
                    "status": status,
                }
            )

    pipe_state = json.dumps(
        pipe_state, sort_keys=True, indent=4, separators=(",", ": ")
    )
    elapsed_time = time.time() - start_time
    return render_template(
        "pipeline.html",
        pipeline_fname=pipeline_fname,
        pipe_data=pipe_data,
        pipe_state=pipe_state,
        name=name,
        elapsed_time=elapsed_time,
        init_atoms=str(init_atoms),
    )


@app.route("/step", methods=["GET"])
def step():
    name = request.args.get("name")
    step_name = request.args.get("step_name")
    pipeline_fname = os.path.join(app.ROOT_PATH, name, "pipeline.json")
    if os.path.isfile(pipeline_fname):
        start_time = time.time()
        pipe = Pipeline.read_json(pipeline_fname)
        step = pipe[step_name]
        if step is not None:
            value_str = json.dumps(
                step.value,
                sort_keys=True,
                indent=4,
                separators=(",", ": "),
                cls=JsonNumpyEncoder,
            )
        else:
            value_str = "<<NO-DATA>>"
        elapsed_time = time.time() - start_time
        return render_template(
            "step.html",
            name=name,
            step_name=step_name,
            value_str=value_str,
            elapsed_time=elapsed_time,
        )
    else:
        return "{} not found".format(pipeline_fname)


@app.route("/files", defaults={"req_path": ""})
@app.route("/files/<path:req_path>")
def dir_listing(req_path):
    start_time = time.time()
    BASE_DIR = app.ROOT_PATH
    # Joining the base and the requested path
    abs_path = os.path.join(BASE_DIR, req_path)

    # Return 404 if path doesn't exist
    if not os.path.exists(abs_path):
        return abort(404)

    # Check if path is a file and serve
    if os.path.isfile(abs_path):
        return send_file(abs_path, mimetype="text/plain")

    # Show directory contents
    files = sorted(os.listdir(abs_path))
    elapsed_time = time.time() - start_time
    return render_template(
        "files.html", files=files, req_path=req_path, elapsed_time=elapsed_time
    )
