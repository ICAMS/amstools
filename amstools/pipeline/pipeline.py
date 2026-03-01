import hashlib
import logging
import numpy as np
import os
import traceback
from ase import Atoms

from collections import OrderedDict

from ase.io import jsonio

from amstools.pipeline import scheduler as pq
from amstools.pipeline.exc import PausedException
from amstools.pipeline.generalstructure import GeneralStructure
from amstools.pipeline.pipelinestep import (
    PipelineEngine,
    ERROR,
    CALCULATING,
    FAILED,
    PAUSED,
    get_pipeline_global_status,
    PipelineStep,
    PARTIALLY_FINISHED,
)
from amstools.utils import general_atoms_copy

PIPELINE_LOCK = "pipeline.lock"


def my_encode(d):
    if isinstance(d, dict):
        # If the current object is a dictionary, traverse its keys and values
        for key, value in d.items():
            d[key] = my_encode(value)
    elif isinstance(d, list):
        # If the current object is a list, traverse its elements
        d = [my_encode(item) for item in d]
    elif isinstance(d, tuple):
        # If the current object is a tuple, traverse its elements and recreate the tuple
        d = tuple(my_encode(item) for item in d)
    elif isinstance(d, (np.float32, np.float64)):
        # Convert numpy.float32 to float
        d = float(d)
    return d


class Pipeline:


    def __init__(
        self,
        steps=None,
        init_structure=None,
        engine=None,
        path=None,
        write_input_only=False,
        autosave=False,
        json_filename="pipeline.json",
        verbose=True,
    ):
        self._steps = None
        self._path = None
        self._init_structure = None
        self._engine = None
        self.json_filename = json_filename
        self.autosave = autosave
        self.verbose = verbose

        self.write_input_only = write_input_only

        if isinstance(steps, (list, tuple)):
            new_steps = OrderedDict()
            for step in steps:
                name = self._get_unique_step_name(step, new_steps.keys())
                step.name = name
                new_steps[name] = step
            steps = new_steps

        self.steps = steps
        self.path = path  # after setting self.steps (so property will initialize)
        self.init_structure = init_structure # set property
        self.engine = engine

        self.break_flag = False
        self.after_step_iterate_callback = None

    def _get_unique_step_name(self, step, existing_names):
        if step.name:
            base_name = step.name
        elif hasattr(step, "property_name") and step.property_name:
            base_name = step.property_name
        else:
            base_name = step.__class__.__name__

        name = base_name
        counter = 1
        while name in existing_names:
            name = f"{base_name}_{counter}"
            counter += 1
        return name

    def __add__(self, other):
        if isinstance(other, PipelineStep):
            new_steps = self.steps.copy() if self.steps else OrderedDict()

            name = self._get_unique_step_name(other, new_steps.keys())
            other.name = name
            new_steps[name] = other

            return Pipeline(
                steps=new_steps,
                init_structure=self.init_structure,
                engine=self.engine,
                path=self.path,
                write_input_only=self.write_input_only,
                autosave=self.autosave,
                json_filename=self.json_filename,
                verbose=self.verbose,
            )
        elif isinstance(other, Pipeline):
            new_steps = self.steps.copy() if self.steps else OrderedDict()
            if other.steps:
                for k, v in other.steps.items():
                    name = self._get_unique_step_name(v, new_steps.keys())
                    v.name = name
                    new_steps[name] = v

            return Pipeline(
                steps=new_steps,
                init_structure=self.init_structure,
                engine=self.engine,
                path=self.path,
                write_input_only=self.write_input_only,
                autosave=self.autosave,
                json_filename=self.json_filename,
                verbose=self.verbose,
            )
        return NotImplemented

    def __repr__(self):
        """Friendly representation of the pipeline"""
        s = f"Pipeline(status={repr(self.status)}, path={repr(self.path)})\n"
        if self.steps:
            for i, (name, step) in enumerate(self.steps.items()):
                # Use a marker if it's currently calculating
                marker = "*" if step.status == CALCULATING else " "
                s += f"{marker} [{i}] {name}: {repr(step)}\n"
        return s

    @property
    def path(self):
        return self._path

    @path.setter
    def path(self, val):
        self._path = val
        if self.steps:
            for name, step in self.steps.items():
                step.parent_path = self._path

    @property
    def init_structure(self):
        return self._init_structure

    @init_structure.setter
    def init_structure(self, value):
        if value is None:
            self._init_structure = None
        elif isinstance(value, Atoms):
            self._init_structure = general_atoms_copy(value)
        elif hasattr(value, "atoms"): # Back compat for GeneralStructure
            self._init_structure = general_atoms_copy(value.atoms)
        else:
            self._init_structure = value

    @property
    def engine(self):
        return self._engine

    @engine.setter
    def engine(self, value):
        if value is None:
            self._engine = None
        elif isinstance(value, PipelineEngine):
            self._engine = value
        else:
            self._engine = PipelineEngine(value)

    def iterate(self, init_structure=None, verbose=None, **kwargs):
        if verbose is None:
            verbose = self.verbose

        if init_structure is None:
            current_structure = self.init_structure
        else:
            current_structure = init_structure

        steps = self.steps

        self.break_flag = False
        prev_step = None
        for step_ind, (step_name, step) in enumerate(steps.items()):
            # autosave option
            if self.autosave:
                self.to_json()
                logging.info("Pipeline saved into JSON")

            step.parent_path = self.path  # always update path
            step.pipeline = self  # pass reference to pipeline

            if step.finished:
                prev_step = step
                continue

            if step_ind > 0:
                current_structure = prev_step.get_final_structure()
            if verbose:
                logging.info("=======================")
                logging.info("Step: " + str(step_name))

            logging.debug(
                f"Pipeline.iterate step={step}, structure = {current_structure}"
            )
            step.structure = current_structure
            self.engine.write_input_only = self.write_input_only
            try:
                self.engine.calculator.paused = False
            except AttributeError as e:
                logging.warning(f"Could not setup calculator.paused attribute: {e}")
            step.engine = self.engine
            step.JOB_NAME = str(step_name)
            try:
                step.status = CALCULATING
                # autosave option
                if self.autosave:
                    self.to_json()
                    logging.info("Pipeline saved into JSON")
                step.run(verbose=verbose)
                if self.engine.paused:
                    raise PausedException
            except KeyboardInterrupt:
                raise
            except (RuntimeError, ConnectionError) as e:
                step.status = ERROR
                logging.error(f"Error has been occurred: {e}")
                if verbose:
                    logging.info(f"Error has been occurred: {e}")
                traceback.print_exc()
                if self.autosave:
                    self.to_json()
                    logging.info("Pipeline saved into JSON")
                raise e
            except PausedException:
                step.status = PAUSED
                step.flag_job_is_done = False
                break
            if verbose:
                logging.info(f"Step status: {step.status}")
                logging.info("=======================")
            if self.after_step_iterate_callback:
                self.after_step_iterate_callback(self, step)

            if step.step_is_done():
                self.break_flag = False
                step.finished = True
            elif not step.allow_fail:
                if verbose:
                    logging.info("***********************")
                    logging.info("* Pipeline paused ... *")
                    logging.info("***********************")
                self.break_flag = True
                break
            else:
                # allow to fail
                self.break_flag = False
                step.status = PARTIALLY_FINISHED
            prev_step = step
        # autosave option
        if self.autosave:
            self.to_json()
            logging.info("Pipeline saved into JSON")

    def is_finished(self):
        return np.all([step.finished for name, step in self.steps.items()])

    def run(self, init_structure=None, engine=None, verbose=None, **kwargs):
        if init_structure is not None:
            self.init_structure = init_structure
        if engine is not None:
            self.engine = engine

        self.validate_setups()

        self.iterate(verbose=verbose, **kwargs)

    def validate_setups(self):
        if self.engine is None:
            raise ValueError("Could not run pipeline, because `engine` is None")
        if self.init_structure is None:
            raise ValueError("Could not run pipeline, because `init_structure` is None")

    def is_error(self):
        for step_name, step in self.steps.items():
            if not step.finished:
                if step.status == ERROR:
                    return True

        return False

    @property
    def status(self):
        return get_pipeline_global_status(self)

    @property
    def steps_statuses(self):
        return [step.status for name, step in self.steps.items()]

    @property
    def named_steps_statuses(self):
        return {name: step.status for name, step in self.steps.items()}

    def __getitem__(self, item):
        if isinstance(item, int):
            return list(self.steps.values())[item]
        return self.steps[item]

    def get_final_structure(self):
        return self[-1].get_final_structure()
        
    def _ipython_key_completions_(self):
        return self.steps.keys()

    def plot(self, property_axes=None, **kwargs):
        """
        Plot results for all steps that support plotting
        """
        import matplotlib.pyplot as plt
        
        # Count steps that can plot
        plot_steps = [s for s in self.steps.values() if hasattr(s, "plot")]
        
        if not plot_steps:
            logging.warning("No steps support plotting")
            return

        # Simple plotting strategy: if multiple steps, create subplots? 
        # Or just plot one by one independently?
        # For now, let's just allow individual steps to plot to the current figure 
        # or create a new figure if needed.
        # But plotting everything on one axis usually doesn't make sense for different properties.
        # So we just call plot() on each step. 
        # If the user wants specific axes, they should probably do it manually or we need more sophisticated logic.
        
        axes = []
        for name, step in self.steps.items():
            try:
                logging.info(f"Plotting step: {name}")
                if property_axes is None:
                    ax = plt.gca()
                else:
                    ax = property_axes.get(name) or plt.gca()
                step.plot(ax=ax,**kwargs)
                
                axes.append(ax)
                plt.show()
            except (NotImplementedError, AttributeError):
                pass
            except Exception as e:
                logging.warning(f"Failed to plot step {name}: {e}")
        return axes


    def todict(self):
        steps_dct = {}
        for step_name, step in self.steps.items():
            steps_dct[step_name] = step.todict()
        dct = {
            "steps": steps_dct,
            "path": self.path,  # TODO: do we need to store path ?
            "autosave": self.autosave,
            "write_input_only": self.write_input_only,
        }

        if self.init_structure is not None:
            dct["init_structure"] = self.init_structure.todict()

        if self.engine is not None:
            dct["engine"] = self.engine.todict()

        return dct

    @classmethod
    def fromdict(self, dct, ignore_import_errors=False, load_engine=True):
        steps_dict = {}
        for step_name, step_dict in dct["steps"].items():
            step = PipelineStep.fromdict(
                step_dict,
                ignore_import_errors=ignore_import_errors,
                load_engine=load_engine,
            )
            step.name = step_name
            steps_dict[step_name] = step

        if load_engine:
            engine = (
                PipelineEngine.fromdict(
                    dct["engine"], ignore_import_errors=ignore_import_errors
                )
                if "engine" in dct
                else None
            )
        else:
            engine = None

        init_structure = (
            GeneralStructure.fromdict(dct["init_structure"])
            if "init_structure" in dct
            else None
        )
        path = dct.get("path")
        pipe = Pipeline(init_structure=init_structure, engine=engine, path=path)
        pipe.steps = steps_dict
        if "autosave" in dct:
            pipe.autosave = dct["autosave"]
        if "write_input_only" in dct:
            pipe.write_input_only = dct["write_input_only"]
        return pipe

    def to_json(self, filename=None):
        if filename is None:
            # autosave option
            if self.path is not None:
                filename = os.path.join(self.path, self.json_filename)
            else:
                # do not do autosave if path is not provided
                return
        pipe_dict = self.todict()
        dirname = os.path.dirname(filename)
        if not os.path.isdir(dirname) and dirname not in ["", "."]:
            os.makedirs(dirname, exist_ok=True)
        # this extra encoder converts np.float to float
        pipe_dict = my_encode(pipe_dict)
        with open(filename, "w") as f:
            jsonio.write_json(f, pipe_dict)

    @classmethod
    def read_json(
        cls, filename, autopath=True, ignore_import_errors=False, load_engine=True
    ):
        with open(filename, "r") as f:
            pipe_dict_load = jsonio.read_json(f)

        pipe_loaded = Pipeline.fromdict(
            pipe_dict_load,
            ignore_import_errors=ignore_import_errors,
            load_engine=load_engine,
        )

        # set path to current filename path if autopath option
        if autopath:
            absfname = os.path.abspath(filename)
            path = os.path.dirname(absfname)
            pipe_loaded.path = path

        return pipe_loaded

    def submit_to_scheduler(self, options: dict, working_dir=None):
        if working_dir is None:
            if self.path is not None:
                working_dir = self.path
            else:
                raise ValueError("Neither pipeline.path nor working_dir are provided")

        self.validate_setups()

        working_dir = os.path.abspath(working_dir)
        if not os.path.isdir(working_dir):
            os.makedirs(working_dir, exist_ok=True)

        if options["scheduler"] == "slurm":
            scheduler = pq.SLURM(options, cores=options["cores"], directory=working_dir)
        elif options["scheduler"] == "sge":
            scheduler = pq.SGE(options, cores=options["cores"], directory=working_dir)
        else:
            raise ValueError("Unknown scheduler")

        self.to_json(os.path.join(working_dir, self.json_filename))
        scheduler.maincommand = "ams_pipeline {}".format(self.json_filename)
        submission_script = os.path.join(working_dir, "ams_pipeline_job.sh")
        scheduler.write_script(submission_script)

        res = scheduler.submit()
        lines = res.stdout.readlines()
        return scheduler.get_job_id(lines)

    def hash(self):
        """
        Compute MD5 hase based on steps name and order
        """
        uniq_str = "::UNIQUE_STEP_SEPARATOR::".join(self.steps.keys())
        h = hashlib.md5(uniq_str.encode("utf-8")).hexdigest()
        return h

    def copy_reset(self):
        """
        Create a new pipeline with a steps copied from the current one (incl. job options)
        :return new pipeline
        """
        new_steps = {}
        for step_name, step in self.steps.items():
            new_steps[step_name] = step.copy_reset()
        new_pipe = Pipeline(new_steps, init_structure=self.init_structure)
        return new_pipe

    def rerun_for(self, structure=None, engine=None):
        """
        Copy current pipeline and re-run it with another engine
        :return new pipeline
        """
        new_pipe = self.copy_reset()
        new_pipe.run(structure=structure, engine=engine)
        return new_pipe
