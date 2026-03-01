import logging
import os
from ase import Atoms
from abc import ABCMeta, abstractmethod
from collections import defaultdict

from amstools.calculators.dft.vasp import AMSVasp
from amstools.pipeline.exc import PausedException
from amstools.utils import serialize_class, deserialize_class, atoms_fromdict, general_atoms_copy

from amstools.pipeline.generalstructure import GeneralStructure

# step statuses
FINISHED = "finished"
ERROR = "error"
PAUSED = "paused"
CALCULATING = "calculating"
CREATED = "created"
FAILED = "failed"
PARTIALLY_FINISHED = "partially_finished"

# Ranking of statuses, statuses rank must be unique!
STATUS_ORDER_DICT = {
    FINISHED: 2,
    PARTIALLY_FINISHED: 1.5,  # when one or more steps are allowed to fail
    CALCULATING: 1,
    CREATED: 0,
    PAUSED: -0.25,
    ERROR: -0.5,
    FAILED: -2,
}

ORDER_STATUS_DICT = {v: k for k, v in STATUS_ORDER_DICT.items()}


def get_pipeline_global_status(pipeline):
    """
    Return the reduced status of Pipeline or ParallelStep
    :param pipeline: Pipeline or ParallelStep
    :return: reduced status of Pipeline or ParallelStep
    """
    statuses = []
    for name, step in pipeline.steps.items():
        st = step.status
        if step.allow_fail and st in [FAILED, ERROR]:
            st = PARTIALLY_FINISHED
        statuses.append(st)
    # if calculating status in list, the remove all created
    if CALCULATING in statuses:
        statuses = [status for status in statuses if status != CREATED]
    statuses_order = [STATUS_ORDER_DICT[status] for status in statuses]
    # if part jobs are created but some calculating
    if (
        min(statuses_order) == STATUS_ORDER_DICT[CREATED]
        and max(statuses_order) == STATUS_ORDER_DICT[CALCULATING]
    ):
        return CALCULATING
    else:
        return ORDER_STATUS_DICT[min(statuses_order)]


def extract_job_options(settings, verbose=True):
    job_options = {}
    whitelist_options = PipelineJobOptions.options_key
    for option_name, option_value in settings.options.items():
        if verbose:
            logging.info("{} {}".format(option_name, option_value))
        if option_name in whitelist_options:
            for option_k, option_v in option_value.items():
                if verbose:
                    logging.info("- {} {}".format(option_k, option_v))
                job_options[option_k] = option_v
    return job_options


class Options:
    """
    Parent class for options, which consume options as a **kwargs param for constructor and have update_options method
    """

    options_key = []

    def __init__(self, **kwargs):
        # self.options = {k: {} for k in self.options_key}
        self.options = defaultdict(dict)
        self.update_options(**kwargs)

    def update_options(self, **kwargs):
        if kwargs is not None:
            for k, v in kwargs.items():
                if k in self.options_key:
                    self.options[k] = v

    def copy(self):
        new_object = Options(**self.options.copy())
        return new_object


class PipelineJobOptions(Options):
    options_key = ["job_options"]

    def todict(self):
        return self.options.copy()

    @classmethod
    def fromdict(cls, dct):
        options = cls()
        options.options = dct.copy()


# Backward compatibility aliases
CLASS_ALIASES = {
    "AMStoolsMurnaghanStep": "amstools.properties.murnaghan.MurnaghanCalculator",
    "AMStoolsElasticmatrixStep": "amstools.properties.elasticmatrix.ElasticMatrixCalculator",
    "AMStoolsTransformationPathStep": "amstools.properties.transformationpath.TransformationPathCalculator",
    "AMStoolsPhonopyStep": "amstools.properties.phonons.PhonopyCalculator",
    "AMStoolsDefectFormationStep": "amstools.properties.defectformation.DefectFormationCalculator",
    "AMStoolsThermodynamicQHAStep": "amstools.properties.tqha.ThermodynamicQHACalculator",
    "AMStoolsNNExpansionStep": "amstools.properties.nnexpansion.NearestNeighboursExpansionCalculator",
    "AMStoolsSurfaceEnergyStep": "amstools.properties.surface.SurfaceEnergyCalculator",
    "AMStoolsStackingFaultStep": "amstools.properties.stackingfault.StackingFaultCalculator",
    "AMStoolsInterstitialFormationStep": "amstools.properties.interstitial.InterstitialFormationCalculator",
    "AMStoolsRandomDeformationStep": "amstools.properties.randomdeform.RandomDeformationCalculator",
    "AMStoolsFullRelaxationStep": "amstools.properties.relaxation.StepwiseOptimizer",
    "AMStoolsRelaxationStep": "amstools.properties.relaxation.SpecialOptimizer",
    "AMStoolsVolumeOnlyRelaxationStep": "amstools.properties.relaxation.IsoOptimizer",
    "AMStoolsStaticStep": "amstools.properties.static.StaticCalculator",
    "AMStoolsSurfaceDecohesionStep": "amstools.properties.surface.SurfaceDecohesionCalculator",
    # Full legacy paths (module paths as serialized in pipeline.json files)
    "amstools.pipeline.amstoolssteps.elasticmatrixstep.AMStoolsElasticmatrixStep": "amstools.properties.elasticmatrix.ElasticMatrixCalculator",
    "amstools.pipeline.amstoolssteps.murnaghanstep.AMStoolsMurnaghanStep": "amstools.properties.murnaghan.MurnaghanCalculator",
    "amstools.pipeline.amstoolssteps.phonopystep.AMStoolsPhonopyStep": "amstools.properties.phonons.PhonopyCalculator",
    "amstools.pipeline.amstoolssteps.relaxation.AMStoolsFullRelaxationStep": "amstools.properties.relaxation.StepwiseOptimizer",
    "amstools.pipeline.amstoolssteps.relaxation.AMStoolsRelaxationStep": "amstools.properties.relaxation.SpecialOptimizer",
    "amstools.pipeline.amstoolssteps.relaxation.AMStoolsVolumeOnlyRelaxationStep": "amstools.properties.relaxation.IsoOptimizer",
    "amstools.pipeline.amstoolssteps.relaxation.AMStoolsStaticStep": "amstools.properties.static.StaticCalculator",
    "amstools.relaxation.relaxation.StepwiseOptimizer": "amstools.properties.relaxation.StepwiseOptimizer",
    "amstools.pipeline.amstoolssteps.defectformation.AMStoolsDefectFormationStep": "amstools.properties.defectformation.DefectFormationCalculator",
    "amstools.pipeline.amstoolssteps.defectformationstep.AMStoolsDefectFormationStep": "amstools.properties.defectformation.DefectFormationCalculator",
    "amstools.pipeline.amstoolssteps.interstitialstep.AMStoolsInterstitialFormationStep": "amstools.properties.interstitial.InterstitialFormationCalculator",
    "amstools.pipeline.amstoolssteps.nnexpansionstep.AMStoolsNNExpansionStep": "amstools.properties.nnexpansion.NearestNeighboursExpansionCalculator",
    "amstools.pipeline.amstoolssteps.randomdeformstep.AMStoolsRandomDeformationStep": "amstools.properties.randomdeform.RandomDeformationCalculator",
    "amstools.pipeline.amstoolssteps.stackingfaultstep.AMStoolsStackingFaultStep": "amstools.properties.stackingfault.StackingFaultCalculator",
    "amstools.pipeline.amstoolssteps.surfacestep.AMStoolsSurfaceEnergyStep": "amstools.properties.surface.SurfaceEnergyCalculator",
    "amstools.pipeline.amstoolssteps.surfacestep.AMStoolsSurfaceDecohesionStep": "amstools.properties.surface.SurfaceDecohesionCalculator",
    "amstools.pipeline.amstoolssteps.transformationpathstep.AMStoolsTransformationPathStep": "amstools.properties.transformationpath.TransformationPathCalculator",
    "amstools.pipeline.amstoolssteps.tqha.AMStoolsThermodynamicQHAStep": "amstools.properties.tqha.ThermodynamicQHACalculator",
}


def serialize_class(cls):
    return "{}.{}".format(cls.__module__, cls.__name__)


def deserialize_class(cls_str):
    if cls_str in CLASS_ALIASES:
        cls_str = CLASS_ALIASES[cls_str]
    components = cls_str.split(".")
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


class PipelineEngine(Options):
    """
    calculator: ase.calculator, CalculatorType
    """

    def __init__(self, calculator, **kwargs):
        Options.__init__(self, **kwargs)

        self.calculator = calculator

    def todict(self):
        # TODO: implement
        # calc_dct["__cls__"] = serialize_class(self.inner_calculator.__class__)
        calc_dct = self.calculator.todict()
        if "__cls__" not in calc_dct:
            calc_dct["__cls__"] = serialize_class(self.calculator.__class__)
        return {"calculator": calc_dct}  # using ASE serialization protocol

    @classmethod
    def fromdict(cls, dct, ignore_import_errors=False):
        calc_dct = dct["calculator"]
        _ams_calculator_class = calc_dct.get("__cls__") or calc_dct.get(
            "_ams_calculator_class"
        )
        _cls = None
        if _ams_calculator_class == "AMSVasp":
            # for backward compatibility
            _cls = AMSVasp
        else:
            try:
                _cls = deserialize_class(_ams_calculator_class)
            except (ImportError, ValueError) as e:
                if not ignore_import_errors:
                    raise e
                else:
                    logging.info(
                        "Could not import {}, but `ignore_import_errors=True`, will be ignored".format(
                            _ams_calculator_class
                        )
                    )

        if _cls is not None:
            try:
                _calc = _cls.fromdict(calc_dct)
            except AttributeError:
                try:
                    _calc = _cls(**calc_dct)
                except Exception as e:
                    if not ignore_import_errors:
                        raise e
                    else:
                        return None  # engine
            engine = PipelineEngine(calculator=_calc)
        else:
            engine = None

        return engine

    def copy(self):
        if hasattr(self.calculator, "copy"):
            new_calculator = self.calculator.copy()
        else:
            new_calculator = self.calculator

        if hasattr(self.calculator, "id"):
            new_calculator.id = self.calculator.id

        new_object = PipelineEngine(
            calculator=new_calculator, options=self.options.copy()
        )
        return new_object

    @property
    def paused(self):
        if hasattr(self.calculator, "paused"):
            return self.calculator.paused
        return False

    @property
    def write_input_only(self):
        if hasattr(self.calculator, "write_input_only"):
            return self.calculator.write_input_only
        return False

    @write_input_only.setter
    def write_input_only(self, val):
        try:
            self.calculator.write_input_only = val
        except AttributeError as e:
            logging.warning(
                "Could not setup calculator.write_input_only attribute: {}".format(e)
            )


class PipelineStep(metaclass=ABCMeta):
    """
    structure: GeneralStructure
    engine: PipelineEngine object

    General settings:
    name_tag: additional name tag for properties (default - None)
    cleanup: clean up flag for job (default = False)

    """

    JOB_NAME = ""

    def __init__(
        self,
        structure=None,
        engine=None,
        allow_fail=False,
        verbose=True,
        name=None,
        name_tag=None,
        cleanup=None,
        force_restart=False,
        **kwargs,
    ):
        self._engine = None
        self._structure = None
        self.property = None

        self.name = name
        self.pipeline = None  # reference to pipeline

        self.finished = False
        self.parent_path = None
        self.working_dir = None
        # TODO: work on and test!
        self.allow_fail = allow_fail
        self.job = None

        self.name_tag = name_tag
        self.cleanup = cleanup
        self.force_restart = force_restart

        self.structure = structure
        self.engine = engine
        if self.engine is not None:
            self.engine.update_options(**kwargs)
        self.status = CREATED

        self.job_options = PipelineJobOptions(**kwargs)

        self.verbose = verbose

    def __repr__(self):
        """Friendly representation of the pipeline step"""
        return f"{self.__class__.__name__}(name={repr(self.name)}, status={repr(self.status)})"

    def __add__(self, other):
        from amstools.pipeline.pipeline import Pipeline

        if isinstance(other, PipelineStep):
            return Pipeline(steps=[self, other])
        elif isinstance(other, Pipeline):
            return Pipeline(steps=[self]) + other
        return NotImplemented

    def copy_reset(self, copy_engine=False):
        kwargs = self.job_options.options.copy()
        kwargs["allow_fail"] = self.allow_fail
        kwargs["name"] = self.name
        kwargs["name_tag"] = self.name_tag
        kwargs["cleanup"] = self.cleanup
        kwargs["force_restart"] = self.force_restart

        return self.__class__(
            structure=self.structure,
            engine=self.engine if copy_engine else None,
            **kwargs,
        )

    @property
    def structure(self):
        return self._structure

    @structure.setter
    def structure(self, value):
        if value is not None:
            if isinstance(value, Atoms):
                self._structure = general_atoms_copy(value)
            elif hasattr(value, "atoms"): # Handle GeneralStructure for back compat
                self._structure = general_atoms_copy(value.atoms)
            else:
                self._structure = value # Assume it's already an atoms-like object or handled elsewhere
        else:
            self._structure = None

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

    @property
    def value(self):
        if self.job is not None:
            return self.job.value

    def step_is_done(self):
        return self.job_is_done()

    def run(self, verbose=True, raise_errors=True):
        try:
            self.run_submit_check(verbose, raise_errors)

            if self.job_is_done():
                self.status = FINISHED
                logging.info("Job is done")
                if self.cleanup:
                    self.job_cleanup()
                return True

        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            self.status = ERROR
            # TODO: add tests
            if self.allow_fail:
                logging.info("Current step is failed with error: ")
                logging.info(e, exc_info=True)
                logging.info("but the step is allowed to fail, continue to next step")
            else:
                raise e

    def run_submit_check(self, verbose=False,raise_errors=True):
        self.working_dir = (
            os.path.join(self.parent_path, self.JOB_NAME) if self.parent_path else None
        )
        if verbose:
            logging.debug("Check if job is done")
        if self.job_is_done():
            return

        if verbose:
            logging.debug("job is not done")

        if self.job_is_running():
            if verbose:
                logging.debug("job is submitted")
            self.status = CALCULATING
            if self.job_has_error():
                if verbose:
                    logging.debug("job has error, try to repair")
                self.status = ERROR
                self.job_repair()
        else:
            if verbose:
                logging.debug("calculate job")
            self.status = CALCULATING
            try:
                if self.working_dir:
                    self.engine.calculator.directory = self.working_dir
                    logging.debug(
                        f"PipelineStep::run_submit_check: self.engine.directory={self.working_dir}"
                    )
                self.calculate(verbose=verbose,raise_errors=raise_errors)
                if self.working_dir and not self.engine.paused:
                    self.save_results_locally(path=self.working_dir)
            except PausedException as e:
                self.status = PAUSED
                raise e
            except Exception as e:
                self.status = ERROR
                raise

    def get_final_structure(self):
        try:
            if self.step_is_done():
                structure = self.load_final_structure()
                if structure is not None:
                    # Return raw Atoms if it was wrapped or just return as is
                    if hasattr(structure, "atoms"):
                        return structure.atoms
                    return structure
                else:
                    raise ValueError("Couldn't load a final structure")
            elif self.allow_fail:
                # if not done and allow fail, get initial structure
                return self.structure
            else:
                raise ValueError("Step is not finished")
        except KeyboardInterrupt as e:
            raise e
        except Exception as e:
            self.status = ERROR
            raise e

    def todict(self):
        """
        Serialize the state of the pipelinestep to dictionary, including
            structure, engine, storage, export_to_db, export_as_visible
        :return: dict-JSON
        """
        step_dict = {}

        if self.structure:
            step_dict["structure"] = self.structure.todict()
        if self.engine:
            step_dict["engine"] = self.engine.todict()
        if self.job:
            step_dict["job"] = self.job.todict()
        if self.job_options:
            step_dict["job_options"] = self.job_options.todict()

        step_dict["status"] = self.status
        step_dict["finished"] = self.finished
        step_dict["name"] = self.name
        step_dict["__cls__"] = serialize_class(self.__class__)
        step_dict["options"] = {
            "allow_fail": self.allow_fail,
            "name_tag": self.name_tag,
            "cleanup": self.cleanup,
            "force_restart": self.force_restart,
        }
        return step_dict

    @classmethod
    def fromdict(
        cls, step_dict, storage=None, ignore_import_errors=False, load_engine=True
    ):
        structure_dict = step_dict.get("structure") or step_dict.get("basis_ref")
        structure = (
            atoms_fromdict(structure_dict)
            if structure_dict
            else None
        )

        engine = (
            PipelineEngine.fromdict(
                step_dict["engine"], ignore_import_errors=ignore_import_errors
            )
            if "engine" in step_dict and load_engine
            else None
        )
        options = step_dict.get("options", {})
        job_options = step_dict.get("job_options", {})

        # TODO: resolve cls
        cls_resolved = deserialize_class(step_dict["__cls__"])
        pipelinestep = cls_resolved(
            structure=structure,
            engine=engine,
            storage=storage,
            **job_options,
            name=step_dict.get("name"),
            allow_fail=options.get("allow_fail", False),
            name_tag=options.get("name_tag"),
            cleanup=options.get("cleanup"),
            force_restart=options.get("force_restart", False),
        )
        pipelinestep.status = step_dict["status"]
        pipelinestep.finished = step_dict["finished"]

        # # Restore internal state if present in step_dict (for combined step-calculators)
        _value = step_dict.get("_value") or step_dict.get("_VALUE")
        if _value is not None:
            pipelinestep._value = _value
            # Handle postponed loading for Phonopy
            # if hasattr(pipelinestep, "_postpone_load_phonopy_fromdict"):
            #     pipelinestep._postpone_load_phonopy_fromdict = True

        if "output_structures" in step_dict:
            from amstools.utils import output_structures_fromdict
            pipelinestep.output_structures_dict = output_structures_fromdict(
                step_dict["output_structures"]
            )

        if "job" in step_dict:
            job_class = deserialize_class(step_dict["job"]["__cls__"])
            pipelinestep.job = job_class.fromdict(step_dict["job"])

        return pipelinestep

    @abstractmethod
    def job_is_done(self):
        raise NotImplementedError("Should be implemented in derived classes")

    @abstractmethod
    def job_has_error(self):
        raise NotImplementedError("Should be implemented in derived classes")

    @abstractmethod
    def job_repair(self):
        raise NotImplementedError("Should be implemented in derived classes")

    @abstractmethod
    def job_is_running(self):
        raise NotImplementedError("Should be implemented in derived classes")

    @abstractmethod
    def calculate(self):
        raise NotImplementedError("Should be implemented in derived classes")

    @abstractmethod
    def load_final_structure(self):
        raise NotImplementedError("Shoule be implemented in derived classes")

    @abstractmethod
    def job_cleanup(self):
        raise NotImplementedError("Should be implemented in derived classes")

    @abstractmethod
    def job_delete(self):
        raise NotImplementedError("Should be implemented in derived classes")

    @classmethod
    def HELP(cls):
        description = "job name is {JOB_NAME}".format(JOB_NAME=cls.JOB_NAME)
        return description

    def save_results_locally(self, path=None):
        raise NotImplementedError()

    def plot(self, **kwargs):
        if self.job is not None and hasattr(self.job, "plot"):
            return self.job.plot(**kwargs)
        # Fallback if self is the job (e.g. GeneralCalculator is both step and job logic often mixed)
        if hasattr(self, "job_is_done") and self != self.job:
             # This meant to catch cases where we might be a calculator itself
             pass
        raise NotImplementedError(f"Plotting not implemented for {self.__class__.__name__}")
