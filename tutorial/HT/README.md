# AMS-highthroughput calculations

AMStools provides several utilities for running high-throughput (HT) DFT calculations on workstations or clusters (SGE,
SLURM).

# Installation

To get started, follow these steps:

1. Create a new/separate conda environment, e.g., `ace`. Make sure to use the same environment name in both
   the `run_vasp_custodian.sh` and `.amstools` files (in `conda activate ace` commands).

   ```bash
   mamba env create -f environment.yml --force
   ```

2. Install the AMStools package:

   ```bash
   python setup.py install
   ```

3. Clone the `structure` repository
   from [here](https://git.noc.ruhr-uni-bochum.de/atomicclusterexpansion/data/structures)  to your local file
   system.

4. (Optional for `custodian` + `VASP`): Install [`pymatgen`](https://pymatgen.org/installation.html)
   and [`custodian`](http://materialsproject.github.io/custodian/).

5. (Optional for `ams_webinterface`): Install `flask` with `pip install`.

# General configuration

To configure AMStools, follow these steps:

1. Add the following environment variables to your `~/.bashrc` file:

```bash
   # ASE-related variables
   ## VASP
   export ASE_VASP_COMMAND=/path/to/bin/run_vasp_custodian.sh
   export VASP_PP_PATH=/path/to/VASP/VASP_PP/

   ## FHIaims (if planned to use)
   export ASE_AIMS_COMMAND=/path/to/bin/run_fhi.sh
   export AIMS_SPECIES_DIR=/path/to/fhi-aims/species_defaults/tight

   ### AMStools-related variables
   export AMS_STRUCTURE_REPOSITORY_PATH=/path/to/repo/structures
```

Examples of `run_vasp.sh` and `run_vasp_custodian.sh` can be found in this folder. If you do not want to use `custodian`
for VASP, then configure:

   ```bash
   export ASE_VASP_COMMAND=/path/to/bin/run_vasp.sh
   ```

Instead of using the `$AMS_STRUCTURE_REPOSITORY_PATH` environment variable, you can use
the `ams_highthroughput ... --repository-path /path/to/repo/structures` option.

Make sure that run_vasp_custodian.sh and/or run_vasp.sh contain the correct names of the loaded modules and
environments.

NOTE!!! Be sure, that  `run_vasp.sh` and `run_vasp_custodian.sh` has `+x` permission!!!

1. Configure the `~/.amstools` file to have different queues configuration. See the `tutorial/HT/.amstools` file for an example.

# Quick start

To run HT-DFT calculations, you need to pay attention to the following points:

1. HT setup: The HT setup is defined in a YAML file (e.g., `HTlist_example.yaml`) that specifies a list of structures
   and corresponding pipelines steps. You can use the provided examples or generate a template for the HT setup and
   calculator YAML files using `ams_highthroughput -t`.
2. Calculator: The calculator setup is defined in a YAML file (e.g., `vasp_tight.yaml` or `aims_tight.yaml`) that
   specifies a DFT calculator. Note that the calculator name (e.g., `name: VASP_PBE_500_0.125_0.1_NM`) in these files
   must be unique and determines the path for running calculations.
3. Run mode: By default, the calculations run locally. If you provide queue options (e.g., `-q zghlogin`
   or `-q vulcan_parallel16`), the calculations will be submitted to the queue. You can use the `-w` option for worker
   mode, which can be combined with queue mode. See the [Run Modes](#run-modes) section for more details.
4. HT-DFT root folder: This is the folder where all calculations will be stored. Either navigate to the corresponding
   folder and run `ams_highthroughput` there or use `-wd /path/to/my/calculations/root/folder` option.

Examples:

* Run **locally** a list of calculations from `HTlist_example.yaml` using `vasp_tight.yaml` calculator in the current
  folder:

```bash
ams_highthroughput ht_Al.yaml -c vasp_light.yaml 
```

* Submit to the queue `zghlogin` a list of calculations from `HTlist_example.yaml` using `vasp_tight.yaml` calculator
  in the current folder:

```bash
ams_highthroughput HTlist_example.yaml -c vasp_tight.yaml -q zghlogin
```

* Write only input files for DFT code

```bash
ams_highthroughput HTlist_example.yaml -c vasp_tight.yaml -i 
```

> NOTE! Default working folder is the current working dir, where you run `ams_highthroughput` command. You can change
> it with `-wd` option, i.e. `ams_highthroughput HTlist_example.yaml -c vasp_tight.yaml -wd /path/to/my/calculations/root/folder`

You can use `ams_highthroughput -t` to generate a template for high-throughput setup and calculator
YAML files.

## Dry Run

Use the `-dr` or `--dry-run` option with your `ams_highthroughput` command to see the list of jobs to be performed
**without** submitting any jobs. Add the `--verbose` flag to see more details.

# HT setup

HT setup YAML file starts with general configuration:

```yaml
## random seed
seed: 42
## Global permutation type that applies to all the prototypes or folders
## listed in the structures and pipelines section:
global_permutation_type: 'cfg'
## Upper bound on the possible number of permutations, if the number 
## of combinations is larger nsample will be selected randomly: 
nsample: 100
## Unordered list of chemical species: Al, Li, Nb, Ni, ...
composition: [ Mo, C ]
```

Then list prototypes/structures with corresponding list of pipeline steps and extra options are following:

* Option 1: the provided path is a cfg file

```yaml
/binaries/bulk/SZn_Wurtzite-B4/reference/gen_232_B4_SZn-Wurtzite.cfg:
  - relax
  - murnaghan
  - elastic

```

* Option 2: the provided path is a directory

```yaml
/ternaries/bulk/ABO3/reference:
  - random: 10    # number of cfg files to randomly select in the specified folder 
  - relax
  - murnaghan
  - elastic
```

* Option 3: 'constrained' mode where random combinations are chosen among the possible configurations:

```yaml
/binary/bulk/AB/reference:
  - { composition: { EleA: [ Mo, Ir, W ], EleB: [ C ] } , random: 2 }
  - random: 10
  - relax
  - murnaghan
  - elastic
```

* Option 4: Using exact structures (and generalized .cfg prototypes)

```yaml
CUSTOM/NiNb_gen133/*.cfg: [ static, relax ]
```

The type of permutation for the atomic species can be chosen among
the possible options: ```'cfg'```, ```'atoms'```, ```'wyckoff'```, ```'constrained'```, ```'mixed'```
(default is ```'cfg'```).

> NOTE! `ams_highthroughput` first try to read structure file with `ase.io.read` and if it fails, it will try to read it
> as a generalized `cfg` file with prototype

## Permutation types

The type of permutation for the atomic species can be chosen among
the possible options: `cfg`, `atoms`, `wyckoff`, `constrained`, `mixed`
(default is `cfg`).

The ```'cfg'``` is the representation in terms of EleA, EleB, ...
and ```'wyckoff'``` is the mapping of the ```'cfg'``` to a representation in terms of
Wyckoff sites. The 'atoms' mode is given by all the possible
permutations. The ```'mixed'``` option is obtained by mixing the ```'atoms'```
with the ```'wyckoff'``` representation, therefore in this case we get rid of
symmetry equivalent configurations corresponding to permutations of atoms at
the same Wyckoff sites. The ```'constrained'``` mode allows to specifiy a
The `cfg` is the representation in terms of EleA, EleB, ...
and `wyckoff` is the mapping of the `cfg` to a representation in terms of
Wyckoff sites. The `atoms` mode is given by all the possible
permutations. The `mixed` option is obtained by mixing the `atoms`
with the `wyckoff` representation, therefore in this case we get rid of
symmetry equivalent configurations corresponding to permutations of atoms at
the same Wyckoff sites. The `constrained` mode allows to specifiy a
list of species for each EleA, EleB, .... and select the possible
combinations.

Example: given a binary {EleA:1, EleB:2} of Wyckoff type AB2 and
considering 'Al' and 'Ni' we would have:

1) 'cfg':
   ```[['Al', 'Ni', 'Ni'], ['Ni', 'Al', 'Al']]```
2) 'wyckoff':
   ```[['Al', 'Ni', 'Ni'], ['Ni', 'Al', 'Al']]```
3) 'atoms':
   ```[['Al', 'Al', 'Ni'], ['Al', 'Ni', 'Al'], ['Al', 'Ni', 'Ni'], ['Ni', 'Al', 'Al'], ['Ni', 'Al', 'Ni'], ['Ni', 'Ni', 'Al']]```
4) 'mixed':
   ```[['Al', 'Ni', 'Al'], ['Al', 'Ni', 'Ni'] ,['Ni', 'Al', 'Al'], ['Ni', 'Al', 'Ni']]```

In the ```'constrained'``` option one needs to specify for each structure below a
dictionary of the possible EleA, EleB, EleC, .... considering a ternary
identified by:
In the `constrained` option one needs to specify for each structure below a dictionary of the possible EleA, EleB,
EleC, .... considering a ternary identified by:

`{ composition: {EleA: ['Ba','Ca','Sr'], EleB:['Ti','Sn'], EleC: ['O']} , random: 3}`
we will
get: `[EleA, EleB, EleC] = [['Ba', 'Ti', 'O'], ['Ba', 'Sn', 'O'], ['Ca', 'Ti', 'O'], ['Ca', 'Sn', 'O'], ['Sr', 'Ti', 'O'], ['Sr', 'Sn', 'O']]`

we would
have: ```[EleA, EleB, EleC] = [['Ba', 'Ti', 'O'], ['Ba', 'Sn', 'O'], ['Ca', 'Ti', 'O'], ['Ca', 'Sn', 'O'], ['Sr', 'Ti', 'O'], ['Sr', 'Sn', 'O']]```

## Pipeline steps

The available pipeline steps are:

* `static` - static calculation

Default settings for Enn calculations (lists will be multiplied with estimated nearest neighbor distance for
element/compound):

* `relax` - full relaxation for periodic structures and atomic-only for non-periodic), `relax-atomic`, `relax-full` -
  exact relaxation type.
  Full relaxation will be done in several stages, until relative change of volume is below certain threshold.

* `Enn-coarse`, `Enn-fine`, `Enn-local`, `Enn-local-few` - energy-nearest neighbor distance (ENN) curves calculations.
  Default settings for ENN calculations, in a format \[start:stop:step\]. Lists will be multiplied with estimated
  nearest neighbor distance for element/compound:

```
    nn_list_coarse = [0.600 : 1.600 : 0.05] + [1.700 : 2.000 : 0.1] + [2.200 : 3.000 : 0.2]
    nn_list_fine = [0.600 : 0.960 : 0.005] + [0.961 : 1.050 : 0.001]  +[1.055 : 1.100 : 0.005] + [1.110 : 1.350 : 0.01] + [1.400 : 2.400 : 0.05] + [2.500 : 3.000 : 0.1]
    nn_list_few = [0.980, 1.000, 1.020]
    nn_list_local = [0.88 : 1.12 : 0.02]
```

* `murnaghan` - energy-volume (EV) calculations, aka. Murnaghan equation-of-state (EOS). Default
  options: `num_of_point=11, volume_range=0.1, fit_order=5`
* `elastic` - elastic matrix calculation. Default options: `eps_range=0.015, num_of_point=5`
* `phonons` - phonon dispersion calculations.
* `tp_hexagonal`, `tp_orthogonal`, `tp_tetrag`, `tp_trigonal`, `tp_general_cubic_tetragonal` - transofrmation paths
  calculations. Valid only for FCC/BCC and cubic (tp_general_cubic_tetaragonal) crystal structures.
  Default options: `num_of_point=50`
* `randomdeformation` - several samples of randomly deformed structure, with E-V calculation for each sample. Default
  options: `nsample=3, random_atom_displacement=0.1, random_cell_strain=0.05, volume_range=0.05, num_volume_deformations=5,seed=42`
* `stacking_fault` - stacking faults (valid only for FCC).
* `defectformation` - defect formation energy (only `vacancy` type is supported). Default
  options: `interaction_range=10., defect_type="vacancy"`

For each pipeline steps it is possible to specify the calculator settings and permutation type, i.e.

```yaml
- Enn-coarse: { nn_distance_range: [ 2.0,  7.0 ], nn_distance_step: 0.5, fix_kmesh: False }
- Enn-fine: { nn_distance_range: [ 2.0,  7.0 ], nn_distance_step: 0.15, fix_kmesh: False }
- Enn-local: { nn_distance_range: [ 2.0,  2.1 ], nn_distance_step: 0.05, fix_kmesh: True }
- elastic: { eps_range: 0.015, num_of_point: 5 }
- murnaghan: { num_of_point: 11, volume_range: 0.1, fit_order: 5 }
- permutation_type: 'cfg'
```

# Calculator setup

Calculator setup file includes:

* name of the calculator. This name must be unique, since it will be top-level folder for all calcualtions.
* calculator type. Currently only `vasp` or `FHIaims` are supported.
* related DFT settings, that will be forwarded to ASE's calculator of given type.

Example:

```yaml
# unique name of the DFT configuration (500 eV, kspacing 0.125, gaussian 0.1, non-magnetic)
name: VASP_PBE_500_0.125_0.1_NM
# DFT calculator type
calculator: VASP
# related DFT settings
xc: pbe
gamma: True
setups: recommended
prec: Accurate
ediff: 1.0e-6
encut: 500
ispin: 0
nelm: 120
lreal: False
lcharg: False
lwave: False
ismear: 0
sigma: 0.1
kmesh_spacing: 0.125
```

# Run modes

## Local mode

`ams_highthroughput HTlist_example.yaml -c vasp_tight.yaml`

In this mode, pipelines will be run one after another, however if DFT code is configured with mpirun (f.e.
in `run_vasp.sh`),
then it will be running on all CPUs.

## Cluster queue mode

From cluster login node run:

`ams_highthroughput HTlist_example.yaml -c vasp_tight.yaml -q zghlogin`

where `zghlogin` is queue settings (from `~/.amstools` file).
Use `-d` or `--daemon` option to run ams_highthroughput on login node in infinite loop, when it will check and
update the statuses of running pipelines.

## Cluster worker mode

In this mode you could submit several `ams_highthroughput` workers into the queues and these jobs will process the
common
HT pipelines (f.e. defined in `HTlist_example.yaml`) but avoid duplicated calculations on the same pipeline (
synchronization is done via lock files written to shared network file system).

You could automatically submit `N` workers with the command:

`ams_highthroughput  HTlist_example.yaml -c vasp_tight.yaml -w N -q zghlogin`

that will generate the script `ams_pipeline_job.sh` and submit it to the `zghlogin` queue `N` times.

## Write-input-only mode

This mode runs locally and requires the `-i` or `--write-input-only` option:

`ams_highthroughput HTlist_example.yaml -c vasp_tight.yaml -i`

In this mode, `ams_highthroughput` will execute pipelines as in *local* mode. However, instead of running DFT calculations,
it will only generate input files for the DFT code (currently, only VASP is supported) with an additional `paused` lock file in
each DFT folder, and change the status of the pipeline to `paused`. The DFT calculation should be performed manually (locally
or remotely), and the resulting files should be copied back to the DFT folder. After that, `ams_highthroughput` can be run
again to push the pipeline to the next step.

# Registry database (state_dict.db)

To avoid scanning over the whole directory tree every time, `ams_highthroughput` creates a registry SQLite database in
the `state_dict.db` file (or in file provided by `--state-dict-file` option) and check the status of the pipeline there
before reading it from the disk. This database contains a list of all pipelines, identified by unique names, their
statuses and some other information and is updated automatically by `ams_highthroughput`.

If you want to rebuild the database:

* Use `--rebuild-db` option to rebuild the registry database from the scratch.
  The database will be created starting from the current working folder (`-wd` option) and all subfolders will be
  scanned. If no file name is provided, then default `rebuild_state_dict.db` will be used.
* Use extra `--rebuild-include` and `--rebuild-exclude` options to include or exclude some patterns in folder names.
* You can provide multiple options, separated by space, i.e. `--rebuild-include folder1 folder2`.
* If included folder ends with `/`, then ONLY corresponding folder in working directory will be included,
  i.e. `--rebuild-include VASP_PBE_500_0.125_0.1_NM/Al/` will include all subfolders
  of `/path/to/working/dir/VASP_PBE_500_0.125_0.1_NM/Al/`.
* You can include several top-level
  subfolders `--rebuild-include VASP_PBE_500_0.125_0.1_NM/Al/ VASP_PBE_500_0.125_0.1_NM/Ni/`.

> NOTE! If you want to rebuild the database, only in certain sub-folder of working directory, you still need to specify
> top working directory with `-wd /path/to/HTroot` and use `--rebuild-include VASP_PBE_500_0.125_0.1_NM/Al/` with `/` at
> the end.

# Re-run, persistence and idempotence

There are several levels of persistence in `ams_highthroughput`:

1. **Single DFT calculation** is stored into separate folder. If the calculation is already done, then minimal necessary
   information (structure, energy, free energy, forces, stresses) is extracted and stored in `data.json` file. Remaining
   files are store into archive, i.e. `vasp.tar.gz` file.
2. **Property (or pipeline step)** consists of one or more DFT calculations. If all calculations are done, then
   `property.json` file is created, that contains all necessary information about the property (or pipeline step),
   including postprocessing results.
3. **Pipeline** consists of one or more properties (or pipeline steps). It is stored in separate folder
   in `pipeline.json` that contains all necessary information about the pipeline (including all properties). A relative
   path, consisting of the calculator name, prototype name, and structure name, is used as a *unique* identifier of the
   pipeline.
4. **State dict** file contains a list of pipelines names and their statuses (finished, error, submitted to the queue,
   calculating, etc.). By default, it is stored in `state_dict.db` file.

`ams_highthroughput` avoids running the same calculation multiple times (*idempotence*). It checks the statuses from
top (state dict)
to bottom (single DFT calculation) levels and if the calculation is already done, then it will not be re-run but only
update the status. Separate pipelines (executed by `ams_highthroughput` or submitted by it to the queue)
are responsible for the first three levels of persistence (from top to bottom level), while the last one is handled
by `ams_highthroughput` itself.

> NOTE! If you run `ams_highthroughput` again with the same HT setup, calculator and queue settings, by default it will
> update pipeline statuses. It will not re-run the pipeline if it is in a finished, partially finished, running, waiting
> in the queue, or error state, but it will try to run pipelines again in all other states.

If you want to re-run calculations, then you can use some (or all) of the following options:

* Use `--rerun-error-pipelines` (or `-rep`) option to rerun pipelines in error state
* Use `--rerun-partially-finished-pipelines` (or `-rpfp`) option to rerun pipelines in partially finished state

To update the status of the pipelines (i.e. syncronize it with the actual state of the files on the disk):

* Use `--update-pipeline-statuses` (or `-u`) option to update all pipeline statuses

### Other options

* Use `--reset-pipeline-lock` option to remove 'pipeline.lock' file in 'worker' mode.
* (CAUTION!) Use `--cancel-all-jobs` option to cancel all jobs in the queue, given with `-q` option

# Migration of the HT-DFT campaign data

Only `data.json`, `property.json`, `pipeline.json` files,  `vasp.tar.gz` archives and subfolder structure where they are
stored are important results of the HT-DFT campaign. `state_dict.db` file is not important and can be reconstructed by
running `ams_highthroughput` in normal mode or by `--rebuild-db` option.
If you want to migrate the data from one folder to another, you can use, i.e. `rsync` command:

```bash
rsync -avz --prune-empty-dirs --include="*/" --include="data.json" --include="pipeline.json" --include="property.json" --include="vasp.tar.gz" --exclude="*" --info=progress2 -P  /source/HTroot/  /target/HTroot/
```

> Note!  Source argument `/source/HTroot/` has trailing slash `/`, means that content of this folder will be syncronized
> into /target/HTroot/.
>
> From rsync documentation:
>
> A trailing slash on a source path means "copy the contents of this directory". Without a trailing slash it means "copy
> the directory".

You also can add `rsync ... --dry-run` option to see what will be synchronized.

# `ams_highthroughput` utility

`ams_highthroughput` is the utility to run HT calculations. The available settings are:

```bash
usage: ams_highthroughput [-h] [-c CALCULATOR_SETUP_FNAME] [-wd WORKING_DIR] [-q [QUEUE_NAME]] [-i] [-p PERMUTATION_TYPE] [-r REPOSITORY_PATH] [-dr] [-u] [-rep] [-rpfp] [--cancel-all-jobs] [--worker-mode] [-w WORKERS] [--split-worker-output-files] [-d] [--time SLEEPING_TIME]
                          [--reset-pipeline-lock] [--state-dict-file STATE_DICT_FNAME] [--rebuild-state-dict [REBUILD_STATE_DICT]] [--rebuild-include REBUILD_INCLUDE [REBUILD_INCLUDE ...]] [--rebuild-exclude REBUILD_EXCLUDE [REBUILD_EXCLUDE ...]] [-l LOG] [-t [TEMPLATE]]
                          [--verbose] [-v]
                          ht_yaml_filename

AMS high-throughput (HT) utility. version: -0.post0.dev459-py3.8.egg. Possible pipeline steps names: Enn-coarse,Enn-fine,Enn-local,Enn-local-few,murnaghan,elastic,relax,relax-atomic,relax-
full,phonons,tp_tetrag,tp_orthogonal,tp_trigonal,tp_hexagonal,tp_general_cubic_tetragonal,defectformation,static,stacking_fault,randomdeformation

positional arguments:
  ht_yaml_filename      HT YAML filename.

optional arguments:
  -h, --help            show this help message and exit
  -c CALCULATOR_SETUP_FNAME, --calculator CALCULATOR_SETUP_FNAME
                        YAML file with calculator setup
  -wd WORKING_DIR, --working-dir WORKING_DIR
                        top directory where keep and execute HT calculations
  -q [QUEUE_NAME], --queue-name [QUEUE_NAME]
                        submit calculations to the queue. Options: <QUEUE_NAME> (~/.amstools::queues[QUEUE_NAME]). Default is first queue from ~/.amstools::queues (if exists)
  -i, --write-input-only
                        Write input files only. Default is False.
  -p PERMUTATION_TYPE, --permutation-type PERMUTATION_TYPE
                        Global permutation type: atoms, mixed, wyckoff, cfg, constrained. Default: cfg
  -r REPOSITORY_PATH, --repository-path REPOSITORY_PATH
                        structure repository path. Default is env variable AMS_STRUCTURE_REPOSITORY_PATH=/home/users/lysogy36/tools/structures
  -dr, --dry-run        Dry run: generate combinations but do not run calculations. Default is False.
  -u, --update-pipeline-statuses
                        (Force) Update all pipeline statuses. Default is False.
  -rep, --rerun-error-pipelines
                        Rerun pipelines in ERROR state. Default is False.
  -rpfp, --rerun-partially-finished-pipelines
                        Rerun pipelines in ERROR state. Default is False.
  --cancel-all-jobs     (DANGER!) will cancel all jobs in the given queue. Default is False.
  --worker-mode         run in worker mode. Synchronization is done via pipeline.lock file for each pipeline. Default is False.
  -w WORKERS, --workers WORKERS
                        Submit several workers mode. Use -q option to submit them to the queue. Default is 0.
  --split-worker-output-files
                        Option shows if different workers should write log and state dict in files with unique names of worker host. Default is False.
  -d, --daemon          Running infinitely until all pipelines are finished or failed. Default is False.
  --time SLEEPING_TIME  Sleeping time in seconds for daemon mode. Default is 10
  --reset-pipeline-lock
                        Force to remove pipeline.lock file in 'worker' mode. Default is False
  --state-dict-file STATE_DICT_FNAME
                        file with calculations state: .db for SQLite3 (default: state_dict.db)
  --rebuild-state-dict [REBUILD_STATE_DICT]
                        Rebuild state dict from scratch. If provided, then default is 'rebuild_state_dict.db'
  --rebuild-include REBUILD_INCLUDE [REBUILD_INCLUDE ...]
                        List of folders (space separated) to include when rebuilding state dict. If folder ends with '/', then only this subfolder will be scanned
  --rebuild-exclude REBUILD_EXCLUDE [REBUILD_EXCLUDE ...]
                        List of folders to exclude when rebuilding state dict
  -l LOG, --log LOG     log filename, Default is log.txt
  -t [TEMPLATE], --template [TEMPLATE]
                        Generate template for ht.yaml and calculator.yaml, provide calculator type (vasp or aims)
  --verbose             Verbose detailed output. Default is None
  -v, --version         Show version info
```

# FAQ

## Pipelines

Q: How to organize my HT-DFT calculations?<br>
A: It is better always to do HT-DFT in the same top working directory. Then you will benefit from the registry database
and avoid duplicate calculations automatically.

Q: I want to re-run some pipelines, how to do it?<br>
A: Manually remove `pipeline.lock` files in the corresponding pipeline folders and update pipeline statuses
(by re-running `ams_highthroughput` with `-u` option). This will change pipelines statuses to `non-existing`.
After that run `ams_highthroughput` again and all non-finished pipelines will be re-run. If pipeline's steps are
finished, then they will be just collected again.

Q: I want to re-run only few pipelines and not the whole list, how to do it?<br>
A: Create a new HT-DFT YAML file where you specify only those pipelines you want to re-run. Then follow the previous
answer.

Q: I want to start new HT-DFT campaign and some of the calculations are already done (by me or another person) on
another cluster. How to avoid duplications?<br>
A: You should copy registry file `state_dict.db` from the previous campaign. You can use automatically constructed one
or use `--rebuild-db` option to rebuild it (you can fine tune it using  `--rebuild-include` and `--rebuild-exclude`, see
above).

## Prototypes and structures

Q: How to organize prototypes and structures?<br>
A: It is better to have all prototypes (from AMS's `structures` repository) in the same folder. This location is
specified by
`-r`  or `--repository-path` option. Default is taken from environment variable `AMS_STRUCTURE_REPOSITORY_PATH`.

Q: I want to use my own exact structures in ASE readable [format](https://wiki.fysik.dtu.dk/ase/ase/io/io.html) and not
prototypes, how to do it?<br>
A: Just put them in some folder, better in the subfolder of top-folder of the structure repository and use them in
HT-DFT YAML file. `ams_highthroughput` will try first load the structure file as normal, and only then consider it as
general prototype.

# Collect data for parameterization

Use `ams_collect` in the topmost directory from which you would like to collect data. See `ams_collect -h` for a short
help. It will collect all `data.json` files from all subdirectories, that contain results of successful DFT
calculations and put them as pandas DataFrame in `collected.pckl.gzip` file.

> NOTE! `ams_collect` will collect **all** successful DFT calculations, does not matter in fully or partially finished
> steps/pipelines.

# (experimental feature) Webinterface

`ams_webinterface` is the utility to run the web interface at a given port:

```
usage: ams_ht_webinterface [-h] [-p PORT] [--host HOST] [-wd WORKING_DIR]

AMS HT webinterface

optional arguments:
  -h, --help            show this help message and exit
  -p PORT, --port PORT  port
  --host HOST           hostname
  -wd WORKING_DIR, --working-dir WORKING_DIR
                        working dir

```

To access the web page, use the link: ```http://hostname.rub.icams.de:port```, where ```hostname``` can
be ```vulcan``` or ```zghlogin```.

> NOTE! This feature works only if the port is open on the cluster.
