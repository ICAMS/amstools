import argparse
import logging
import pandas as pd
import os
import subprocess
import sys
import shutil
import numpy as np
import ruamel

from subprocess import Popen, PIPE

from ase.db import connect
from fabric.connection import Connection
from ruamel import yaml

from amstools.calculators.dft.vasp import AMSVasp

from pyace import *
from pyace.activelearning import (
    compute_active_set_by_batches,
    compute_active_set,
    compute_B_projections,
)
from pyace.activelearning import read_extrapolation_data

LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
logging.basicConfig(level=logging.INFO, format=LOG_FMT)
logger = logging.getLogger()

fileh = logging.FileHandler("al_log.txt", "a")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
fileh.setFormatter(formatter)
logger.addHandler(fileh)

# remote host connection setting from ~/.amstools
REMOTE_HOST = "zghlogin"

# prefix for local bash commands
command_prefix = """
/bin/bash $HOME/.bashrc
source activate ace
"""

remote_cmd_prefix = """/bin/bash $HOME/.bashrc
source activate ace"""

# lmp_executable = "mpirun -np 3 /home/users/lysogy36/CLionProjects/lammps-ace/build_al/lmp"
lmp_executable = (
    "mpirun -np 3 /home/users/lysogy36/CLionProjects/lammps-ace/build_al_glatom/lmp"
)
# prepare input files
lammps_in_template_fname = "in.lammps.template"
initial_structure_filename = "water_dens.lammps-data"

# REMOTE PROJECT PATH, GLOBAL PARAMETER
project_prefix = "auto_al/water"

collected_df_fname = "collected_df.pckl.gzip"
calc_filename = "calculator.json"


# Criteria for fitting is done: >1000 steps or output_potential.yaml
def get_fitted_potential(path, min_num_steps=800):
    """
    Check if final potential exists,
    otherwise check if `min_num_steps` is done and
    """
    output_potential_fname = os.path.join(path, "output_potential.yaml")
    if os.path.isfile(output_potential_fname):
        return output_potential_fname
    else:
        metrics_fname = os.path.join(path, "metrics.txt")
        if os.path.isfile(metrics_fname):
            metrics_df = pd.read_csv(metrics_fname, sep="\s+")
            if len(metrics_df) > 0:
                last_row = metrics_df.iloc[-1]
                iter_num = last_row["iter_num"]
                if iter_num >= min_num_steps:
                    cycle_step = int(last_row["cycle_step"])
                    interim_file_name = os.path.join(
                        path, "interim_potential_{}.yaml".format(cycle_step)
                    )
                    if os.path.isfile(interim_file_name):
                        return interim_file_name


def run_bash_command(cmd, path):
    if not cmd.startswith(command_prefix):
        cmd = command_prefix + cmd
    logger.info("Running cmd: {}".format(cmd))
    logger.info("in {}".format(path))
    out = Popen(
        cmd,
        shell=True,
        stdin=PIPE,
        stdout=PIPE,
        stderr=subprocess.STDOUT,
        cwd=path,
        executable="/bin/bash",
    )
    print("STDOUT:")
    for c in iter(lambda: out.stdout.read(1), b""):
        print(c.decode("utf-8", "ignore"), end="")


def compute_asi(current_gen_path, current_potential_filename):
    fit_data_fname = os.path.join(current_gen_path, "fitting_data_info.pckl.gzip")
    generation_potential_fname = os.path.join(
        current_gen_path, "generation_potential.yaml"
    )

    shutil.copyfile(current_potential_filename, generation_potential_fname)

    fit_data_fname = os.path.abspath(fit_data_fname)
    generation_potential_fname = os.path.abspath(generation_potential_fname)
    asi_potential_fname = generation_potential_fname.replace(".yaml", ".asi")

    if not os.path.isfile(asi_potential_fname):
        cmd = "pace_activeset -d {} {}".format(
            fit_data_fname, generation_potential_fname
        )
        cmd = command_prefix + "\n" + cmd
        run_bash_command(cmd, current_gen_path)

        assert os.path.isfile(asi_potential_fname), "Could not find file {}".format(
            asi_potential_fname
        )
    else:
        print("ASI file already exists: ", asi_potential_fname)
    return generation_potential_fname, asi_potential_fname, fit_data_fname


def run_remote_ams_dft_manager(
    remote_calc_json_fname, remote_directory, remote_db_filename, c
):
    remote_commands = remote_cmd_prefix.split("\n") + [
        "cd " + remote_directory,
        "ams_dft_manager -c {} -d {} -p {}".format(
            remote_calc_json_fname, remote_db_filename, remote_directory
        ),
    ]

    remote_cmd = "&&".join(remote_commands)

    c.run(remote_cmd)


def connect_to_cluster(remote_connection_options):
    c = Connection(**remote_connection_options.get("connection"))
    s = c.sftp()
    return c, s


def get_remote_path(c, s, remote_project_current_path):
    """
    Get remote path = remote root path + current calculations's working directory
    :param c: fabric.Connection
    :param s: fabric.sftp
    :return: remote_directory:str
    """
    if hasattr(s, "remote_project_root_path"):
        remote_project_root_path = s.remote_project_root_path
    else:
        remote_project_root_path = c.run(
            "echo $AMS_REMOTE_CALC_PATH", hide=True
        ).stdout.strip()
        if len(remote_project_root_path) == 0:
            remote_project_root_path = "${HOME}/ams_remote_calc"
        s.remote_project_root_path = remote_project_root_path
    logger.info("remote_project_root_path = {}".format(remote_project_root_path))
    s.chdir(remote_project_root_path)
    remote_directory = os.path.join(
        remote_project_root_path, remote_project_current_path
    )
    return remote_directory


# from https://stackoverflow.com/questions/14819681/upload-files-using-sftp-in-python-but-create-directories-if-path-doesnt-exist
def mkdir_p(sftp, remote_directory):
    """Change to this directory, recursively making new folders if needed.
    Returns True if any folders were created."""
    if remote_directory == "/":
        # absolute path so change directory to root
        sftp.chdir("/")
        return
    if remote_directory == "":
        # top-level relative directory must exist
        return
    try:
        sftp.chdir(remote_directory)  # sub-directory exists
        logger.info("Remote chdir {}".format(remote_directory))
    except IOError:
        dirname, basename = os.path.split(remote_directory.rstrip("/"))
        mkdir_p(sftp, dirname)  # make parent directories
        sftp.mkdir(basename)  # sub-directory missing, so created it
        logger.info("Remote mkdir {}".format(basename))
        sftp.chdir(basename)
        logger.info("Remote chdir {}".format(basename))
        return True


def setup_remote_calculator(c, s):
    calc = AMSVasp(
        xc="pbe",
        gga="RE",  # revPBE
        setups="recommended",
        prec="Accurate",
        ediff=1e-8,
        encut=400,  # 400 eV cutoff
        ispin=0,
        nelm=120,
        nelmin=4,
        lreal=False,
        lcharg=False,
        lwave=False,
        addgrid=True,
        ncore=4,
        ismear=0,  # Gaussian smearing
        sigma=0.01,
        kpts=[1, 1, 1],
        kmesh_spacing=None,
    )
    calc.set(ivdw=11)

    calc.save("./" + calc_filename)

    remote_project_directory = get_remote_path(c, s, project_prefix)
    mkdir_p(s, remote_project_directory)
    remote_calc_json_fname = os.path.join(remote_project_directory, "calculator.json")
    s.put(calc_filename, remote_calc_json_fname)
    return remote_calc_json_fname


def check_remote_calculations(
    remote_directory,
    remote_collected_df_filename,
    current_collected_df_fname,
    collected_df_fname,
    s,
):
    s.chdir(remote_directory)
    dirlist = s.listdir()
    if collected_df_fname in dirlist:
        logger.info(
            "Remote DFT collected data file {} is found in remote directory {}".format(
                collected_df_fname, remote_directory
            )
        )
        logger.info(
            "Get {} from remote location {}".format(
                current_collected_df_fname, remote_collected_df_filename
            )
        )
        s.get(remote_collected_df_filename, current_collected_df_fname)
    else:
        raise RuntimeError(
            "Remote `finished` flag is found, but no collected data found"
        )


def main(args):
    parser = argparse.ArgumentParser(
        prog="pace_activelearning", description="PACE active learning master process"
    )

    parser.add_argument(
        "path", help="root path for collecting", nargs="?", type=str, default="."
    )

    parser.add_argument(
        "-i",
        "--initial-gen",
        help="initial generation index",
        dest="initial_generation_index",
        default=0,
    )

    parser.add_argument(
        "-m",
        "--max-gen",
        help="maximum generation index",
        dest="max_generation_index",
        default=20,
    )

    args_parse = parser.parse_args(args)

    initial_generation_index = int(args_parse.initial_generation_index)
    max_generation_index = int(args_parse.max_generation_index)

    logger.info(
        "Starting pace_activelearning. Initial gen: {}, max gen: {}".format(
            initial_generation_index, max_generation_index
        )
    )
    # initial
    current_generation_index = initial_generation_index  # =0

    while current_generation_index <= max_generation_index:
        logger.info("=" * 40)
        logger.info("=" * 40)
        logger.info("Current generation: {}".format(current_generation_index))
        logger.info("=" * 40)
        logger.info("=" * 40)

        current_gen_path = "gen{}".format(current_generation_index)

        logger.info(
            "Gen: {} / Stage 1. Potential fitting".format(current_generation_index)
        )
        # check if fitting is done
        current_potential_filename = get_fitted_potential(current_gen_path)

        # if not
        if not current_potential_filename:
            # run pacemaker in current_gen_path
            logger.info("Run pacemaker in {}".format(current_gen_path))
            run_bash_command("pacemaker", current_gen_path)
            current_potential_filename = get_fitted_potential(current_gen_path)
        else:
            logger.info(
                "Current potential file is found at {}".format(
                    current_potential_filename
                )
            )

        # if current potential is fitted
        if current_potential_filename:
            logger.info(
                "Compute active set with `pace_activeset` utility for {}".format(
                    current_potential_filename
                )
            )
            generation_potential_fname, asi_potential_fname, fit_data_fname = (
                compute_asi(current_gen_path, current_potential_filename)
            )
        else:
            raise RuntimeError(
                "Could not find fitted potential in {}".format(current_gen_path)
            )

        # 2. run exploration lammps
        logger.info("-" * 40)
        logger.info(
            "Gen: {} / Stage 2. New structures exploration with LAMMPS".format(
                current_generation_index
            )
        )
        lammps_run_path = os.path.join(current_gen_path, "LAMMPS")
        if not os.path.isdir(lammps_run_path):
            os.makedirs(lammps_run_path)

        with open(lammps_in_template_fname) as f:
            in_lammps_template = "".join(f.readlines())

        initial_structure_filename = "water_dens.lammps-data"
        initial_structure_filename = os.path.abspath(initial_structure_filename)

        in_lammps = in_lammps_template.format(
            initial_structure=initial_structure_filename,
            gamma_lo=5,
            gamma_hi=10,
            pot_yaml=generation_potential_fname,
            pot_asi=asi_potential_fname,
        )
        lammps_run_path = os.path.abspath(lammps_run_path)

        in_lammps_filename = os.path.join(lammps_run_path, "in.lammps")

        with open(in_lammps_filename, "w") as f:
            print(in_lammps, file=f)

        lmp_commmand = "{} -in {}".format(lmp_executable, in_lammps_filename)

        extrapolation_structures_fname = os.path.join(
            lammps_run_path, "extrapolative_structures.dat"
        )

        if not os.path.isfile(extrapolation_structures_fname):
            logger.info("Run LAMMPS")
            run_bash_command(lmp_commmand, path=lammps_run_path)
        else:
            logger.info(
                "Extrapolation structures file already exists: {}".format(
                    extrapolation_structures_fname
                )
            )

        extrapolation_atoms = read_extrapolation_data(extrapolation_structures_fname)
        # TODO: check automatically the PBC
        for at in extrapolation_atoms:
            if np.any(at.cell > 0):
                at.set_pbc(True)
        num_extrapolative_structures = len(extrapolation_atoms)
        logger.info(
            "{} extrapolative structures are found in {}".format(
                num_extrapolative_structures, lammps_run_path
            )
        )

        ### compute active set (with previous fitting set)
        logger.info("-" * 40)
        logger.info(
            "Gen: {} / Stage 3. Selection of new structures into new active set".format(
                current_generation_index
            )
        )

        sel_struc_filename = "selected_structures_gen{}.db".format(
            current_generation_index
        )
        db_filename = os.path.join(current_gen_path, sel_struc_filename)

        if not os.path.isfile(db_filename):
            logger.info("Loading current training set from {}".format(fit_data_fname))
            fit_df = pd.read_pickle(fit_data_fname, compression="gzip")
            fit_df.reset_index(drop=True, inplace=True)
            max_ind_fit_df = fit_df.index.max()

            logger.info("Loading new extrapolative structures")
            ext_df = pd.DataFrame({"ase_atoms": extrapolation_atoms})

            logger.info("Concatenating datasets")
            mdf = pd.concat([fit_df, ext_df])
            mdf.reset_index(drop=True, inplace=True)

            logger.info("Total size: {}".format(len(mdf)))

            bconf = BBasisConfiguration(generation_potential_fname)

            # TODO: check the memlimit, do it optionally by batches
            logger.info(
                "Computing B-projections using potential from {}".format(
                    generation_potential_fname
                )
            )
            bpro_res = compute_B_projections(
                bconf, mdf["ase_atoms"], structure_ind_list=mdf.index
            )
            as_res = compute_active_set(*bpro_res)
            as_sel_inds = as_res[1]
            sel_inds_set = set()
            for k, v in as_sel_inds.items():
                sel_inds_set.update(v)
            sel_inds = np.array(list(sel_inds_set))
            new_sel_inds = sel_inds[sel_inds > max_ind_fit_df]
            selected_structures = mdf.loc[new_sel_inds, "ase_atoms"]
            logger.info(
                "{}/{} new structures were selected using MaxVol".format(
                    len(selected_structures), num_extrapolative_structures
                )
            )
            # pack structures
            with connect(db_filename) as db:
                for at in selected_structures:
                    db.write(at)
            logger.info(
                "Selected new structures were stored into {}".format(db_filename)
            )
        else:
            logger.info(
                "Active set selected is already stored in {}".format(db_filename)
            )

        # Submit to DFT
        # 1. Check locally if the collected_df_fname file is presented
        # 2. sftp to remote login node
        # 3. run remote utility for DFT submission (all cluster settings are responsibility of remote utility)
        logger.info("-" * 40)
        logger.info(
            "Gen: {} / Stage 4. Remote DFT calculations".format(
                current_generation_index
            )
        )

        current_collected_df_fname = os.path.join(current_gen_path, collected_df_fname)

        if os.path.isfile(current_collected_df_fname):
            logger.info(
                "DFT calculations for selected structures are already presented LOCALLY in {}".format(
                    current_collected_df_fname
                )
            )
            # do nothing
        else:
            logger.info("Setting up remote calculations")
            with open("/home/users/lysogy36/.amstools") as f:
                amstools_conf = yaml.load(f, Loader=ruamel.yaml.Loader)
            remote_connection_options = amstools_conf["remotes"][REMOTE_HOST]
            c, s = connect_to_cluster(remote_connection_options)
            remote_hostname = remote_connection_options["connection"]["host"]
            remote_project_current_path = os.path.join(project_prefix, current_gen_path)
            remote_directory = get_remote_path(c, s, remote_project_current_path)
            # do it only once
            remote_calc_json_fname = setup_remote_calculator(c, s)

            mkdir_p(s, remote_directory)
            s.chdir(remote_directory)
            dirlist = s.listdir()
            # check if the finished lock presented
            remote_db_filename = os.path.join(remote_directory, sel_struc_filename)
            remote_collected_df_filename = os.path.join(
                remote_directory, collected_df_fname
            )
            if "finished" not in dirlist:
                if sel_struc_filename not in dirlist:
                    logger.info(
                        "No {} found on remote host {}:{}, uploading".format(
                            sel_struc_filename, remote_hostname, remote_directory
                        )
                    )
                    s.put(db_filename, remote_db_filename)
                else:
                    logger.info(
                        "{} already is on remote host, skip uploading".format(
                            remote_db_filename
                        )
                    )

                # run calculations remotely
                run_remote_ams_dft_manager(
                    remote_calc_json_fname, remote_directory, remote_db_filename, c
                )
                check_remote_calculations(
                    remote_directory,
                    remote_collected_df_filename,
                    current_collected_df_fname,
                    collected_df_fname,
                    s,
                )
            else:
                logger.info(
                    "Remote 'finished' flag is found in remote directory {}".format(
                        remote_directory
                    )
                )
                check_remote_calculations(
                    remote_directory,
                    remote_collected_df_filename,
                    current_collected_df_fname,
                    collected_df_fname,
                    s,
                )

        # Prepare next generation
        next_generation_index = current_generation_index + 1
        next_gen_path = "gen{}".format(next_generation_index)
        logger.info("-" * 40)
        logger.info(
            "Gen: {} / Stage 5. Preparing next generation input files in {}".format(
                current_generation_index, next_gen_path
            )
        )

        new_dft_data = pd.read_pickle(current_collected_df_fname, compression="gzip")

        # TODO: do energy_corrected calculations
        new_dft_data["energy_corrected"] = new_dft_data["energy"]

        fit_data_fname = os.path.join(current_gen_path, "data.pckl.gzip")

        old_dft_data = pd.read_pickle(fit_data_fname, compression="gzip")
        tot_dft_data = pd.concat([old_dft_data, new_dft_data])

        tot_dft_data = tot_dft_data[
            ["ase_atoms", "results", "name", "energy", "energy_corrected", "forces"]
        ].reset_index(drop=True)

        os.makedirs(next_gen_path, exist_ok=True)

        tot_data_fname = os.path.join(next_gen_path, "data.pckl.gzip")
        tot_dft_data.to_pickle(tot_data_fname, compression="gzip")

        new_restart_potential_file = os.path.join(next_gen_path, "previous_gen.yaml")

        shutil.copyfile(current_potential_filename, new_restart_potential_file)

        old_input_yaml = os.path.join(current_gen_path, "input.yaml")
        new_input_yaml = os.path.join(next_gen_path, "input.yaml")

        shutil.copyfile(old_input_yaml, new_input_yaml)

        with open(new_input_yaml) as f:
            new_input_yaml_data = yaml.load(f, Loader=ruamel.yaml.Loader)

        new_input_yaml_data["potential"] = "previous_gen.yaml"

        with open(new_input_yaml, "w") as f:
            yaml.dump(new_input_yaml_data, f)

        current_generation_index += 1


if __name__ == "__main__":
    main(sys.argv[1:])
