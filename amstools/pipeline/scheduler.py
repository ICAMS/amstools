"""
This file originally was taken from

calphy: a Python library and command line interface for automated free
energy calculations.

Copyright 2021  (c) Sarath Menon^1, Yury Lysogorskiy^1, Ralf Drautz^1
^1: Ruhr-University Bochum, Bochum, Germany

More information about the program can be found in:
Menon, Sarath, Yury Lysogorskiy, Jutta Rogal, and Ralf Drautz.
“Automated Free Energy Calculation from Atomistic Simulations.”
ArXiv:2107.08980 [Cond-Mat], July 19, 2021.
http://arxiv.org/abs/2107.08980.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

See the LICENSE file.

For more information contact:
sarath.menon@ruhr-uni-bochum.de
"""

import subprocess as sub
import os
import stat
import getpass


class Local:
    """
    Local submission script
    """

    def __init__(self, options, cores=1, directory=None):
        if directory is None:
            directory = os.getcwd()
        self.queueoptions = {
            "scheduler": "local",
            "jobname": "ams_pipeline",
            "walltime": None,
            "queuename": None,
            "memory": None,
            "cores": cores,
            "hint": None,
            "directory": directory,
            "options": [],
            "commands": [],
            "modules": [],
            "header": "#!/bin/bash",
        }
        for key, val in options.items():
            if key in self.queueoptions.keys():
                if val is not None:
                    self.queueoptions[key] = val
        self.maincommand = ""

    def write_script(self, outfile):
        """
        Write the script file
        """
        jobout = ".".join([outfile, "out"])
        joberr = ".".join([outfile, "err"])

        with open(outfile, "w") as fout:
            fout.write(self.queueoptions["header"])
            fout.write("\n")

            # now write modules
            for module in self.queueoptions["modules"]:
                fout.write("module load %s\n" % module)

            # now finally commands
            for command in self.queueoptions["commands"]:
                fout.write("%s\n" % command)
            fout.write("%s > %s 2> %s\n" % (self.maincommand, jobout, joberr))
        self.script = outfile

    def submit(self):
        """
        Submit the job
        """
        st = os.stat(self.script)
        os.chmod(self.script, st.st_mode | stat.S_IEXEC)
        cmd = [self.script]
        proc = sub.Popen(cmd, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        return proc

    def get_job_id(self, submission_res):
        raise NotImplementedError()


class SLURM:
    """
    Slurm class for writing submission script
    """

    def __init__(self, options, cores=1, directory=None):
        """
        Create class
        """
        if directory is None:
            directory = os.getcwd()
        self.queueoptions = {
            "scheduler": "slurm",
            "jobname": "ams_pipeline",
            "walltime": "23:59:00",
            "queuename": "compute",
            "memory": None,
            "cores": cores,
            "hint": "nomultithread",
            "directory": directory,
            "options": [],
            "commands": [
                "uss=$(whoami)",
                "find /dev/shm/ -user $uss -type f -mmin +30 -delete",
            ],
            "modules": [],
            "header": "#!/bin/bash",
        }
        for key, val in options.items():
            if key in self.queueoptions.keys():
                if val is not None:
                    self.queueoptions[key] = val
        self.maincommand = ""

    def write_script(self, outfile):
        """
        Write the script file
        """
        jobout = ".".join([outfile, "out"])
        joberr = ".".join([outfile, "err"])

        with open(outfile, "w") as fout:
            fout.write(self.queueoptions["header"])
            fout.write("\n")

            # write the main header options
            fout.write("#SBATCH --job-name=%s\n" % self.queueoptions["jobname"])
            if self.queueoptions["walltime"]:
                fout.write("#SBATCH --time=%s\n" % self.queueoptions["walltime"])
            fout.write("#SBATCH --partition=%s\n" % self.queueoptions["queuename"])
            fout.write("#SBATCH --ntasks=%s\n" % str(self.queueoptions["cores"]))
            if self.queueoptions["memory"]:
                fout.write("#SBATCH --mem-per-cpu=%s\n" % self.queueoptions["memory"])
            fout.write("#SBATCH --hint=%s\n" % self.queueoptions["hint"])
            fout.write("#SBATCH --chdir=%s\n" % self.queueoptions["directory"])

            # now write extra options
            for option in self.queueoptions["options"]:
                fout.write("#SBATCH %s\n" % option)

            # now write modules
            for module in self.queueoptions["modules"]:
                fout.write("module load %s\n" % module)

            # now finally commands
            for command in self.queueoptions["commands"]:
                fout.write("%s\n" % command)
            fout.write('cd "{}"\n'.format(self.queueoptions["directory"]))
            fout.write('%s > "%s" 2> "%s"\n' % (self.maincommand, jobout, joberr))

        self.script = outfile

    def submit(self):
        """
        Submit the job
        """
        cmd = ["sbatch", self.script]
        proc = sub.Popen(cmd, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        return proc

    def get_job_id(self, submission_res):
        try:
            submission_res = submission_res[0]
            job_id = submission_res.split()[-1].decode("utf-8")
            # job_id = int(job_id)
            return job_id
        except Exception as e:
            print("Error during job submission:", e)


class SGE:
    """
    Slurm class for writing submission script
    """

    def __init__(self, options, cores=1, directory=None):
        """
        Create class
        """
        if directory is None:
            directory = os.getcwd()
        self.queueoptions = {
            "scheduler": "sge",
            "jobname": "ams_pipeline",
            "walltime": "23:59:00",
            "queuename": None,
            "memory": None,
            "system": "smp",
            "commands": [],
            "modules": [],
            "options": [
                "-j y",
                "-R y",
                "-P ams.p",
                "-S /bin/bash",
                "-o queue.out",
                "-e error.out",
            ],
            "cores": cores,
            "hint": None,
            "directory": directory,
            "header": "#!/bin/bash",
        }
        for key, val in options.items():
            if key in self.queueoptions.keys():
                if val is not None:
                    self.queueoptions[key] = val
        self.maincommand = ""

    def write_script(self, outfile):
        """
        Write the script file
        """
        jobout = ".".join([outfile, "out"])
        joberr = ".".join([outfile, "err"])

        with open(outfile, "w") as fout:
            fout.write(self.queueoptions["header"])
            fout.write("\n")

            # write the main header options
            fout.write("#$ -N %s\n" % self.queueoptions["jobname"])
            if self.queueoptions["walltime"]:
                fout.write("#$ -l h_rt=%s\n" % self.queueoptions["walltime"])
            fout.write("#$ -q %s\n" % self.queueoptions["queuename"])
            fout.write(
                "#$ -pe %s %s\n"
                % (self.queueoptions["system"], str(self.queueoptions["cores"]))
            )
            if self.queueoptions["memory"]:
                fout.write("#$ -l h_vmem=%s\n" % self.queueoptions["memory"])
            fout.write("#$ -wd %s\n" % self.queueoptions["directory"])

            # now write extra options
            for option in self.queueoptions["options"]:
                fout.write("#$ %s\n" % option)

            # now write modules
            for module in self.queueoptions["modules"]:
                fout.write("module load %s\n" % module)

            # now finally commands
            for command in self.queueoptions["commands"]:
                fout.write("%s\n" % command)

            fout.write('cd "{}"\n'.format(self.queueoptions["directory"]))
            fout.write('%s > "%s" 2> "%s"\n' % (self.maincommand, jobout, joberr))

        self.script = outfile

    def submit(self):
        """
        Submit the job
        """
        cmd = ["qsub", self.script]
        proc = sub.Popen(cmd, stdin=sub.PIPE, stdout=sub.PIPE, stderr=sub.PIPE)
        return proc

    def get_job_id(self, submission_res):
        try:
            submission_res = submission_res[0]
            job_id = submission_res.split()[2].decode("utf-8")
            # job_id = int(job_id)
            return job_id
        except Exception as e:
            print("Error during job submission:", e)


def get_current_running_job_ids(queue_options):
    scheduler_type = queue_options.get("scheduler")
    username = getpass.getuser()
    if scheduler_type == "slurm":
        proc = sub.Popen(["squeue", f"--user={username}"], stdout=sub.PIPE)
        lines = proc.stdout.readlines()
        job_ids = {}
        for l in lines[1:]:
            l = l.decode("utf-8").split()
            job_id = l[0]  # no int conversion
            job_ids[job_id] = {
                "partition": str(l[1]),
                "user": str(l[3]),
                "state": str(l[4]),
                "time": str(l[5]),
                "nodes": str(l[6]),
                "nodelist": str(l[7]),
            }

        return job_ids
    elif scheduler_type == "sge":
        proc = sub.Popen(["qstat"], stdout=sub.PIPE)
        lines = proc.stdout.readlines()
        job_ids = {}
        for l in lines[2:]:
            l = l.decode("utf-8").split()
            n_keys = len(l)
            job_id = l[0]  # no int conversion
            job_ids[job_id] = {
                "user": str(l[3]),
                "state": str(l[4]),
                "time": str(" ".join(l[5:7])),
                "slots": str(l[-1]),
            }
            if n_keys == 9:
                qstr = str(l[7]).split("@")
                job_ids[job_id]["queue"] = qstr[0]
                job_ids[job_id]["nodes"] = qstr[1]
        return job_ids
    else:
        raise NotImplementedError(
            "get_current_running_job_ids is not implemented for {}".format(
                scheduler_type
            )
        )


def cancel_queue_jobs(job_ids, queue_options):
    if not isinstance(job_ids, (list, tuple)):
        job_ids = [job_ids]
    scheduler_type = queue_options.get("scheduler")
    if scheduler_type == "slurm":
        for job_id in job_ids:
            sub.Popen(["scancel", "{}".format(job_id)])
    else:
        raise NotImplementedError(
            "cancel_queue_jobs is not implemented for {}".format(scheduler_type)
        )
