#!/bin/bash

source ~/.bashrc
module load vasp/mpi/5.4.4
mpirun /cluster/vasp/5.4.4/mpi/bin/vasp_std
