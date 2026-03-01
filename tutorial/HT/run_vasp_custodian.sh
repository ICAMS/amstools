#!/bin/bash

source ~/.bashrc
module load vasp/mpi/5.4.4
conda activate ams

ams_custodian_vasp mpirun vasp_std