import os
import numpy as np

from ase import Atoms
from ase.build import fcc110, fcc111, fcc100, bulk, surface, make_supercell
from ase.io import read, write

import subprocess

from textwrap import dedent

from ase.data import atomic_numbers, atomic_masses


def write_in_lammps(
    structure,
    potname,
    working_dir,
    T=300,
    dt=None,
    with_extrapolation_grade=False,
    dump_extrapolative=False,
    is_gpu=False,
    steps=("min_full",),
    n_min_steps=500,
    n_eq_steps=10000,
    n_mdmc_steps=50000,
    minimize_thermo_freq=50,
    equilibration_thermo_freq=250,
    mdmc_thermo_freq=250,
    is_triclinic=False,
    aniso_style=None,
    specorder=None,
    gamma_lo=5,
    gamma_hi=15,
    check_pot_asi_file=True,
    pairstyle="pace",
):
    """
    Write input scripts for a LAMMPS simulation as a sequence of steps  for the provided structure and ACE potential.

    Note, that `working_dir` should be unique for given structure+list of steps.
    Use rerun=True, if you want to rerun the simulation in the `working_dir`

    Args:
        structure (ase.Atoms): The atomic structure for the simulation.
        potname (str): The name of the ACE interatomic potential file.
        working_dir (str): The working directory for the simulation.
            If the directory already contains output files from the specified steps
            and rerun is set to False, the simulation will not be re-run and the
            existing output files will be used.
        T (float, optional): The temperature (K) for the simulation. Defaults to 300.
        with_extrapolation_grade (bool, optional): Whether to compute ACE extrapolation grade. Defaults to False.
        dump_extrapolative (bool, optional): whether to dump extrapolative structures. Defaults to False.
        is_gpu (bool, optional): Whether to use GPU acceleration (pace/kk style). Defaults to False.
        lmp_exec (str or list, optional): The LAMMPS executable or a list containing the
            executable path and additional arguments. Defaults to "lmp".
        steps (list of dict, optional): A list of dictionaries specifying the simulation steps.
            Each dictionary should have the following keys:
                * type (str): The type of simulation step,
                    e.g. "min_full", "min_atomic", "min_iso", "min_aniso", "eq", "mdmc".
                * name (str, optional): The name of the step for output identification.
                    Defaults to the step type.
                * thermo_freq (int, optional): The frequency (in steps) for writing thermodynamic
                    data during the step. Defaults to the value specified by the corresponding
                    `n_*_thermo_freq` argument (e.g. `minimize_thermo_freq` for minimization steps).
                * Other keys (optional): Additional arguments specific to the step type
                    (e.g. `T_init`, `T1`, `T2`, `P1`, `P2` for equilibration).
        n_min_steps (int, optional): The number of steps for minimization step. Defaults to 2500.
        n_eq_steps (int, optional): The number of steps for equilibration step. Defaults to 10000.
        n_mdmc_steps (int, optional): The number of steps for the molecular dynamics+Monte Carlo  (MDMC) stage.
                  Defaults to 50000.
        minimize_thermo_freq (int, optional): The frequency (in steps) for writing thermodynamic
            data during minimization. Defaults to 50.
        equilibration_thermo_freq (int, optional): The frequency (in steps) for writing thermodynamic
            data during equilibration. Defaults to 250.
        mdmc_thermo_freq (int, optional): The frequency (in steps) for writing thermodynamic
            data during the MDMC. Defaults to 250.
        is_triclinic (bool, optional): Whether the simulation box is triclinic. Defaults to False.
                If structure is provided as ASE atoms, then this flag will be computed based on the structure itself.
        aniso_style (str, optional): The style for applying anisotropic pressure (e.g. "iso", "aniso", "tri").
            Defaults to None.
        specorder (list, optional): The species order for the simulation. If not provided, it will
            be inferred from the `structure` using `specorder_from_structure`.
        rerun (bool, optional): Whether to re-run the simulation even if the output files already
            exist in the working directory. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            * energy (float): The total energy data from the last simulation step.
            * structure (ase.Atoms): The atomic structure after the last simulation step.


    Examples:
        Here are examples of the `steps` parameter for different simulation protocols.
        You can copy and paste these into your Python code.

        .. code-block:: python

            # Example: Full relaxation (atomic positions and cell)
            steps = [
                {
                    "type": "min_full",
                    # "name": "my_full_relaxation",  # Optional: custom name for the step
                    "thermo_freq": 100,           # Optional: thermodynamic data output frequency
                    "steps": 1000,                # Optional: number of minimization steps
                    "aniso_style": "aniso",       # Optional: 'iso', 'aniso', or 'tri'
                },
            ]

            # Example: Atomic relaxation only
            steps = [{"type": "min_atomic", "steps": 500}]

            # Example: Isotropic cell relaxation only
            steps = [{"type": "min_iso", "steps": 500}]

            # Example: Anisotropic cell relaxation only
            steps = [{"type": "min_aniso", "steps": 500}]

            # Example: NPT equilibration
            steps = [
                {
                    "type": "eq",
                    # "name": "my_npt_equilibration",
                    "eq_type": "npt",             # 'npt', 'nvt', or 'nve'
                    "thermo_freq": 250,
                    "steps": 10000,
                    "T_init": 600,                # Initial temperature for velocity creation
                    "T1": 300, "T2": 300,          # Target temperature range
                    "P1": 0, "P2": 0,              # Target pressure range (for NPT)
                    "aniso_style": "iso",         # Pressure coupling style (for NPT)
                    "seed": 12345,                # Random seed for velocity creation
                    "dump": True,                 # Whether to dump trajectory
                },
            ]

            # Example: MDMC for atom swapping (e.g., for high-entropy alloys)
            steps = [
                {
                    "type": "mdmc",
                    # "name": "my_mdmc_swap",
                    "thermo_freq": 500,
                    "steps": 50000,
                    "T_MDMC": 1000,               # Temperature for Monte Carlo swap attempts
                    "T1": 1000, "T2": 1000,        # Temperature for MD (NPT)
                    "P1": 0, "P2": 0,              # Pressure for MD (NPT)
                    "aniso_style": "iso",
                    # "elements": ["Ni", "Co", "Cr"], # Elements to be swapped, by default - all possible pairs
                    "N": 100,                     # Perform M swap attempts every N steps
                    "M": 100,                     # Number of swap attempts to perform
                },
            ]

            # Example: Tensile deformation along 'z' with NPT on other axes
            steps = [
                {
                    "type": "deform_npt",
                    # "name": "my_tensile_test",
                    "thermo_freq": 100,
                    "steps": 20000,
                    "direction": "z",             # 'x', 'y', or 'z'
                    "erate": 0.001,               # Strain rate
                    "T1": 300, "T2": 300,          # Temperature for NPT
                },
            ]

            # Example: Tensile deformation along 'z' with NVT on other axes
            steps = [
                {
                    "type": "deform_nvt",
                    # "name": "my_tensile_test",
                    "thermo_freq": 100,
                    "steps": 20000,
                    "direction": "z",             # 'x', 'y', or 'z'
                    "erate": 0.001,               # Strain rate
                    "T1": 300, "T2": 300,          # Temperature for NPT
                },
            ]

            # Example: Shear deformation in 'xz' plane with NVT
            steps = [
                {
                    "type": "shear_nvt",
                    # "name": "my_shear_test",
                    "thermo_freq": 100,
                    "steps": 20000,
                    "direction": "xz",            # 'xy', 'xz', or 'yz'
                    "erate": 0.001,               # Strain rate
                    "T1": 300, "T2": 300,          # Temperature for NVT
                },
            ]

            # Example: Grand Canonical Monte Carlo (GCMC)
            steps = [
                {
                    "type": "gcmc",
                    # "name": "my_gcmc_run",
                    "thermo_freq": 250,
                    "steps": 50000,
                    "ensemble": "npt",            # 'npt' or 'nvt'
                    "T": 500,                     # Temperature for ensemble and GCMC
                    "P": 1.0,                     # Pressure for NPT ensemble
                    "aniso_style": "iso",         # For NPT ('iso', 'aniso', 'tri')
                    "elem": ["H", "He"],          # Element(s) to insert/delete. Can be a string or a list.
                    "mu": [-2.5, -3.0],           # Chemical potential(s) of the reservoir(s). Must be a list if multiple elems.
                    "N": 100,                     # Attempt GCMC every N steps. Can be a list.
                    "X": 100,                     # Number of exchange attempts. Can be a list.
                    "M": 0,                       # Number of move attempts (for inserted atoms). Can be a list.
                    "displace": 0.0,              # Max MC translation distance. Can be a list.
                    "overlap_cutoff": 1.0,        # Rejection distance for insertion. Can be a list.
                    "seed": 12345,
                },
            ]

            # Example: Combined GCMC and Atom Swapping (MDMC)
            steps = [
                {
                    "type": "gcmc_mdmc",
                    # "name": "my_combined_run",
                    "ensemble": "npt", "T": 800, "P": 1.0,
                    # GCMC parameters
                    "gcmc_elements": ["H"], "mu": [-1.5], "N": 100, "X": 100,
                    # Atom Swap parameters
                    "atom_swap_elements": ["Ni", "Co"],
                    "swap_N": 100, "swap_M": 100,
                    "swap_T": 800, # Can be different from MD temperature
                },
            ]
    """

    if isinstance(structure, str):
        if specorder is None:
            raise ValueError(
                "`specorder` must be provided, if `structure` is file name"
            )
        lammps_data_filename = structure
    elif isinstance(structure, Atoms):
        symbs = set(structure.get_chemical_symbols())
        if specorder is None:
            specorder = sorted(list(symbs))
        # specorder = [s for s in specorder if s in symbs]

        os.makedirs(working_dir, exist_ok=True)
        lammps_data_filename = "structure.lammps-data"
        write(
            os.path.join(working_dir, lammps_data_filename),
            structure,
            specorder=specorder,
        )
        is_triclinic = not np.allclose(structure.cell.angles(), 90)

    else:
        raise ValueError("`structure` must be either lammps data filename or ASE.Atoms")

    if aniso_style is None:
        aniso_style = "tri" if is_triclinic else "aniso"

    if with_extrapolation_grade:
        if potname.endswith(".yaml"):
            asi_filename = potname.replace(".yaml", ".asi")
        elif potname.endswith(".yace"):
            asi_filename = potname.replace(".yace", ".asi")
        else:
            asi_filename = os.path.splitext(potname)[0] + ".asi"

        # The path to the potential can be absolute or relative to the working_dir
        full_asi_path = os.path.join(working_dir, asi_filename)
        if check_pot_asi_file:
            if not os.path.isfile(full_asi_path):
                raise RuntimeError(
                    f"Potential file {potname} is used with extrapolation, but no active set {full_asi_path} (.asi) is found"
                )

    input_lammps_text = ""

    mass_data = "\n".join(
        [
            f"mass {i + 1} {atomic_masses[atomic_numbers[s]]} # {s}"
            for i, s in enumerate(specorder)
        ]
    )

    if dt is None:
        dt = 0.0005 if "H" in specorder else 0.001

    LAMMPS_INPUT_FILE_TEMPLATE_INIT = """
    #-------------- INIT ---------------------
    units		metal
    dimension	3
    boundary 	p p p
    atom_style	atomic
    variable 	dt equal {dt}

    #---------------- ATOM DEFINITION -------------------
    read_data	{LAMMPS_DATA}

    {mass_data}
    """.format(
        LAMMPS_DATA=lammps_data_filename, dt=dt, mass_data=mass_data
    )

    els = " ".join(specorder)
    el_to_ind = {e: (i + 1) for i, e in enumerate(specorder)}

    fix_pairstyle = pairstyle
    if with_extrapolation_grade:
        if "pace" in pairstyle and "extrapolation" not in pairstyle:
            pairstyle = pairstyle.replace("pace", "pace/extrapolation")
            fix_pairstyle = pairstyle
        elif "grace/fs" in pairstyle:
            if "extrapolation" not in pairstyle:
                pairstyle = pairstyle.replace("grace/fs", "grace/fs extrapolation")
            fix_pairstyle = pairstyle.replace(" extrapolation", "")

    if with_extrapolation_grade:
        LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE = """
        #---------------- FORCE FIELDS --------------------"""

        if is_gpu:
            LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE += dedent(
                f"""
            pair_style 	{pairstyle} chunksize 1024"""
            )
        else:
            LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE += dedent(
                f"""
            pair_style 	{pairstyle}"""
            )

        LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE += dedent(
            """
        pair_coeff	* * {potname} {asi_filename} {els}
        
        fix gamma all pair {minimize_thermo_freq} {fix_pairstyle} gamma 1
        compute max_gamma all reduce max f_gamma
        """.format(
                potname=potname,
                asi_filename=asi_filename,
                minimize_thermo_freq=minimize_thermo_freq,
                pairstyle=pairstyle,
                fix_pairstyle=fix_pairstyle,
                els=els,
            )
        )
    else:  # recursive yace
        if is_gpu:
            LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE = dedent(
                f"""
            pair_style {pairstyle} product chunksize 1024
            """
            )
        else:
            LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE = dedent(
                f"""
            pair_style {pairstyle}
            """
            )

        LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE += dedent(
            """
        pair_coeff      * * {potname} {els}
        """.format(
                potname=potname, els=els
            )
        )

    LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE += dedent(
        """
        neighbor	2.0 bin
        neigh_modify	delay 0 every 1 check yes

        # -------------------- min_dist --------------------
        compute dist all pair/local dist 
        compute  min_dist all reduce  min c_dist inputs local
        """
    )

    input_lammps_text += dedent(LAMMPS_INPUT_FILE_TEMPLATE_INIT)
    input_lammps_text += dedent(LAMMPS_INPUT_FILE_TEMPLATE_PAIRSTYLE)

    c_max_gamma = "c_max_gamma" if with_extrapolation_grade else ""
    f_gamma = "f_gamma" if with_extrapolation_grade else ""

    # backward compat.: steps - convert text only steps to new list of dict
    if isinstance(steps[0], str):
        steps = [{"name": step, "type": step} for step in steps]
    # loop over steps
    for step_dict in steps:
        # step_name
        step_type = step_dict["type"]
        step_name = step_dict.get("name") or step_type

        thermo_freq = step_dict.get("thermo_freq", minimize_thermo_freq)
        gl = step_dict.get("gamma_lo", gamma_lo)
        gh = step_dict.get("gamma_hi", gamma_hi)

        if dump_extrapolative:
            dump_skip_in = f"""variable dump_skip equal "c_max_gamma < {gl}"
                            dump dump_extrapolative all custom {thermo_freq} extrapolative_structures.{step_name}.dump id type element x y z f_gamma
                            dump_modify dump_extrapolative skip v_dump_skip
                            dump_modify dump_extrapolative element {els}
                            variable max_gamma equal c_max_gamma
                            fix extreme_extrapolation all halt {thermo_freq} v_max_gamma > {gh}
                            """
            dump_skip_out = "undump dump_extrapolative"
        else:
            dump_skip_in = dump_skip_out = ""

        if step_type.startswith("min_"):
            # "min_full", "min_atomic", "min_iso", "min_aniso",
            if "min_atomic" in step_type:
                fix_box_relax = ""
            elif "min_full" in step_type:
                fix_box_relax = (
                    "fix box_relax all box/relax {aniso_style} 0.0 vmax 0.001".format(
                        aniso_style=step_dict.get("aniso_style", aniso_style)
                    )
                )
            elif "min_iso" in step_type:
                fix_box_relax = "fix box_relax all box/relax iso 0.0 vmax 0.001"
            elif "min_aniso" in step_type:
                fix_box_relax = "fix box_relax all box/relax aniso 0.0 vmax 0.001"
            else:
                raise ValueError("Unknown minimization step type: {}".format(step_type))

            un_fix_box_relax = "unfix box_relax" if fix_box_relax else ""

            LAMMPS_INPUT_MINIMIZE_TEMPLATE = dedent(
                """
            print "-------------- MINIMIZE  {step_name} ---------------------"
            {fix_gamma}
            reset_timestep	0
            thermo_style custom step cpuremain pe  fmax c_min_dist {c_max_gamma}  press vol density pxx pyy pzz pxy pxz pyz xy xz yz
            thermo	{minimize_thermo_freq}
            thermo_modify flush yes
            dump	dump_relax all custom {minimize_thermo_freq} dump.{step_name}.dump id type element xu yu zu {f_gamma}
            dump_modify dump_relax element {els}
            {dump_skip_in}
            {fix_box_relax}

            min_style cg
            minimize 0 1.0e-3 {n_min_steps} {n_min_steps}

            undump dump_relax
            {dump_skip_out}
            {un_fix_box_relax}
            
            write_data {step_name}.lammps-data

            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print

            """.format(
                    c_max_gamma=c_max_gamma,
                    f_gamma=f_gamma,
                    n_min_steps=step_dict.get("steps", n_min_steps),
                    minimize_thermo_freq=step_dict.get(
                        "thermo_freq", minimize_thermo_freq
                    ),
                    fix_box_relax=fix_box_relax,
                    un_fix_box_relax=un_fix_box_relax,
                    fix_gamma=(
                        f"fix gamma all pair {step_dict.get('thermo_freq', minimize_thermo_freq)} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    step_name=step_name,
                    els=els,
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                )
            )

            input_lammps_text += dedent(LAMMPS_INPUT_MINIMIZE_TEMPLATE)

        elif "eq" == step_type:
            eq_type = step_dict.get("eq_type", "npt")
            if eq_type == "npt":
                fix_eq_str = "fix eq all {eq_type} temp {T1} {T2} 0.1 {fix_npt_style} {P1} {P2} 0.1".format(
                    eq_type=eq_type,
                    T1=step_dict.get("T1", T),
                    T2=step_dict.get("T2", T),
                    P1=step_dict.get("P1", 0),
                    P2=step_dict.get("P2", 0),
                    fix_npt_style=step_dict.get("aniso_style", aniso_style),
                )
            elif eq_type == "nvt":
                fix_eq_str = "fix eq all {eq_type} temp {T1} {T2} 0.1".format(
                    eq_type=eq_type,
                    T1=step_dict.get("T1", T),
                    T2=step_dict.get("T2", T),
                )
            elif eq_type == "nve":
                fix_eq_str = "fix eq all {eq_type}".format(
                    eq_type=eq_type,
                )
            else:
                raise ValueError(f"Unknown eq_type {eq_type}")

            velocity_str = ""
            if "T_init" in step_dict:
                T_init = step_dict["T_init"]
                if T_init is not None:
                    velocity_str = (
                        "velocity all create {T_init} {seed} mom yes rot yes".format(
                            T_init=T_init,
                            seed=step_dict.get("seed", 424242),
                        )
                    )
            else:
                velocity_str = (
                    "velocity all create {T_init} {seed} mom yes rot yes".format(
                        T_init=2 * T,
                        seed=step_dict.get("seed", 424242),
                    )
                )

            if step_dict.get("dump", True):
                dump_in = """dump	eq_dump all custom {equlibration_thermo_freq} dump.{step_name}.dump  id type element xu yu zu {f_gamma}
                dump_modify eq_dump element {els}""".format(
                    equlibration_thermo_freq=step_dict.get(
                        "thermo_freq", equilibration_thermo_freq
                    ),
                    step_name=step_name,
                    f_gamma=f_gamma,
                    els=els,
                )
                dump_out = "undump	eq_dump"
            else:
                dump_in = dump_out = ""

            LAMMPS_INPUT_EQUILIBRATION_TEMPLATE = dedent(
                """
            print "-------------- EQUILIBRATION {step_name} ---------------------"
            {fix_gamma}
            thermo_style custom step cpuremain temp pe c_min_dist fmax {c_max_gamma} press vol density pxx pyy pzz pxy pxz pyz 
            thermo_modify flush yes
            reset_timestep	0
            timestep $(dt)
            
            {velocity_str}
            
            # thermostat + barostat
            {fix_eq_str}
            
            thermo	{equlibration_thermo_freq}
            
            {dump_in}
            {dump_skip_in}
            run	{n_eq_steps}
            
            unfix 	eq
            {dump_out}
            {dump_skip_out}
            write_data {step_name}.lammps-data
            
            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print

            """.format(
                    velocity_str=velocity_str,
                    fix_eq_str=fix_eq_str,
                    equlibration_thermo_freq=step_dict.get(
                        "thermo_freq", equilibration_thermo_freq
                    ),
                    dump_in=dump_in,
                    dump_out=dump_out,
                    step_name=step_name,
                    n_eq_steps=step_dict.get("steps", n_eq_steps),
                    fix_npt_style=step_dict.get("aniso_style", aniso_style),
                    c_max_gamma=c_max_gamma,
                    fix_gamma=(
                        f"fix gamma all pair {step_dict.get('thermo_freq', equilibration_thermo_freq)} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                    seed=step_dict.get("seed", 424242),
                )
            )

            input_lammps_text += dedent(LAMMPS_INPUT_EQUILIBRATION_TEMPLATE)

        elif "mdmc" == step_type:
            swap_commands = []
            swap_columns = []
            fix_names = []
            swap_elements = step_dict.get("elements", specorder)

            for t1 in range(len(swap_elements) - 1):
                for t2 in range(t1 + 1, len(swap_elements)):
                    fix_name = f"{swap_elements[t1]}{swap_elements[t2]}"
                    swap_commands.append(
                        f"fix {fix_name}  all atom/swap {step_dict.get('N', 100)} {step_dict.get('M', 100)} {10 + t1}{20 + t2} {step_dict.get('T_MDMC', T)} ke yes types {el_to_ind[swap_elements[t1]]} {el_to_ind[swap_elements[t2]]}"
                    )
                    swap_columns.append(f"f_{fix_name}[2]")
                    fix_names.append(fix_name)

            LAMMPS_INPUT_MDMC_TEMPLATE = dedent(
                """
            print "-------------- MDMC {step_name} ---------------------"
            {fix_gamma}
            reset_timestep	0
            timestep $(dt)
            {swap_commands}
            
            # thermostat + barostat
            fix mdmc_npt all npt temp {T1} {T2} 0.1 {fix_npt_style} {P1} {P2} 0.1
            
            thermo_style custom step cpuremain temp pe fmax c_min_dist {swap_columns} {c_max_gamma} press vol density pxx pyy pzz pxy pxz pyz 
            thermo	{mdmc_thermo_freq}
            thermo_modify flush yes
            dump	mdmc_dump all custom {mdmc_thermo_freq} dump.{step_name}.dump  id type element xu yu zu {f_gamma}
            dump_modify mdmc_dump element {els}
            {dump_skip_in}

            run {n_mdmc_steps}
            
            {unfix_swaps}
            
            unfix mdmc_npt
            undump mdmc_dump
            {dump_skip_out}
            
            thermo_style custom step cpuremain temp pe fmax c_min_dist {c_max_gamma} press vol density pxx pyy pzz pxy pxz pyz
            write_data {step_name}.lammps-data
            
            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print
            """.format(
                    T1=step_dict.get("T1", T),
                    T2=step_dict.get("T2", T),
                    P1=step_dict.get("P1", 0),
                    P2=step_dict.get("P2", 0),
                    fix_npt_style=step_dict.get("aniso_style", aniso_style),
                    n_mdmc_steps=step_dict.get("steps", n_mdmc_steps),
                    mdmc_thermo_freq=step_dict.get("thermo_freq", mdmc_thermo_freq),
                    swap_commands="\n".join(swap_commands),
                    swap_columns=" ".join(swap_columns),
                    unfix_swaps="\n".join([f"unfix {fn}" for fn in fix_names]),
                    step_name=step_name,
                    c_max_gamma=c_max_gamma,
                    f_gamma=f_gamma,
                    els=els,
                    fix_gamma=(
                        f"fix gamma all pair {equilibration_thermo_freq} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                )
            )
            input_lammps_text += dedent(LAMMPS_INPUT_MDMC_TEMPLATE)
        elif "deform_npt" == step_type:
            def_direction = step_dict.get("direction", "z")
            erate = step_dict.get("erate", 0.01)
            fix_def_npt_pres = " ".join(
                [f"{a} 0 0 0.5" for a in ["x", "y", "z"] if a != def_direction]
            )
            LAMMPS_INPUT_DEFORM_TEMPLATE = dedent(
                """
            print "-------------- DEFORM_NPT {step_name} ---------------------"
            {fix_gamma}
            thermo_style custom step cpuremain temp pe c_min_dist fmax {c_max_gamma} press vol density cella cellb cellc pxx pyy pzz pxy pxz pyz 
            thermo_modify flush yes
            reset_timestep	0
            timestep $(dt)
            
            # thermostat + barostat
            fix def_npt all npt temp {T1} {T2} 0.1 {fix_def_npt_pres} drag 0.5
            fix fix_def all deform 1 {def_direction} erate {erate} units box remap x remap v

            thermo	{equlibration_thermo_freq}
            
            dump	def_dump all custom {equlibration_thermo_freq} dump.{step_name}.dump  id type element xu yu zu {f_gamma}
            dump_modify def_dump element {els}

            {dump_skip_in}
            run	{n_eq_steps}
            
            unfix   def_npt
            unfix   fix_def
            undump	def_dump
            {dump_skip_out}
            write_data {step_name}.lammps-data
            
            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print

            """.format(
                    T_init=step_dict.get("T_init", 2 * T),
                    T1=step_dict.get("T1", T),
                    T2=step_dict.get("T2", T),
                    P1=step_dict.get("P1", 0),
                    P2=step_dict.get("P2", 0),
                    equlibration_thermo_freq=step_dict.get(
                        "thermo_freq", equilibration_thermo_freq
                    ),
                    fix_def_npt_pres=fix_def_npt_pres,
                    def_direction=def_direction,
                    erate=erate,
                    step_name=step_name,
                    n_eq_steps=step_dict.get("steps", n_eq_steps),
                    c_max_gamma=c_max_gamma,
                    f_gamma=f_gamma,
                    els=els,
                    fix_gamma=(
                        f"fix gamma all pair {step_dict.get('thermo_freq', equilibration_thermo_freq)} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                    seed=step_dict.get("seed", 424242),
                )
            )

            input_lammps_text += dedent(LAMMPS_INPUT_DEFORM_TEMPLATE)
        elif "deform_nvt" == step_type:
            def_direction = step_dict.get("direction", "z")
            erate = step_dict.get("erate", 0.01)
            fix_def_npt_pres = " ".join(
                [f"{a} 0 0 0.5" for a in ["x", "y", "z"] if a != def_direction]
            )
            LAMMPS_INPUT_DEFORM_TEMPLATE = dedent(
                """
            print "-------------- DEFORM_NVT {step_name} ---------------------"
            {fix_gamma}
            thermo_style custom step cpuremain temp pe c_min_dist fmax {c_max_gamma} press density cella cellb cellc pxx pyy pzz pxy pxz pyz 
            thermo_modify flush yes
            reset_timestep	0
            timestep $(dt)
            
            # thermostat
            fix def_npt all nvt temp {T1} {T2} 0.1
            fix fix_def all deform 1 {def_direction} erate {erate} units box remap x remap v

            thermo	{equlibration_thermo_freq}
            
            dump	def_dump all custom {equlibration_thermo_freq} dump.{step_name}.dump  id type element xu yu zu {f_gamma}
            dump_modify def_dump element {els}
            {dump_skip_in}
            run	{n_eq_steps}
            
            unfix   def_npt
            unfix   fix_def
            undump	def_dump
            {dump_skip_out}
            write_data {step_name}.lammps-data
            
            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print

            """.format(
                    T_init=step_dict.get("T_init", 2 * T),
                    T1=step_dict.get("T1", T),
                    T2=step_dict.get("T2", T),
                    P1=step_dict.get("P1", 0),
                    P2=step_dict.get("P2", 0),
                    equlibration_thermo_freq=step_dict.get(
                        "thermo_freq", equilibration_thermo_freq
                    ),
                    fix_def_npt_pres=fix_def_npt_pres,
                    def_direction=def_direction,
                    erate=erate,
                    step_name=step_name,
                    n_eq_steps=step_dict.get("steps", n_eq_steps),
                    c_max_gamma=c_max_gamma,
                    f_gamma=f_gamma,
                    fix_gamma=(
                        f"fix gamma all pair {step_dict.get('thermo_freq', equilibration_thermo_freq)} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                    els=els,
                    seed=step_dict.get("seed", 424242),
                )
            )

            input_lammps_text += dedent(LAMMPS_INPUT_DEFORM_TEMPLATE)
        elif "shear_nvt" == step_type:
            def_direction = step_dict.get("direction", "xz")
            erate = step_dict.get("erate", 0.01)
            fix_def_npt_pres = " ".join(
                [f"{a} 0 0 0.5" for a in ["x", "y", "z"] if a != def_direction]
            )
            LAMMPS_INPUT_DEFORM_TEMPLATE = dedent(
                """
            print "-------------- SHEAR_NVT {step_name} ---------------------"
            change_box  all triclinic
            {fix_gamma}
            thermo_style custom step cpuremain temp pe c_min_dist fmax {c_max_gamma} press density cella cellb cellc pxx pyy pzz pxy pxz pyz 
            thermo_modify flush yes
            reset_timestep	0
            timestep $(dt)
            
            # thermostat
            fix def_npt all nvt temp {T1} {T2} 0.1
            fix fix_def all deform 1 {def_direction} erate {erate} units box remap x remap v

            thermo	{equlibration_thermo_freq}
            
            dump	def_dump all custom {equlibration_thermo_freq} dump.{step_name}.dump  id type element xu yu zu {f_gamma}
            dump_modify def_dump element {els}
            {dump_skip_in}
            run	{n_eq_steps}
            
            unfix   def_npt
            unfix   fix_def
            undump	def_dump
            {dump_skip_out}
            write_data {step_name}.lammps-data
            
            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print

            """.format(
                    T_init=step_dict.get("T_init", 2 * T),
                    T1=step_dict.get("T1", T),
                    T2=step_dict.get("T2", T),
                    P1=step_dict.get("P1", 0),
                    P2=step_dict.get("P2", 0),
                    equlibration_thermo_freq=step_dict.get(
                        "thermo_freq", equilibration_thermo_freq
                    ),
                    fix_def_npt_pres=fix_def_npt_pres,
                    def_direction=def_direction,
                    erate=erate,
                    step_name=step_name,
                    n_eq_steps=step_dict.get("steps", n_eq_steps),
                    c_max_gamma=c_max_gamma,
                    f_gamma=f_gamma,
                    fix_gamma=(
                        f"fix gamma all pair {step_dict.get('thermo_freq', equilibration_thermo_freq)} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                    els=els,
                    seed=step_dict.get("seed", 424242),
                )
            )

            input_lammps_text += dedent(LAMMPS_INPUT_DEFORM_TEMPLATE)
        elif "gcmc" == step_type:
            T = step_dict.get("T", 300)
            P = step_dict.get("P", 0.0)

            elems = step_dict.get("elem", "H")
            if isinstance(elems, str):
                elems = [elems]

            def get_param_list(param_name, default_value):
                val = step_dict.get(param_name, default_value)
                if not isinstance(val, list):
                    return [val] * len(elems)
                if len(val) != len(elems):
                    raise ValueError(
                        f"Length of '{param_name}' list ({len(val)}) does not match number of elements ({len(elems)}) for gcmc"
                    )
                return val

            mus = get_param_list("mu", 0.0)
            Ms = get_param_list("M", 0)
            displaces = get_param_list("displace", 0.0)
            overlap_cutoffs = get_param_list("overlap_cutoff", 1.0)
            Ns = get_param_list("N", 100)
            Xs = get_param_list("X", 100)

            gcmc_commands = []
            gcmc_var_cmds = []
            gcmc_columns = []
            unfix_gcmc_cmds = []
            group_cmds = []

            for i, elem in enumerate(elems):
                at_type = el_to_ind[elem]
                group_cmds.append(f"group {elem} type {at_type}")

                gcmc_command = (
                    f"fix gcmc_{elem} {elem} gcmc {Ns[i]} {Xs[i]} "
                    f"{Ms[i]} {at_type} {step_dict.get('seed', 12345) + i} "
                    f"{T} {mus[i]} {displaces[i]} "
                    f"overlap_cutoff {overlap_cutoffs[i]}"
                )
                gcmc_commands.append(gcmc_command)

                var_cmd = []
                cols = []

                if Xs[i] > 0:
                    var_cmd.append(
                        f"variable {elem}_conc equal count({elem})/count(all)"
                    )
                    var_cmd.append(f"variable {elem}_ins equal f_gcmc_{elem}[4]")
                    var_cmd.append(f"variable {elem}_del equal f_gcmc_{elem}[6]")
                    cols.extend([f"v_{elem}_conc", f"v_{elem}_ins", f"v_{elem}_del"])

                if Ms[i] > 0:
                    var_cmd.append(f"variable {elem}_trans equal f_gcmc_{elem}[2]")
                    cols.append(f"v_{elem}_trans")

                gcmc_var_cmds.extend(var_cmd)
                gcmc_columns.extend(cols)

                unfix_gcmc_cmds.append(f"unfix gcmc_{elem}")

            gcmc_command_str = "\n".join(gcmc_commands)
            gcmc_var_cmd_str = "\n".join(gcmc_var_cmds)
            gcmc_columns_str = " ".join(gcmc_columns)
            unfix_gcmc_str = "\n".join(unfix_gcmc_cmds)
            group_cmds_str = "\n".join(group_cmds)

            # ensemble command (NVT, NPT)
            ensemble = step_dict.get("ensemble", "npt")
            if ensemble == "npt":
                ensemble_command = f"fix dyn all {ensemble} temp {T} {T} 0.1 {step_dict.get('aniso_style', 'iso')} {P} {P} 0.1"
            elif ensemble == "nvt":
                ensemble_command = f"fix dyn all {ensemble} temp {T} {T} 0.1"
            else:
                raise ValueError(
                    f"Unknown ensemble type {ensemble} (should be npt or nvt)"
                )
            tfreq = step_dict.get("thermo_freq", mdmc_thermo_freq)
            LAMMPS_INPUT_TEMPLATE = dedent(
                """
            print "-------------- GCMC {step_name} ---------------------"
            {group_cmds}

            {fix_gamma}
            reset_timestep	0
            timestep $(dt)
            {gcmc_command}
            {gcmc_var_cmd}

            # thermostat + barostat
            {ensemble_command}

            thermo_style custom step cpuremain temp pe fmax c_min_dist {gcmc_columns} {c_max_gamma} press vol density pxx pyy pzz pxy pxz pyz
            thermo	{mdmc_thermo_freq}
            thermo_modify flush yes
            dump	mdmc_dump all custom {mdmc_thermo_freq} dump.{step_name}.dump  id type element xu yu zu {f_gamma}
            dump_modify mdmc_dump element {els}

            {dump_skip_in}

            run {n_mdmc_steps}

            {unfix_gcmc}
            unfix dyn

            undump mdmc_dump
            {dump_skip_out}

            thermo_style custom step cpuremain temp pe fmax c_min_dist {c_max_gamma} press vol density pxx pyy pzz pxy pxz pyz
            write_data {step_name}.lammps-data

            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print
            """.format(
                    n_mdmc_steps=step_dict.get("steps", n_mdmc_steps),
                    mdmc_thermo_freq=tfreq,
                    group_cmds=group_cmds_str,
                    gcmc_var_cmd=gcmc_var_cmd_str,
                    gcmc_command=gcmc_command_str,
                    gcmc_columns=gcmc_columns_str,
                    unfix_gcmc=unfix_gcmc_str,
                    ensemble_command=ensemble_command,
                    step_name=step_name,
                    c_max_gamma=c_max_gamma,
                    f_gamma=f_gamma,
                    fix_gamma=(
                        f"fix gamma all pair {tfreq} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    els=els,
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                )
            )
            input_lammps_text += dedent(LAMMPS_INPUT_TEMPLATE)
        elif "gcmc_mdmc" == step_type:
            T = step_dict.get("T", 300)
            P = step_dict.get("P", 0.0)

            # --- GCMC part ---
            gcmc_elems = step_dict.get("gcmc_elements", step_dict.get("elem"))
            gcmc_commands = []
            gcmc_var_cmds = []
            gcmc_columns = []
            unfix_gcmc_cmds = []
            group_cmds = []

            if gcmc_elems:
                if isinstance(gcmc_elems, str):
                    gcmc_elems = [gcmc_elems]

                def get_param_list(param_name, default_value):
                    val = step_dict.get(param_name, default_value)
                    if not isinstance(val, list):
                        return [val] * len(gcmc_elems)
                    if len(val) != len(gcmc_elems):
                        raise ValueError(
                            f"Length of '{param_name}' list ({len(val)}) does not match number of gcmc_elements ({len(gcmc_elems)})"
                        )
                    return val

                mus = get_param_list("mu", 0.0)
                Ms = get_param_list("M", 0)
                displaces = get_param_list("displace", 0.0)
                overlap_cutoffs = get_param_list("overlap_cutoff", 1.0)
                Ns = get_param_list("N", 100)
                Xs = get_param_list("X", 100)

                for i, elem in enumerate(gcmc_elems):
                    at_type = el_to_ind[elem]
                    group_cmds.append(f"group {elem} type {at_type}")

                    gcmc_command = (
                        f"fix gcmc_{elem} {elem} gcmc {Ns[i]} {Xs[i]} "
                        f"{Ms[i]} {at_type} {step_dict.get('seed', 12345) + i} "
                        f"{T} {mus[i]} {displaces[i]} "
                        f"overlap_cutoff {overlap_cutoffs[i]}"
                    )
                    gcmc_commands.append(gcmc_command)

                    var_cmd = []
                    cols = []

                    if Xs[i] > 0:
                        var_cmd.append(
                            f"variable {elem}_conc equal count({elem})/count(all)"
                        )
                        var_cmd.append(f"variable {elem}_ins equal f_gcmc_{elem}[4]")
                        var_cmd.append(f"variable {elem}_del equal f_gcmc_{elem}[6]")
                        cols.extend(
                            [f"v_{elem}_conc", f"v_{elem}_ins", f"v_{elem}_del"]
                        )

                    if Ms[i] > 0:
                        var_cmd.append(f"variable {elem}_trans equal f_gcmc_{elem}[2]")
                        cols.append(f"v_{elem}_trans")

                    gcmc_var_cmds.extend(var_cmd)
                    gcmc_columns.extend(cols)

                    unfix_gcmc_cmds.append(f"unfix gcmc_{elem}")

            # --- Atom Swap (MDMC) part ---
            swap_commands = []
            swap_columns = []
            unfix_swap_cmds = []
            atom_swap_elements = step_dict.get("atom_swap_elements")
            if atom_swap_elements:
                swap_N = step_dict.get("swap_N", 100)
                swap_M = step_dict.get("swap_M", 100)
                swap_T = step_dict.get("swap_T", T)
                swap_seed_start = step_dict.get("seed", 12345) + 1000  # offset seed

                for t1 in range(len(atom_swap_elements) - 1):
                    for t2 in range(t1 + 1, len(atom_swap_elements)):
                        el1 = atom_swap_elements[t1]
                        el2 = atom_swap_elements[t2]
                        fix_name = f"{el1}{el2}"
                        swap_commands.append(
                            f"fix {fix_name} all atom/swap {swap_N} {swap_M} {swap_seed_start + t1 * 10 + t2} {swap_T} ke yes types {el_to_ind[el1]} {el_to_ind[el2]}"
                        )
                        swap_columns.append(f"f_{fix_name}[2]")
                        unfix_swap_cmds.append(f"unfix {fix_name}")

            # --- Combine commands ---
            all_mc_commands = gcmc_commands + swap_commands
            all_mc_command_str = "\n".join(all_mc_commands)

            all_var_cmds = gcmc_var_cmds
            all_var_cmd_str = "\n".join(all_var_cmds)

            all_columns = gcmc_columns + swap_columns
            all_columns_str = " ".join(all_columns)

            all_unfix_cmds = unfix_gcmc_cmds + unfix_swap_cmds
            all_unfix_str = "\n".join(all_unfix_cmds)

            group_cmds_str = "\n".join(group_cmds)

            # ensemble command (NVT, NPT)
            ensemble = step_dict.get("ensemble", "npt")
            if ensemble == "npt":
                ensemble_command = f"fix dyn all {ensemble} temp {T} {T} 0.1 {step_dict.get('aniso_style', 'iso')} {P} {P} 0.1"
            elif ensemble == "nvt":
                ensemble_command = f"fix dyn all {ensemble} temp {T} {T} 0.1"
            else:
                raise ValueError(
                    f"Unknown ensemble type {ensemble} (should be npt or nvt)"
                )
            tfreq = step_dict.get("thermo_freq", mdmc_thermo_freq)
            LAMMPS_INPUT_TEMPLATE = dedent(
                """
            print "-------------- GCMC+MDMC {step_name} ---------------------"
            {group_cmds}

            {fix_gamma}
            reset_timestep	0
            timestep $(dt)
            {all_mc_command}
            {all_var_cmd}

            # thermostat + barostat
            {ensemble_command}

            thermo_style custom step cpuremain temp pe fmax c_min_dist {all_columns} {c_max_gamma} press vol density pxx pyy pzz pxy pxz pyz
            thermo	{mdmc_thermo_freq}
            thermo_modify flush yes
            dump	mdmc_dump all custom {mdmc_thermo_freq} dump.{step_name}.dump  id type element xu yu zu {f_gamma}
            dump_modify mdmc_dump element {els}

            {dump_skip_in}

            run {n_mdmc_steps}

            {all_unfix}
            unfix dyn

            undump mdmc_dump
            {dump_skip_out}

            thermo_style custom step cpuremain temp pe fmax c_min_dist {c_max_gamma} press vol density pxx pyy pzz pxy pxz pyz
            write_data {step_name}.lammps-data

            fix fix_print all print 1 "$(pe)" file {step_name}.energy.dat
            run 0
            unfix fix_print
            """.format(
                    n_mdmc_steps=step_dict.get("steps", n_mdmc_steps),
                    mdmc_thermo_freq=tfreq,
                    group_cmds=group_cmds_str,
                    all_var_cmd=all_var_cmd_str,
                    all_mc_command=all_mc_command_str,
                    all_columns=all_columns_str,
                    all_unfix=all_unfix_str,
                    ensemble_command=ensemble_command,
                    step_name=step_name,
                    c_max_gamma=c_max_gamma,
                    f_gamma=f_gamma,
                    fix_gamma=(
                        f"fix gamma all pair {tfreq} {fix_pairstyle} gamma 1"
                        if with_extrapolation_grade
                        else ""
                    ),
                    els=els,
                    dump_skip_in=dump_skip_in,
                    dump_skip_out=dump_skip_out,
                )
            )
            input_lammps_text += dedent(LAMMPS_INPUT_TEMPLATE)
        else:
            raise ValueError(f"Unknown step: {step_type}")

    input_lammps_text = dedent(input_lammps_text)
    input_lammps_text = "\n".join(s.strip() for s in input_lammps_text.split("\n"))

    os.makedirs(working_dir, exist_ok=True)
    with open(os.path.join(working_dir, "in.lammps"), "w") as f:
        print(input_lammps_text, file=f)


def read_lammps_energy(fname):
    with open(fname) as f:
        lines = f.readlines()
    lines = [l.strip() for l in lines if not l.startswith("#")]
    e = float(lines[0])
    return e


def restore_chemical_symbols(atoms, specorder):
    rev_atom_types = {i + 1: s for i, s in enumerate(specorder)}
    try:
        symbs = [rev_atom_types[z] for z in atoms.get_atomic_numbers()]
        atoms.set_chemical_symbols(symbs)
    except KeyError:
        pass


def read_lammps_structure(fname, specorder):
    atoms = read(fname, format="lammps-data", style="atomic")
    restore_chemical_symbols(atoms, specorder)
    return atoms


# def collect_lammps_simulations(working_dir, specorder, steps=("min_full",)):
#     working_dir = os.path.abspath(working_dir)
#     last_step_out = os.path.join(working_dir, steps[-1] + ".energy.dat")
#     last_step_structure = os.path.join(working_dir, steps[-1] + ".lammps-data")
#     if os.path.isfile(last_step_out) and os.path.isfile(last_step_structure):
#         return read_lammps_energy(last_step_out), read_lammps_structure(
#             last_step_structure, specorder
#         )


def specorder_from_structure(structure):
    symbs = set(structure.get_chemical_symbols())
    specorder = sorted(list(symbs))
    return specorder


def run_lammps_simulation(
    structure,
    potname,
    working_dir,
    T=300,
    dt=None,
    with_extrapolation_grade=False,
    dump_extrapolative=False,
    is_gpu=False,
    steps=("min_full",),
    n_min_steps=500,
    n_eq_steps=10000,
    n_mdmc_steps=50000,
    minimize_thermo_freq=50,
    equilibration_thermo_freq=250,
    mdmc_thermo_freq=250,
    is_triclinic=False,
    aniso_style=None,
    specorder=None,
    gamma_lo=5,
    gamma_hi=15,
    check_pot_asi_file=True,
    pairstyle="pace",
    lmp_exec="lmp",
    rerun=False,
):
    """
    Runs a LAMMPS simulation as a sequence of steps for a given structure and potential.

    This function writes a LAMMPS input script based on the specified steps, runs the
    simulation, and returns the final energy and structure.

    Note: `working_dir` should be unique for a given structure and list of steps.
    The simulation will be skipped if the output files already exist, unless `rerun=True`.

    Args:
        structure (ase.Atoms or str): The atomic structure for the simulation. Can be an
            ASE Atoms object or a path to a LAMMPS data file.
        potname (str): The filename of the interatomic potential (e.g., 'pot.yace').
        working_dir (str): The working directory for the simulation.
        T (float, optional): Default temperature (K) for MD steps. Defaults to 300.
        dt (float, optional): Timestep in ps. Defaults to 0.001 for systems with H,
            otherwise 0.0005.
        with_extrapolation_grade (bool, optional): If True, compute and monitor the
            ACE extrapolation grade. Requires a `.asi` file. Defaults to False.
        dump_extrapolative (bool, optional): If True, dump structures where the
            extrapolation grade exceeds `gamma_lo`. Defaults to False.
        is_gpu (bool, optional): If True, use GPU-accelerated pair styles. Defaults to False.
        steps (tuple, optional): A sequence of dictionaries defining the simulation steps.
            See `write_in_lammps` docstring for examples. Defaults to `("min_full",)`.
        n_min_steps (int, optional): Default number of steps for minimization. Defaults to 500.
        n_eq_steps (int, optional): Default number of steps for equilibration. Defaults to 10000.
        n_mdmc_steps (int, optional): Default number of steps for MDMC. Defaults to 50000.
        minimize_thermo_freq (int, optional): Default thermo output frequency for minimization.
            Defaults to 50.
        equilibration_thermo_freq (int, optional): Default thermo output frequency for
            equilibration. Defaults to 250.
        mdmc_thermo_freq (int, optional): Default thermo output frequency for MDMC.
            Defaults to 250.
        is_triclinic (bool, optional): If True, assumes a triclinic box. Automatically
            detected from ASE Atoms. Defaults to False.
        aniso_style (str, optional): Default pressure coupling style ('iso', 'aniso', 'tri').
            Defaults to 'tri' for triclinic, 'aniso' otherwise.
        specorder (list, optional): Order of species for LAMMPS atom types. Inferred from
            structure if not provided.
        gamma_lo (int, optional): Lower bound of extrapolation grade for `dump_extrapolative`.
            Defaults to 5.
        gamma_hi (int, optional): Upper bound of extrapolation grade to halt simulation.
            Defaults to 15.
        check_pot_asi_file (bool, optional): If True, verify the existence of the `.asi`
            file when `with_extrapolation_grade` is enabled. Defaults to True.
        pairstyle (str, optional): LAMMPS pair style to use. Defaults to "pace".
        lmp_exec (str or list, optional): LAMMPS executable command. Can be a string or a
            list of arguments (e.g., ['mpirun', '-np', '4', 'lmp']). Defaults to "lmp".
        rerun (bool, optional): If True, force the simulation to run even if output
            files exist. Defaults to False.

    Returns:
        tuple: A tuple containing the following elements:
            - energy (float): The final potential energy of the system.
            - structure (ase.Atoms): The final atomic structure.

    Examples of `steps` dictionaries:

        .. code-block:: python

            # Full relaxation (atomic positions and cell)
            steps = [{"type": "min_full"}]

            # NPT equilibration at 500K
            steps = [{"type": "eq", "eq_type": "npt", "T1": 500, "T2": 500}]

            # MDMC atom swapping for a high-entropy alloy
            steps = [{
                "type": "mdmc",
                "elements": ["Ni", "Co", "Cr"],
                "T_MDMC": 1200,
                "T1": 1200, "T2": 1200
            }]

            # Tensile deformation in 'z' direction
            steps = [{"type": "deform_npt", "direction": "z", "erate": 0.001}]

            # GCMC simulation to insert Hydrogen
            steps = [{
                "type": "gcmc",
                "elem": "H",
                "mu": -2.5,
                "T": 600
            }]
    """
    if isinstance(lmp_exec, str):
        lmp_exec = [lmp_exec]
    specorder = specorder or specorder_from_structure(structure)
    # backward compat.: steps - convert text only steps to new list of dict
    steps = [
        {"name": step, "type": step} if isinstance(step, str) else step
        for step in steps
    ]
    for step_dict in steps:
        if "name" not in step_dict:
            step_dict["name"] = step_dict["type"]

    working_dir = os.path.abspath(working_dir)
    last_step_out = os.path.join(working_dir, steps[-1]["name"] + ".energy.dat")
    last_step_structure = os.path.join(working_dir, steps[-1]["name"] + ".lammps-data")
    if (
        os.path.isfile(last_step_out)
        and os.path.isfile(last_step_structure)
        and not rerun
    ):
        return read_lammps_energy(last_step_out), read_lammps_structure(
            last_step_structure, specorder
        )
    else:

        write_in_lammps(
            structure=structure,
            potname=potname,
            working_dir=working_dir,
            T=T,
            dt=dt,
            with_extrapolation_grade=with_extrapolation_grade,
            dump_extrapolative=dump_extrapolative,
            is_gpu=is_gpu,
            steps=steps,
            n_min_steps=n_min_steps,
            n_eq_steps=n_eq_steps,
            n_mdmc_steps=n_mdmc_steps,
            minimize_thermo_freq=minimize_thermo_freq,
            equilibration_thermo_freq=equilibration_thermo_freq,
            mdmc_thermo_freq=mdmc_thermo_freq,
            is_triclinic=is_triclinic,
            aniso_style=aniso_style,
            specorder=specorder,
            gamma_lo=gamma_lo,
            gamma_hi=gamma_hi,
            check_pot_asi_file=check_pot_asi_file,
            pairstyle=pairstyle,
        )

        lmp_exec_str = " ".join(lmp_exec)
        print(f"RUN `{lmp_exec_str} -in in.lammps` in {working_dir}")

        cur_cwd = os.getcwd()
        try:
            os.chdir(working_dir)
            process = subprocess.Popen(
                lmp_exec + ["-in", "in.lammps"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            stdout, stderr = process.communicate()
            print("Finish run")
        finally:
            os.chdir(cur_cwd)

        return read_lammps_energy(last_step_out), read_lammps_structure(
            last_step_structure, specorder
        )
