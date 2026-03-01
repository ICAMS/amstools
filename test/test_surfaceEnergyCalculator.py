import json

from amstools.properties.surface import (
    SurfaceEnergyCalculator,
    SurfaceAtomAdsorptionCurveCalculator,
    SurfaceDecohesionCalculator,
)
from test.utils import *
from amstools.utils import JsonNumpyEncoder


def test_calculate(atoms, pipeline_calculator_zero_stress):
    atoms.calc = pipeline_calculator_zero_stress
    prop_calc = SurfaceEnergyCalculator(
        atoms=atoms,
        surface_orientation="100",
        surface_name="X100_Y010_Z001_6at",
        fmax=0.5,
    )
    prop_calc.calculate()
    print(prop_calc._value)

    assert "structure_type" in prop_calc._value
    assert "surface_name" in prop_calc._value
    assert "surface_energy" in prop_calc._value
    assert "surface_area" in prop_calc._value
    assert "number_of_atoms" in prop_calc._value
    assert "surface_orientation" in prop_calc._value
    assert "ref_energy_per_atom" in prop_calc._value
    assert "surface_structure_energy" in prop_calc._value
    assert "surface_energy(mJ/m^2)" in prop_calc._value


def test_available_surfaces(atoms, pipeline_calculator_zero_stress):
    surf_types = SurfaceEnergyCalculator.available_surfaces(atoms, "100")
    print(surf_types)
    assert len(surf_types) > 0


def test_available_surfaces_na_orientation(atoms, pipeline_calculator_zero_stress):
    surf_types2 = SurfaceEnergyCalculator.available_surfaces(atoms, "113")
    print(surf_types2)
    assert len(surf_types2) == 0


def test_available_surfaces_all_orientations(atoms, pipeline_calculator_zero_stress):
    surf_types = SurfaceEnergyCalculator.available_surfaces(atoms)
    print(surf_types)
    assert len(surf_types) > 0


def test_to_from_dict(atoms, pipeline_calculator_zero_stress):
    prop = SurfaceEnergyCalculator(
        atoms=atoms,
        surface_orientation="100",
        surface_name="X100_Y010_Z001_6at",
        fmax=0.5,
    )
    prop.calculate(calculator=pipeline_calculator_zero_stress)

    prop_dict = prop.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    print("prop_dict=", prop_dict)
    print("prop_dict_json=", prop_dict_json)
    prop_dict = json.loads(prop_dict_json)
    new_prop = SurfaceEnergyCalculator.fromdict(prop_dict)
    print("new_prop=", new_prop)
    print("new_prop.basis_ref=", new_prop.basis_ref)
    print("new_prop.value=", new_prop.value)

    assert prop.basis_ref == new_prop.basis_ref
    assert prop.value == new_prop.value


def test_surface_atom_adsorption_curve_calculate(
    atoms, pipeline_calculator_zero_stress
):
    atoms.calc = pipeline_calculator_zero_stress
    surf_calc = SurfaceEnergyCalculator(
        atoms=atoms,
        surface_orientation="100",
        surface_name="X100_Y010_Z001_6at",
        fmax=0.5,
    )
    surf_calc.calculate()
    surface_structure = surf_calc._structure_dict["SURFACE___atomic"]
    surface_structure.calc = pipeline_calculator_zero_stress
    prop_calc = SurfaceAtomAdsorptionCurveCalculator(
        surface_structure,
        adsorption_position_atom_indices=[0],
        z_min=0.5,
        z_max=7.5,
        dz=0.5,
        surface_supercell_size=(2, 2),
    )
    prop_calc.calculate(calculator=pipeline_calculator_zero_stress)
    print("KEYS=", prop_calc._value.keys())

    assert "surface_supercell_size" in prop_calc._value
    assert "z_shift" in prop_calc._value
    assert "adsorption_position_atom_indices" in prop_calc._value
    assert "adsorption_energy_alignment" in prop_calc._value
    assert "top_atoms_positions" in prop_calc._value
    assert "adsorption_position" in prop_calc._value
    assert "raw_adsorption_data" in prop_calc._value
    assert "adsorption_energies" in prop_calc._value


def test_surface_atom_adsorption_curve_to_from_dict(
    atoms, pipeline_calculator_zero_stress
):
    atoms.calc = pipeline_calculator_zero_stress
    surf_calc = SurfaceEnergyCalculator(
        atoms=atoms,
        surface_orientation="100",
        surface_name="X100_Y010_Z001_6at",
        fmax=0.5,
    )
    surf_calc.calculate()
    surface_structure = surf_calc._structure_dict["SURFACE___atomic"]
    surface_structure.calc = pipeline_calculator_zero_stress
    prop = SurfaceAtomAdsorptionCurveCalculator(
        surface_structure,
        adsorption_position_atom_indices=[0],
        z_min=0.5,
        z_max=7.5,
        dz=0.5,
        surface_supercell_size=(2, 2),
    )

    prop.calculate(calculator=pipeline_calculator_zero_stress)

    prop_dict = prop.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    print("prop_dict=", prop_dict)
    print("prop_dict_json=", prop_dict_json)
    print("prop_dict[basis_ref]=", prop_dict["basis_ref"])
    prop_dict = json.loads(prop_dict_json)
    new_prop = SurfaceAtomAdsorptionCurveCalculator.fromdict(prop_dict)
    print("new_prop=", new_prop)
    print("new_prop.basis_ref=", new_prop.basis_ref)
    print("prop.value=", prop.value)
    print("new_prop.value=", new_prop.value)

    assert prop.basis_ref == new_prop.basis_ref
    assert set(prop.value.keys()) == set(new_prop.value.keys())


def test_surface_decohesion_calculate(atoms, pipeline_calculator_zero_stress):
    atoms.calc = pipeline_calculator_zero_stress
    prop_calc = SurfaceDecohesionCalculator(
        atoms=atoms,
        surface_orientation="100",
        surface_name="X100_Y010_Z001_6at",
        dz=1.0,
        fmax=0.5,
    )
    prop_calc.calculate()

    print(prop_calc._value.keys())
    print(prop_calc._value)

    assert "zmin" in prop_calc._value
    assert "zmax" in prop_calc._value
    assert "dz" in prop_calc._value
    assert "structure_type" in prop_calc._value
    assert "surface_name" in prop_calc._value
    assert "surface_orientation" in prop_calc._value
    assert "structure_name" in prop_calc._value
    assert "surface_structure_energy" in prop_calc._value
    assert "number_of_atoms" in prop_calc._value
    assert "surface_area" in prop_calc._value
    assert "ref_energy_per_atom" in prop_calc._value
    assert "surface_energy" in prop_calc._value
    assert "surface_energy(mJ/m^2)" in prop_calc._value
    assert "z" in prop_calc._value
    assert "e_surf" in prop_calc._value
    assert "e_surf_relax" in prop_calc._value
    assert "e_surf(mJ/m^2)" in prop_calc._value
    assert "e_surf_relax(mJ/m^2)" in prop_calc._value

    assert (
        pytest.approx(prop_calc._value["e_surf_relax"][-1], abs=1e-5)
        == prop_calc._value["surface_energy"]
    )

    assert len(prop_calc._value["e_surf_relax"]) == len(prop_calc._value["z"])


def test_surface_decohesion_to_from_dict(atoms, pipeline_calculator_zero_stress):
    atoms.calc = pipeline_calculator_zero_stress
    prop = SurfaceDecohesionCalculator(
        atoms=atoms,
        surface_orientation="100",
        surface_name="X100_Y010_Z001_6at",
        dz=1.0,
        fmax=0.5,
    )

    prop.calculate()

    prop_dict = prop.todict()
    prop_dict_json = json.dumps(prop_dict, cls=JsonNumpyEncoder)
    print("prop_dict=", prop_dict)
    print("prop_dict_json=", prop_dict_json)
    prop_dict = json.loads(prop_dict_json)
    new_prop = SurfaceDecohesionCalculator.fromdict(prop_dict)
    print("new_prop=", new_prop)
    print("new_prop.basis_ref=", new_prop.basis_ref)
    print("new_prop.value=", new_prop.value)

    assert prop.basis_ref == new_prop.basis_ref
    assert prop.value == new_prop.value
