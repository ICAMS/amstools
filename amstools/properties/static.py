import logging

from amstools.properties.generalcalculator import GeneralCalculator
from amstools.calculators.dft.vasp import AMSVasp


class StaticCalculator(GeneralCalculator):
    """Calculation of static energy, forces and stresses
    :param atoms: original ASE Atoms object


    Usage:
    >>  atoms.calc = calculator
    >>  stat = StaticCalculator(atoms)
    >>  stat.calculate()
    >>  print(stat.value)
    """

    property_name = "static"

    param_names = []

    def generate_structures(self, verbose=False):
        basis_ref = self.basis_ref
        self._structure_dict["static"] = basis_ref
        return self._structure_dict

    def get_structure_value(self, structure, name=None):
        if isinstance(structure.calc, AMSVasp):
            structure.calc.auto_kmesh_spacing = True
            structure.calc.static_calc()
        en = structure.get_potential_energy(force_consistent=True)
        forces = structure.get_forces()

        vol = 0.0
        stress = 0.0
        pressure = 0.0

        try:
            vol = structure.get_volume()
            stress = structure.get_stress()
            pressure = -1.0 / 3.0 * sum(stress[0:3])
        except Exception as e:
            logging.warning("StaticCalculator.get_structure_value error: {}".format(e))

        return {
            "energy": en,
            "forces": forces,
            "stress": stress,
            "pressure": pressure,
            "volume": vol,
        }, structure

    def analyse_structures(self, output_dict):
        self._value.update(output_dict)
