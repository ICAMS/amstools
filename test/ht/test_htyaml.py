import pytest
from amstools.highthroughput.generate_ht_pipeline_setup import *
from amstools.highthroughput.utils import create_calc, initialize_pipe


# Make test paths independent of the current working directory
TEST_DIR = os.path.dirname(os.path.abspath(__file__))
print("TEST_DIR=", TEST_DIR)
os.environ[STRUCTURES_PATH] = os.path.join(TEST_DIR, "structures")


def test_run_length_encoding():
    assert run_length_encoding(["Al", "Al", "Cu"]) == "Al2Cu"
    assert run_length_encoding(["Al", "Cu"]) == "AlCu"
    assert run_length_encoding(["Cu", "Al", "Al"]) == "CuAl2"


def test_ht_yaml_unary():  # Wykoff A1B2
    yamlfile = os.path.join(TEST_DIR, "htp2.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "atoms")
    assert (
        res["name"][0]
        == "Al/unaries/bulk/A15/shaken/A15.shakesmallsuper2corrected.1/atoms/Al64/"
    )
    # symbs = [str(struct.symbols) for struct in res['structure']]
    # print(symbs)
    # assert symbs == ['Al64']
    # assert len(res['name']) == 1


def test_ht_yaml_binary_mixed():  # Wykoff A1B2
    yamlfile = os.path.join(TEST_DIR, "htp4.yaml")
    # res = get_dict_from_yaml(yamlfile, 'combinations_positions') # This was commented out
    res = generate_ht_pipelines_setup(yamlfile, "mixed")

    for x in res["name"]:
        print(x)

    assert (
        res["name"][0]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/mixed/Al2Cu/"
    )
    assert (
        res["name"][1]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/mixed/AlCu2/"
    )
    assert (
        res["name"][2]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/mixed/CuAl2/"
    )
    assert (
        res["name"][3]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/mixed/CuAlCu/"
    )
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    assert symbs == ["Al2Cu", "AlCu2", "CuAl2", "CuAlCu"]
    assert len(res["name"]) == 4


def test_ht_yaml_binary_atoms():
    yamlfile = os.path.join(TEST_DIR, "htp4.yaml")
    # res = get_dict_from_yaml(yamlfile, 'permutations_positions') # This was commented out
    res = generate_ht_pipelines_setup(yamlfile, "atoms")
    assert (
        res["name"][0]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/Al2Cu/"
    )
    assert (
        res["name"][1]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/AlCuAl/"
    )
    assert (
        res["name"][2]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/AlCu2/"
    )
    assert (
        res["name"][3]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/CuAl2/"
    )
    assert (
        res["name"][4]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/CuAlCu/"
    )
    assert (
        res["name"][5]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/Cu2Al/"
    )
    assert len(res["name"]) == 6
    symbs = [str(struct.symbols) for struct in res["structure"]]
    assert symbs == ["Al2Cu", "AlCuAl", "AlCu2", "CuAl2", "CuAlCu", "Cu2Al"]


def test_ht_yaml_binary_wyckoff():
    yamlfile = os.path.join(TEST_DIR, "htp4.yaml")
    # res = get_dict_from_yaml(yamlfile, 'Wyckoff') # This was commented out
    res = generate_ht_pipelines_setup(yamlfile, "wyckoff")
    assert (
        res["name"][0]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/wyckoff/AlCu2/"
    )
    assert (
        res["name"][1]
        == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/wyckoff/CuAl2/"
    )
    assert len(res["name"]) == 2
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    assert symbs == ["AlCu2", "CuAl2"]


def test_ht_yaml_binary_cfg():
    yamlfile = os.path.join(TEST_DIR, "htp4.yaml")
    # res = get_dict_from_yaml(yamlfile, 'Ele') # This was commented out
    res = generate_ht_pipelines_setup(yamlfile, "cfg")

    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)

    assert (
        res["name"][0] == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/cfg/AlCu2/"
    )
    assert (
        res["name"][1] == "AlCu/binaries/bulk/MoPt2/reference/gen_182_MoPt2/cfg/CuAl2/"
    )
    assert len(res["name"]) == 2

    assert symbs == ["AlCu2", "CuAl2"]


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_binary(permutation_type):
    print("PERMUTATION TYPE: ", permutation_type)
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(p, "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg")
    print("fname=", fname)
    list_permutations = get_list_permutations(permutation_type, ["Al", "Ni"], fname)

    print(list_permutations)
    for ordsymb in list_permutations:
        structure = (
            create_structure(fname, ordsymb)
            if permutation_type == PERM_TYPE_CFG
            else create_structure(fname, chemical_symbs=ordsymb)
        )
        print("PERM: {} ATOMS: {}".format(ordsymb, str(structure.symbols)))


def test_ht_yaml_ternary_atoms():
    yamlfile = os.path.join(TEST_DIR, "htp5.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "atoms")

    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)

    assert (
        res["name"][0]
        == "AlNiZn/ternaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/AlNiZn/"
    )
    assert (
        res["name"][1]
        == "AlNiZn/ternaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/AlZnNi/"
    )
    assert (
        res["name"][2]
        == "AlNiZn/ternaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/NiAlZn/"
    )
    assert (
        res["name"][3]
        == "AlNiZn/ternaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/NiZnAl/"
    )
    assert (
        res["name"][4]
        == "AlNiZn/ternaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/ZnAlNi/"
    )
    assert (
        res["name"][5]
        == "AlNiZn/ternaries/bulk/MoPt2/reference/gen_182_MoPt2/atoms/ZnNiAl/"
    )
    assert len(res["name"]) == 6

    assert symbs == ["AlNiZn", "AlZnNi", "NiAlZn", "NiZnAl", "ZnAlNi", "ZnNiAl"]


# @pytest.mark.parametrize("permutation_type", ["atoms",
#                                               "mixed",
#                                               "wyckoff",
#                                               "cfg"
#                                               ])
# def test_get_list_permutations_ternary(permutation_type):
#     print("PERMUTATION TYPE: ", permutation_type)
#     p = os.environ[STRUCTURES_PATH]
#     fname=os.path.join(p, "binaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg")
#     print("fname=",fname)
#
#     list_permutations = get_list_permutations(permutation_type, ["Al", "Ni", "Zn"],
#                                   fname)


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_ternary_182_MoPt2(permutation_type):
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(p, "ternaries/bulk/MoPt2/reference/gen_182_MoPt2.cfg")
    print("fname=", fname)
    if permutation_type == PERM_TYPE_WYCKOFF:
        assert AssertionError
    else:
        list_permutations = get_list_permutations(
            permutation_type, ["Al", "Ni", "Zn"], fname
        )
        print("list_permutations=", list_permutations)
        for ordsymb in list_permutations:
            structure = (
                create_structure(fname, ordsymb)
                if permutation_type == PERM_TYPE_CFG
                else create_structure(fname, chemical_symbs=ordsymb)
            )
            print("PERM: {} ATOMS: {}".format(ordsymb, str(structure.symbols)))


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_ternary_23_D0_22_Al3Ti(permutation_type):
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(
        p, "ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti.cfg"
    )
    print("fname=", fname)

    if permutation_type == PERM_TYPE_WYCKOFF:
        assert AssertionError
    else:
        list_permutations = get_list_permutations(
            permutation_type, ["Al", "Ni", "Zn"], fname
        )
        print("list_permutations=", list_permutations, len(list_permutations))
        for ordsymb in list_permutations:
            structure = (
                create_structure(fname, ordsymb)
                if permutation_type == PERM_TYPE_CFG
                else create_structure(fname, chemical_symbs=ordsymb)
            )
            print("PERM: {} ATOMS: {}".format(ordsymb, str(structure.symbols)))


def test_ht_yaml_ternary_atoms_no_repeated_wyckoff_sites_atoms():
    yamlfile = os.path.join(TEST_DIR, "htp6.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "atoms")

    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)

    assert (
        res["name"][0]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/Al2NiZn/"
    )
    assert (
        res["name"][1]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/Al2ZnNi/"
    )
    assert (
        res["name"][2]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlNiAlZn/"
    )
    assert (
        res["name"][3]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlNi2Zn/"
    )
    assert (
        res["name"][4]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlNiZnAl/"
    )
    assert (
        res["name"][5]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlNiZnNi/"
    )
    assert (
        res["name"][6]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlNiZn2/"
    )
    assert (
        res["name"][7]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlZnAlNi/"
    )
    assert (
        res["name"][8]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlZnNiAl/"
    )
    assert (
        res["name"][9]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlZnNi2/"
    )
    assert (
        res["name"][10]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlZnNiZn/"
    )
    assert (
        res["name"][11]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/atoms/AlZn2Ni/"
    )

    assert symbs == [
        "Al2NiZn",
        "Al2ZnNi",
        "AlNiAlZn",
        "AlNi2Zn",
        "AlNiZnAl",
        "AlNiZnNi",
        "AlNiZn2",
        "AlZnAlNi",
        "AlZnNiAl",
        "AlZnNi2",
        "AlZnNiZn",
        "AlZn2Ni",
        "NiAl2Zn",
        "NiAlNiZn",
        "NiAlZnAl",
        "NiAlZnNi",
        "NiAlZn2",
        "Ni2AlZn",
        "Ni2ZnAl",
        "NiZnAl2",
        "NiZnAlNi",
        "NiZnAlZn",
        "NiZnNiAl",
        "NiZn2Al",
        "ZnAl2Ni",
        "ZnAlNiAl",
        "ZnAlNi2",
        "ZnAlNiZn",
        "ZnAlZnNi",
        "ZnNiAl2",
        "ZnNiAlNi",
        "ZnNiAlZn",
        "ZnNi2Al",
        "ZnNiZnAl",
        "Zn2AlNi",
        "Zn2NiAl",
    ]


def test_ht_yaml_ternary_atoms_no_repeated_wyckoff_sites_mixed():
    yamlfile = os.path.join(TEST_DIR, "htp6.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "mixed")

    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)

    assert (
        res["name"][0]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/Al2NiZn/"
    )
    assert (
        res["name"][1]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/Al2ZnNi/"
    )
    assert (
        res["name"][2]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlNiAlZn/"
    )
    assert (
        res["name"][3]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlNi2Zn/"
    )
    assert (
        res["name"][4]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlNiZnAl/"
    )
    assert (
        res["name"][5]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlNiZnNi/"
    )
    assert (
        res["name"][6]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlNiZn2/"
    )
    assert (
        res["name"][7]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlZnAlNi/"
    )
    assert (
        res["name"][8]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlZnNiAl/"
    )
    assert (
        res["name"][9]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlZnNi2/"
    )
    assert (
        res["name"][10]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlZnNiZn/"
    )
    assert (
        res["name"][11]
        == "AlNiZn/ternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti/mixed/AlZn2Ni/"
    )

    assert symbs == [
        "Al2NiZn",
        "Al2ZnNi",
        "AlNiAlZn",
        "AlNi2Zn",
        "AlNiZnAl",
        "AlNiZnNi",
        "AlNiZn2",
        "AlZnAlNi",
        "AlZnNiAl",
        "AlZnNi2",
        "AlZnNiZn",
        "AlZn2Ni",
        "NiAl2Zn",
        "NiAlNiZn",
        "NiAlZnAl",
        "NiAlZnNi",
        "NiAlZn2",
        "Ni2AlZn",
        "Ni2ZnAl",
        "NiZnAl2",
        "NiZnAlNi",
        "NiZnAlZn",
        "NiZnNiAl",
        "NiZn2Al",
        "ZnAl2Ni",
        "ZnAlNiAl",
        "ZnAlNi2",
        "ZnAlNiZn",
        "ZnAlZnNi",
        "ZnNiAl2",
        "ZnNiAlNi",
        "ZnNiAlZn",
        "ZnNi2Al",
        "ZnNiZnAl",
        "Zn2AlNi",
        "Zn2NiAl",
    ]


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_ternary_three_inequivalent_wyckoff_sites(
    permutation_type,
):
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(p, "ternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi.cfg")
    print("fname=", fname)

    list_permutations = get_list_permutations(
        permutation_type, ["Al", "Ni", "Zn"], fname
    )
    print("list_permutations=", list_permutations, len(list_permutations))
    for ordsymb in list_permutations:
        structure = (
            create_structure(fname, ordsymb)
            if permutation_type == PERM_TYPE_CFG
            else create_structure(fname, chemical_symbs=ordsymb)
        )
        print("PERM: {} ATOMS: {}".format(ordsymb, str(structure.symbols)))


def test_get_list_permutations_ternary_three_inequivalent_wyckoff_sites_wyckoff():
    yamlfile = os.path.join(TEST_DIR, "htp7.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "wyckoff")

    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)

    assert (
        res["name"][0]
        == "AlNiZn/ternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi/wyckoff/Al2NiZn/"
    )
    assert (
        res["name"][1]
        == "AlNiZn/ternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi/wyckoff/Al2ZnNi/"
    )
    assert (
        res["name"][2]
        == "AlNiZn/ternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi/wyckoff/Ni2AlZn/"
    )
    assert (
        res["name"][3]
        == "AlNiZn/ternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi/wyckoff/Ni2ZnAl/"
    )
    assert (
        res["name"][4]
        == "AlNiZn/ternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi/wyckoff/Zn2AlNi/"
    )
    assert (
        res["name"][5]
        == "AlNiZn/ternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi/wyckoff/Zn2NiAl/"
    )

    assert symbs == ["Al2NiZn", "Al2ZnNi", "Ni2AlZn", "Ni2ZnAl", "Zn2AlNi", "Zn2NiAl"]


def test_get_list_permutations_ternary_three_inequivalent_wyckoff_sites_mixed():
    yamlfile = os.path.join(TEST_DIR, "htp7.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "mixed")

    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)

    assert symbs == [
        "Al2NiZn",
        "Al2ZnNi",
        "AlNiAlZn",
        "AlNi2Zn",
        "AlNiZnAl",
        "AlNiZnNi",
        "AlNiZn2",
        "AlZnAlNi",
        "AlZnNiAl",
        "AlZnNi2",
        "AlZnNiZn",
        "AlZn2Ni",
        "Ni2AlZn",
        "Ni2ZnAl",
        "NiZnAl2",
        "NiZnAlNi",
        "NiZnAlZn",
        "NiZnNiAl",
        "NiZn2Al",
        "Zn2AlNi",
        "Zn2NiAl",
    ]


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_quaternary(permutation_type):
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(
        p, "quaternaries/bulk/Al3Ti-D0_22/reference/gen_23_D0_22_Al3Ti.cfg"
    )
    print("fname=", fname)

    list_permutations = get_list_permutations(
        permutation_type, ["Al", "Ni", "Cu", "Zn"], fname
    )
    print("list_permutations=", list_permutations, len(list_permutations))
    for ordsymb in list_permutations:
        structure = (
            create_structure(fname, ordsymb)
            if permutation_type == PERM_TYPE_CFG
            else create_structure(fname, chemical_symbs=ordsymb)
        )
        print("PERM: {} ATOMS: {}".format(ordsymb, str(structure.symbols)))


def test_get_list_permutations_quaternary_inequivalent_wyckoff():
    yamlfile = os.path.join(TEST_DIR, "htp8.yaml")
    permutations_types = ["atoms", "mixed", "wyckoff", "cfg"]
    for type in permutations_types:
        res = generate_ht_pipelines_setup(yamlfile, type)
        print(type)
        symbs = [str(struct.symbols) for struct in res["structure"]]
        print(symbs)
        symbs.sort()
        assert symbs == [
            "AlCuNiZn",
            "AlCuZnNi",
            "AlNiCuZn",
            "AlNiZnCu",
            "AlZnCuNi",
            "AlZnNiCu",
            "CuAlNiZn",
            "CuAlZnNi",
            "CuNiAlZn",
            "CuNiZnAl",
            "CuZnAlNi",
            "CuZnNiAl",
            "NiAlCuZn",
            "NiAlZnCu",
            "NiCuAlZn",
            "NiCuZnAl",
            "NiZnAlCu",
            "NiZnCuAl",
            "ZnAlCuNi",
            "ZnAlNiCu",
            "ZnCuAlNi",
            "ZnCuNiAl",
            "ZnNiAlCu",
            "ZnNiCuAl",
        ]


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_quaternary_4atoms(
    permutation_type,
):  # ['a', 'a', 'c', 'b']
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(
        p, "quaternaries/bulk/AsNi-B8_1/reference/gen_34_B8_1_AsNi.cfg"
    )
    print("fname=", fname)

    if permutation_type == PERM_TYPE_WYCKOFF:
        assert AssertionError
    else:
        list_permutations = get_list_permutations(
            permutation_type, ["Al", "Ni", "Cu", "Zn"], fname
        )
        print("list_permutations=", list_permutations, len(list_permutations))
        for ordsymb in list_permutations:
            structure = (
                create_structure(fname, ordsymb)
                if permutation_type == PERM_TYPE_CFG
                else create_structure(fname, chemical_symbs=ordsymb)
            )
            print("PERM: {} ATOMS: {}".format(ordsymb, str(structure.symbols)))


def test_get_list_permutations_quaternary_three_wyckoff_sites():
    yamlfile = os.path.join(TEST_DIR, "htp9.yaml")
    permutations_types = ["atoms", "mixed", "wyckoff", "cfg"]
    for type in permutations_types:
        if type == PERM_TYPE_WYCKOFF:
            assert AssertionError
        else:
            res = generate_ht_pipelines_setup(yamlfile, type)
            symbs = [str(struct.symbols) for struct in res["structure"]]
            print(type, symbs)
            symbs.sort()
            if type == PERM_TYPE_MIXED_WYCKOFF_ATOMS:
                assert symbs == [
                    "AlCuNiZn",
                    "AlCuZnNi",
                    "AlNiCuZn",
                    "AlNiZnCu",
                    "AlZnCuNi",
                    "AlZnNiCu",
                    "CuNiAlZn",
                    "CuNiZnAl",
                    "CuZnAlNi",
                    "CuZnNiAl",
                    "NiZnAlCu",
                    "NiZnCuAl",
                ]
            elif type == PERM_TYPE_CFG or type == PERM_TYPE_ATOMS:
                assert symbs == [
                    "AlCuNiZn",
                    "AlCuZnNi",
                    "AlNiCuZn",
                    "AlNiZnCu",
                    "AlZnCuNi",
                    "AlZnNiCu",
                    "CuAlNiZn",
                    "CuAlZnNi",
                    "CuNiAlZn",
                    "CuNiZnAl",
                    "CuZnAlNi",
                    "CuZnNiAl",
                    "NiAlCuZn",
                    "NiAlZnCu",
                    "NiCuAlZn",
                    "NiCuZnAl",
                    "NiZnAlCu",
                    "NiZnCuAl",
                    "ZnAlCuNi",
                    "ZnAlNiCu",
                    "ZnCuAlNi",
                    "ZnCuNiAl",
                    "ZnNiAlCu",
                    "ZnNiCuAl",
                ]


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_quinary(permutation_type):  # ['a', 'b', 'b', 'a', 'a']
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(p, "quinaries/bulk/moreprototypesYury/gen_9_D1_3_Al4Ba.cfg")
    print("fname=", fname)

    if permutation_type == PERM_TYPE_WYCKOFF:
        assert AssertionError
    else:
        list_permutations = get_list_permutations(
            permutation_type, ["Al", "Ni", "Cu", "Zr", "Zn"], fname
        )
        for ordsymb in list_permutations:
            structure = (
                create_structure(fname, ordsymb)
                if permutation_type == PERM_TYPE_CFG
                else create_structure(fname, chemical_symbs=ordsymb)
            )
            print("PERM: {} ATOMS: {}".format(ordsymb, str(structure.symbols)))


def test_get_list_permutations_quinary_mixed():
    yamlfile = os.path.join(TEST_DIR, "htp10.yaml")
    permutations_types = ["mixed"]
    for type in permutations_types:
        if type == PERM_TYPE_WYCKOFF:
            assert AssertionError
        else:
            res = generate_ht_pipelines_setup(yamlfile, type)
            symbs = [str(struct.symbols) for struct in res["structure"]]
            print(type, symbs)
            # symbs.sort()
            assert symbs == [
                "AlCuNiZnZr",
                "AlCuZnNiZr",
                "AlCuZrNiZn",
                "AlNiZnCuZr",
                "AlNiZrCuZn",
                "AlZnZrCuNi",
                "CuNiZnAlZr",
                "CuNiZrAlZn",
                "CuZnZrAlNi",
                "NiZnZrAlCu",
            ]


@pytest.mark.parametrize("permutation_type", ["atoms", "mixed", "wyckoff", "cfg"])
def test_get_list_permutations_quinary_three_wyckoff_sites(
    permutation_type,
):  # ['b', 'c', 'a', 'b', 'c']
    p = os.environ[STRUCTURES_PATH]
    fname = os.path.join(
        p, "quinaries/bulk/moreprototypesYury/gen_5731_D5_2_La2O3__D52D52D5__2__.cfg"
    )
    print("fname=", fname)

    if permutation_type == PERM_TYPE_WYCKOFF:
        assert AssertionError
    else:
        list_permutations = get_list_permutations(
            permutation_type, ["Al", "Ni", "Cu", "Zr", "Zn"], fname
        )
        print(permutation_type, list_permutations)
        for ordsymb in list_permutations:
            structure = (
                create_structure(fname, ordsymb)
                if permutation_type == PERM_TYPE_CFG
                else create_structure(fname, chemical_symbs=ordsymb)
            )
            chemical_symbols = list(structure.symbols)
            if permutation_type == PERM_TYPE_CFG:
                print(
                    "CHEMICAL SYMBOLS: {}, ATOMS: {}".format(
                        chemical_symbols, str(structure.symbols)
                    )
                )
            else:
                print("PERM: {}, ATOMS: {}".format(ordsymb, str(structure.symbols)))


def test_get_list_permutations_quinary_three_wyckoff_sites_mixed():
    yamlfile = os.path.join(TEST_DIR, "htp11.yaml")
    permutations_types = ["mixed"]
    for type in permutations_types:
        res = generate_ht_pipelines_setup(yamlfile, type)
        symbs = [str(struct.symbols) for struct in res["structure"]]
        print(symbs)
        symbs.sort()
        assert symbs == [
            "AlNiZnZrCu",
            "AlNiZrZnCu",
            "AlZnCuZrNi",
            "AlZnNiZrCu",
            "AlZnZrCuNi",
            "AlZnZrNiCu",
            "AlZrCuNiZn",
            "AlZrCuZnNi",
            "AlZrNiCuZn",
            "AlZrNiZnCu",
            "AlZrZnCuNi",
            "AlZrZnNiCu",
            "CuNiZnZrAl",
            "CuNiZrZnAl",
            "CuZnNiZrAl",
            "CuZnZrAlNi",
            "CuZnZrNiAl",
            "CuZrNiAlZn",
            "CuZrNiZnAl",
            "CuZrZnAlNi",
            "CuZrZnNiAl",
            "NiCuZnZrAl",
            "NiCuZrZnAl",
            "NiZnZrAlCu",
            "NiZnZrCuAl",
            "NiZrZnAlCu",
            "NiZrZnCuAl",
            "ZnCuZrNiAl",
            "ZnNiZrAlCu",
            "ZnNiZrCuAl",
        ]


def test_get_list_permutations_perovskite():
    yamlfile = os.path.join(TEST_DIR, "htp12.yaml")
    permutations_types = ["constrained"]
    for type in permutations_types:
        res = generate_ht_pipelines_setup(yamlfile, type)
        symbs = [str(struct.symbols) for struct in res["structure"]]
        print(symbs)
        symbs.sort()
        assert symbs == ["BaSn2O", "BaTi2O", "CaSn2O", "CaTi2O", "SrSn2O", "SrTi2O"]

        assert (
            res["name"][0]
            == "BaOTi/ternaries/bulk/ABO3/reference/ABO3/constrained/BaTi2O/"
        )
        assert (
            res["name"][1]
            == "BaOSn/ternaries/bulk/ABO3/reference/ABO3/constrained/BaSn2O/"
        )
        assert (
            res["name"][2]
            == "OSnSr/ternaries/bulk/ABO3/reference/ABO3/constrained/SrSn2O/"
        )
        assert (
            res["name"][3]
            == "CaOTi/ternaries/bulk/ABO3/reference/ABO3/constrained/CaTi2O/"
        )
        assert (
            res["name"][4]
            == "OSrTi/ternaries/bulk/ABO3/reference/ABO3/constrained/SrTi2O/"
        )
        assert (
            res["name"][5]
            == "CaOSn/ternaries/bulk/ABO3/reference/ABO3/constrained/CaSn2O/"
        )


def test_get_list_permutations_constrained():
    yamlfile = os.path.join(TEST_DIR, "htp12_a.yaml")

    res = generate_ht_pipelines_setup(yamlfile)
    print(res)
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    symbs.sort()
    assert symbs == ["HPd2"]


def test_get_list_permutations_random_perovskites_from_folder():
    yamlfile = os.path.join(TEST_DIR, "htp13.yaml")
    permutations_types = ["constrained"]
    for type in permutations_types:
        res = generate_ht_pipelines_setup(yamlfile, type)
        print(res)
        symbs = [str(struct.symbols) for struct in res["structure"]]
        # symbs.sort()
        print(symbs)

        for x in res["name"]:
            print(x)

        assert len(symbs) == 6

        assert len(res["name"]) == 6
        assert len(res["pipeline"]) == 6


def test_ht_yaml_ternary_permtype_from_yaml():
    yamlfile = os.path.join(TEST_DIR, "htp14.yaml")
    res = generate_ht_pipelines_setup(yamlfile)
    print(res["name"])
    for name in res["name"]:
        assert ("atoms" in name and "Al3Ti-D0_22" in name) or (
            "mixed" in name and "MoPt2" in name
        ), name


def test_ht_steps_params():
    yamlfile = os.path.join(TEST_DIR, "htp2_steps_params.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "atoms")
    print(res)
    pipe = res["pipeline"][0]
    print(pipe.steps)
    assert len(pipe.steps) == 3
    assert (
        res["name"][0]
        == "Al/unaries/bulk/A15/shaken/A15.shakesmallsuper2corrected.1/atoms/Al64/"
    )
    enn_coarse = pipe.steps["Enn-coarse"]
    print(enn_coarse)
    # print(enn_coarse.job_options.options["job_options"])
    assert enn_coarse.nn_distance_range[0] == 2.0
    assert enn_coarse.nn_distance_range[-1] == 7.0
    assert enn_coarse.fix_kmesh == False


def test_perovskite_from_Matous():
    yamlfile = os.path.join(TEST_DIR, "htp15.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "constrained")
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    assert symbs == ["BaTiO3", "BaSnO3", "CaTiO3", "CaSnO3", "SrTiO3", "SrSnO3"]


def test_create_structure_normal_cfg():
    at = create_structure(os.path.join(TEST_DIR, "structures/melt_Ag.1000000.cfg"))
    print(at)
    assert len(at) == 108
    assert at.get_chemical_formula() == "Ag108"


def test_ht_normal_structures_none_composition():
    res = generate_ht_pipelines_setup(
        {"melt_Ag.1000000.cfg": ["static"], "melt_Ag.2000000.cfg": ["static"]}
    )
    print(res)
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    assert symbs == ["Ag108", "Ag108"]
    assert res["name"] == [
        "Ag/melt_Ag.1000000/cfg/Ag108/",
        "Ag/melt_Ag.2000000/cfg/Ag108/",
    ]


def test_ht_normal_structures():
    res = generate_ht_pipelines_setup(
        {
            "composition": ["Al"],
            "melt_Ag.1000000.cfg": ["static"],
            "melt_Ag.2000000.cfg": ["static"],
        }
    )
    print(res)
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    assert symbs == ["Ag108", "Ag108"]
    assert res["name"] == [
        "Ag/melt_Ag.1000000/cfg/Ag108/",
        "Ag/melt_Ag.2000000/cfg/Ag108/",
    ]


def test_n_elements_larger_n_prototype_elements():
    yamlfile = os.path.join(TEST_DIR, "htp16.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "mixed")
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    assert symbs == ["Al64", "Ni64", "Cu64", "Zr64", "Zn64"]


def test_ht_normal_structures_Ag_prototype():
    yamlfile = os.path.join(TEST_DIR, "htp17.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "mixed")
    symbs = [str(struct.symbols) for struct in res["structure"]]
    print(symbs)
    assert symbs == ["Al108", "Ni108", "Cu108", "Zr108", "Zn108"]


def test_ht_structures_with_vacuum():
    yamlfile = os.path.join(TEST_DIR, "htp18.yaml")
    res = generate_ht_pipelines_setup(yamlfile, "atoms")
    symbs = [str(struct.symbols) for struct in res["structure"]]
    structs = [ele for struct in res["structure"] for ele in list(struct.get_pbc())]
    print(symbs)
    assert np.asarray(structs).all() == False
    assert symbs == ["Al108", "Ni108", "Cu108", "Zr108", "Zn108"]


def test_k_mesh_for_cluster_vasp():
    yamlfile = os.path.join(TEST_DIR, "htp19.yaml")
    calculator_setup_fname = os.path.join(TEST_DIR, "calculator_vasp.yaml")
    calc, enforce_pbc = create_calc(calculator_setup_fname, if_enforce_pbc=True)
    res = generate_ht_pipelines_setup(yamlfile, "atoms", enforce_pbc=enforce_pbc)
    calc_list = []
    for structure, pipe in zip(res["structure"], res["pipeline"]):
        cur_work_dir = TEST_DIR
        initialize_pipe(pipe, cur_work_dir, calc, structure)
        new_calc = pipe.engine.calculator
        calc_list.append(new_calc)
    assert calc_list[0].kmesh_spacing is None
    assert calc_list[1].kmesh_spacing == 0.125


def test_k_mesh_for_cluster_aims():
    yamlfile = os.path.join(TEST_DIR, "htp19.yaml")
    calculator_setup_fname = os.path.join(TEST_DIR, "calculator_aims.yaml")
    calc, enforce_pbc = create_calc(calculator_setup_fname, if_enforce_pbc=True)
    res = generate_ht_pipelines_setup(yamlfile, "atoms", enforce_pbc=enforce_pbc)
    calc_list = []
    for structure, pipe in zip(res["structure"], res["pipeline"]):
        cur_work_dir = TEST_DIR
        initialize_pipe(pipe, cur_work_dir, calc, structure)
        new_calc = pipe.engine.calculator
        calc_list.append(new_calc)
    assert calc_list[0].kmesh_spacing == 0.125
    assert calc_list[1].kmesh_spacing == 0.125


def test_ht_parameterized():
    yamlfile = os.path.join(TEST_DIR, "ht_parameterized.yaml")
    calculator_setup_fname = os.path.join(TEST_DIR, "calculator_vasp.yaml")
    calc, enforce_pbc = create_calc(calculator_setup_fname, if_enforce_pbc=True)

    res = generate_ht_pipelines_setup(yamlfile, "atoms", enforce_pbc=enforce_pbc)
    calc_list = []
    for structure, pipe in zip(res["structure"], res["pipeline"]):
        print(pipe)
        
    for structure, pipe in zip(res["structure"], res["pipeline"]):
        print(pipe)
        
        enn_fine = pipe.steps["Enn-fine"]
        assert enn_fine.fix_kmesh == False
        assert enn_fine.nn_distance_range == [1.2, 2.0]
        assert enn_fine.nn_distance_step == 0.05

        enn_coarse = pipe.steps["Enn-coarse"]
        assert enn_coarse.fix_kmesh == False
        assert enn_coarse.nn_distance_range == [0.6, 4.6]
        assert enn_coarse.nn_distance_step == 0.2


def test_TCP_sublattice():
    yamlfile = os.path.join(TEST_DIR, "htp_TCP.yaml")
    # calculator_setup_fname = os.path.join(TEST_DIR, 'calculator_vasp.yaml')
    res = generate_ht_pipelines_setup(yamlfile)

    for n, s in zip(res["name"], res["structure"]):
        print(n, ":", s)

    assert len(res["structure"]) == 1

    print(res["name"])
    res_name_ref = [
        "MoNb/binaries/bulk/C14/by_sites/C14_AAB/constrained/Mo8Nb4/",
    ]
    assert sorted(res["name"]) == sorted(res_name_ref)


def test_randomdeformation():
    yamlfile = os.path.join(TEST_DIR, "htp_randomdeformation.yaml")

    res = generate_ht_pipelines_setup(yamlfile)

    for n, s in zip(res["name"], res["structure"]):
        print(n, ":", s)

    assert len(res["structure"]) == 2

    print(res["name"])
    res_name_ref = [
        "MoNb/binaries/bulk/MoPt2/reference/gen_182_MoPt2/cfg/MoNb2/",
        "MoNb/binaries/bulk/MoPt2/reference/gen_182_MoPt2/cfg/NbMo2/",
    ]
    assert sorted(res["name"]) == sorted(res_name_ref)
