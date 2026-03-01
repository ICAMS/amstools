import glob
import os

from importlib import resources


def get_data_path():
    return str(resources.files("amstools.resources") / "data")


def get_resources_filenames_by_glob(glob_pattern):
    """
    Return the list of filenames that fulfill `glob_pattern` from the amstools package resources
    :param glob_pattern: str, glob patter
    :return: list of str, list of filenames
    """
    resource_root_path = get_data_path()
    return sorted(glob.glob(os.path.join(resource_root_path, glob_pattern)))


def get_resource_single_filename(path_and_filename):
    fnames = get_resources_filenames_by_glob(path_and_filename)
    if len(fnames) == 0:
        raise ValueError(
            "Couldn't find filename {}. Provide full path and exact name, please".format(
                path_and_filename
            )
        )
    elif len(fnames) > 1:
        raise ValueError(
            "More than one file names ({}) are found. Provide full path and exact name, please".format(
                len(fnames)
            )
        )
    elif len(fnames) == 1:
        return fnames[0]
