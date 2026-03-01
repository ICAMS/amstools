import os
from setuptools import setup

import versioneer


def package_files(directory):
    """Recursively find all files in a directory."""
    paths = []
    for path, directories, filenames in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join("..", path, filename))
    return paths

setup(
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    package_data={
        "": package_files("amstools/resources/data"),
        "amstools.highthroughput.data": package_files("amstools/highthroughput/data"),
        "amstools.highthroughput.webinterface.resources": package_files("amstools/highthroughput/webinterface/resources"),
    }
)
