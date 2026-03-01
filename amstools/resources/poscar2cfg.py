def poscar2cfg(filename, metadata):
    """wrapper function to convert POSCAR to cfg."""
    import logging
    from ase import io
    from amstools.resources.cfgio import create_structure, write_structure

    try:
        strucname = filename.split("POSCAR.")[1]
        metadata["name"] = strucname
        at0 = io.read("POSCAR." + str(strucname))
        write_structure(
            strucname + ".cfg", at0, save_exact=False, metadata_dict=metadata
        )
        print("converted ", filename)
    except (IOError, OSError, ValueError, KeyError, IndexError, RuntimeError) as e:
        print(f"error converting {filename}: {e}")
        logging.error(f"Failed to convert {filename}: {e}", exc_info=True)
    return


"""usage as system call: python poscar2cfg POSCAR.somename"""
if __name__ == "__main__":
    import sys
    from datetime import date

    filename = sys.argv[1]
    metadata = {
        "author": "AMS",
        "date": str(date.today()),
        "description": "elemental bulk structure",
        "purpose": "reference structures",
        "alternative names": "",
        "publication:": "",
        "source": "",
        "scaling_length": "1.0",
        "periodic": "True",
    }
    poscar2cfg(filename, metadata)
