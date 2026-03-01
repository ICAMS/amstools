def cif2cfg(filename, metadata):
    """wrapper function to convert cif to cfg."""
    import logging
    from ase import io
    from amstools.resources.cfgio import create_structure, write_structure

    try:
        strucname = filename.split(".cif")[0]
        metadata["name"] = strucname
        at0 = io.read(filename)
        write_structure(
            strucname + ".cfg", at0, save_exact=False, metadata_dict=metadata
        )
        print("converted ", filename)
    except (IOError, OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"error converting {filename}: {e}")
        logging.error(f"Failed to convert {filename}: {e}", exc_info=True)
    return


"""usage as system call: python cif2cfg somename.cif"""
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
    cif2cfg(filename, metadata)
