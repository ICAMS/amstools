#!/usr/bin/env python
import os
import sys
import filecmp
import glob


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    from custodian.custodian import Custodian
    from custodian.vasp.jobs import VaspJob
    from custodian.vasp.handlers import (
        UnconvergedErrorHandler,
        NonConvergingErrorHandler,
        MeshSymmetryErrorHandler,
        DriftErrorHandler,
        VaspErrorHandler,
    )

    handlers = [
        VaspErrorHandler(),
        UnconvergedErrorHandler(),
        NonConvergingErrorHandler(),
        # removed due to undesirable behaviour regarding KPOINTS mesh re-sampling
        # MeshSymmetryErrorHandler(),
        DriftErrorHandler(),
    ]

    jobs = VaspJob(sys.argv[1:], auto_npar=False, auto_gamma=True)
    c = Custodian(handlers, [jobs], max_errors=3, terminate_on_nonzero_returncode=False)
    try:
        c.run()
    finally:
        # remove .orig files if they are identical
        orig_files = glob.glob("*.orig")
        for orig_fname in orig_files:
            fname = orig_fname.replace(".orig", "")
            if filecmp.cmp(fname, orig_fname, shallow=False):
                os.remove(orig_fname)


if __name__ == "__main__":
    main(sys.argv[1:])
