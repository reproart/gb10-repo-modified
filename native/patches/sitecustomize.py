"""Runs at the start of every Python process that has native/patches on its
PYTHONPATH, which a profile's model_env() sets up for the server only. SGLang
starts its scheduler in fresh (spawned) processes, so a patch has to be
installed at interpreter start to reach the process that loads the model.

It only installs import hooks, gated by environment variables; the patches
themselves run when SGLang imports the module they change. This file shadows
the system's sitecustomize (on Ubuntu, the apport crash hook) for the server
process only.
"""

import os

if os.environ.get("GB10_PLE_MMAP") == "1":
    import gb10_ple_mmap

    gb10_ple_mmap.install_import_hook()

if os.environ.get("GB10_MARLIN_LEAN") == "1":
    import gb10_marlin_lean
    import gb10_ple_mmap

    gb10_ple_mmap.install_import_hook(gb10_marlin_lean.TARGET_MODULE, gb10_marlin_lean.apply)

if os.environ.get("GB10_FP8_SIDE") == "1":
    import gb10_fp8_side
    import gb10_ple_mmap

    gb10_ple_mmap.install_import_hook(gb10_fp8_side.TARGET_MODULE, gb10_fp8_side.apply_target)
    gb10_ple_mmap.install_import_hook(gb10_fp8_side.MTP_MODULE, gb10_fp8_side.apply_mtp)
