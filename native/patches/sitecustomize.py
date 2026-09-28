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

_fp8_target_head = os.environ.get("GB10_FP8_TARGET_HEAD", "0")
if os.environ.get("GB10_FP8_SIDE") == "1" or _fp8_target_head == "load":
    import gb10_fp8_side
    import gb10_ple_mmap

    gb10_ple_mmap.install_import_hook(gb10_fp8_side.TARGET_MODULE, gb10_fp8_side.apply_target)
    if os.environ.get("GB10_FP8_SIDE") == "1":
        gb10_ple_mmap.install_import_hook(gb10_fp8_side.MTP_MODULE, gb10_fp8_side.apply_mtp)

if os.environ.get("GB10_FP8_DRAFT_HEAD") == "1" or _fp8_target_head not in ("0", "load"):
    import gb10_fp8_side
    import gb10_ple_mmap

    for _mod in gb10_fp8_side.SPEC_MODULES:
        gb10_ple_mmap.install_import_hook(_mod, gb10_fp8_side.apply_spec)
