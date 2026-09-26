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
