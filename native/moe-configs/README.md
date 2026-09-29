Triton fused-MoE kernel configs tuned on a GB10 for the models served here.

SGLang looks for `configs/triton_<triton version>/E=<experts>,N=<size>,device_name=NVIDIA_GB10,...json`
under `SGLANG_MOE_CONFIG_DIR`; a profile that wants these sets that variable to this directory
(models/ornith-1.5-35b.sh does when `configs/` exists). It replaces SGLang's own config
directory for that server, so only point models here whose MoE shapes all have a file.

Files come from SGLang's tuner (benchmark/kernels/fused_moe_triton in the source of the
installed version), run on the Spark; README.md, "Ornith 1.5 35B-A3B", has the steps.
