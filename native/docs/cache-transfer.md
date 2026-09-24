# Moving the weights to a new machine, offline

Bring up a new DGX Spark (or rebuild this one) without downloading the
checkpoints again, and without repeating the first boot's kernel compilation
where that can be avoided.

## What lives where

| Path | Contents | Carry it? |
|---|---|---|
| `MODEL_DIR`, `DRAFT_DIR` (default `/models/...`) | the checkpoints the server loads | **yes**, this is the ~32 GB that matters |
| `~/.cache/flashinfer`, `~/.triton` | JIT-compiled kernels and autotune results | optional: they are keyed to the exact FlashInfer / Triton / CUDA versions, so they help only a machine with the same lock file |
| `~/spark/venv` | SGLang and its dependencies | no: rebuild it with `./serve.sh install`, which installs the exact versions from `requirements/` |

## Pack

`hf download --local-dir` directories hold real files, so any copy works:

```bash
tar -C / -cpf /media/root/2TB/gb10-models.tar models/Qwen3.8-27B-FP8 models/Qwen3.8-27B-DFlash2
# optional, same lock file on both machines only:
# tar -C "$HOME" -cpf /media/root/2TB/gb10-kernels.tar .cache/flashinfer .triton
```

Keep each directory's `.cache/huggingface/` subdirectory: it records the
revision the files came from, which the server prints at startup.

**If `serve.sh` points into an HF cache** (`.../snapshots/<sha>`), the files
there are symlinks into `../../blobs/`. Either carry the whole
`models--<org>--<name>` directory with `tar`, `rsync -a` or `cp -a` (never
`cp -rL` on the whole thing, which doubles the size and breaks the layout),
or turn the snapshot into a plain directory on the way:

```bash
mkdir -p /models/Qwen3.8-27B-FP8
cp -rL ~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B-FP8/snapshots/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a/. \
       /models/Qwen3.8-27B-FP8/
```

A directory made that way has no revision record, so the server reports
`revision unknown`; keep a note of it. For the cache layout the target
filesystem must support symlinks: ext4/APFS/btrfs are fine, exFAT/FAT are not
(`stat -f -c %T /media/root/2TB` to check).

## Restore

```bash
sudo mkdir -p /models && sudo chown "$USER" /models
tar -C / -xpf /media/root/2TB/gb10-models.tar
./serve.sh install      # the venv
```

None of these checkpoints needs an `hf` token.

## Verify

The server's startup lines name each directory and its revision; compare
them with the table below before trusting a benchmark. With network,
re-running the `hf download --revision <sha> --local-dir <dir>` command from
the README is an integrity check: it downloads nothing when the directory is
complete and fills in what is missing when not.

`HF_OFFLINE=1` (the default in `serve.sh`) keeps the server off the Hub, so a
boot on a machine without network behaves the same as one with it.

## Revisions used around this repo

| Checkpoint | Revision | Note |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | `017b9c7a…` | the default target; the FP8 leg in `results/` |
| `RadixArk/Qwen3.8-27B-NVFP4` | `554ebba9…` | behind every earlier table in `results/` |
| `RadixArk/Qwen3.8-27B-NVFP4` | `319f741c…` | upstream default since 2026-08-22; the 2026-08-27 FP8-comparison leg |
| `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` | `69d21348…` | side measurement |
| `orcarouter/Qwen3.8-27B-Uncensored-FP8` | `0f3cdb83…` | not benchmarked here |
| `z-lab/Qwen3.8-27B-DFlash2` (draft) | `50307d4c…` | `incoai/Qwen3.8-27B-DFlash2` mirrors the same weights |
