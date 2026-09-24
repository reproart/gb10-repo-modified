# Moving the model cache to a new machine, offline

Bring up a new DGX Spark (or rebuild this one) without re-downloading
~100 GB of checkpoints — and without repeating the one-off 77-minute FP8
autotune boot.

## What lives where

Three caches, three roles:

| Path | Contents | Why it matters |
|---|---|---|
| `~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface` | the HF cache `start.sh` mounts into the container (`$(pwd)`-relative) | **the one the server actually reads — and writes: a cold launch downloads straight into it** |
| `~/.cache/huggingface` | the `hf` CLI cache, where a bare `hf download` lands | **not used by standalone launches** — a model prefetched here is invisible to the container and gets silently re-downloaded. Keep it only for `hf` CLI work or a SparkStation deployment |
| `~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/triton` | flashinfer/triton autotune artifacts | without it, the first boots repeat kernel autotuning |

Standalone launches fetch their own weights: with no prefetch, the first boot
pulls the checkpoint into the mounted toolkit cache — convenient, but it hides
a ~30 GB transfer inside a "slow boot" (a 77-min FP8 boot was exactly this).
To control the timing, prefetch into the *mounted* cache:

```bash
HF_HOME=~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface \
  ~/spark/venv/bin/hf download orcarouter/Qwen3.8-27B-Uncensored-FP8 \
  --revision 0f3cdb83820a8190ffedaef5b29cf4a635e49b4d
```

Survey before packing — `-L` follows the snapshot symlinks, without it you
measure the symlink stubs:

```bash
du -shL ~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface/hub/models--* 2>/dev/null
du -shL ~/.cache/huggingface/hub/models--* 2>/dev/null
```

If the same model appears in both caches, that is the silent re-download at
work; pack one copy and reclaim the other.

## Pack

```bash
tar -C "$HOME" -cpf /media/root/2TB/gb10-caches.tar \
    spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache
# optional: the CLI/SparkStation cache — add only if you use those
# tar -C "$HOME" -rpf /media/root/2TB/gb10-caches.tar .cache/huggingface
```

- `tar`, `rsync -a`, `cp -a` all preserve the cache's **relative** symlinks
  (`snapshots/<sha>/* -> ../../blobs/<hash>`). Never dereference — `cp -rL`
  doubles the size and breaks the layout.
- The target filesystem must support symlinks: ext4/APFS/btrfs are fine,
  exFAT/FAT are not (`stat -f -c %T /media/root/2TB` to check).
- Expect ~100+ GB with several checkpoints; the `du -shL` pass above is the
  real number.

## Restore

```bash
mkdir -p "$HOME"
tar -C "$HOME" -xpf /media/root/2TB/gb10-caches.tar
```

The caches are path-independent internally, but `start.sh` mounts
`$(pwd)/.cache/...`, so the toolkit directory must live where you launch it
from. None of these checkpoints needs an `hf` token.

## Verify — cheap first, then full

With network: an integrity check that downloads **nothing** when the copy is
complete — `hf` prints the snapshot path and exits.

```bash
~/spark/venv/bin/hf download RadixArk/Qwen3.8-27B-NVFP4 \
  --revision 554ebba9b5f1b79dc11246341960360e6ef05ef4
```

Offline: boot with **pinned revisions** and no network. The pin is what makes
offline resolution deterministic — `--revision <sha>` resolves straight to the
local snapshot, while an unpinned load consults the repo's default branch.
The draft model resolves the same way (its revision is pinned inside
`start-dflash.sh`), so it must be in the transferred cache — it is, if you
packed the cache whole.

While verifying, check provenance too: the revision you *served* is the one
whose snapshot directory you *see*, which is not always the one you think —
see the README trap about the 2026-08-22 default-branch move.

```bash
ls ~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface/hub/models--*/snapshots/
```

## Revisions used around this repo

| Checkpoint | Revision | Note |
|---|---|---|
| `RadixArk/Qwen3.8-27B-NVFP4` | `554ebba9…` | behind every published table in `results/` |
| `RadixArk/Qwen3.8-27B-NVFP4` | `319f741c…` | upstream default since 2026-08-22; the 2026-08-27 FP8-comparison leg |
| `Qwen/Qwen3.8-27B-FP8` | `017b9c7a…` | the FP8 leg |
| `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` | `69d21348…` | side measurement |
| `orcarouter/Qwen3.8-27B-Uncensored-FP8` | `0f3cdb83…` | not benchmarked here |
| `z-lab/Qwen3.8-27B-DFlash2` (draft) | `50307d4c…` | pinned by `start-dflash.sh` |

Carry the snapshot you intend to pin. If the pinned sha has no local snapshot
directory, the offline load will not find it — download that revision first.
