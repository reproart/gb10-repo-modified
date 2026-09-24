# Moving the model cache to a new machine, offline

Bring up a new DGX Spark (or rebuild this one) without re-downloading the
checkpoints, and without repeating the first boot's kernel compilation where
it can be avoided.

## What lives where

The native install has one weight cache. The Docker toolkit used to keep a
second one inside its checkout, and a model prefetched into the wrong one was
silently downloaded again. That problem is gone.

| Path | Contents | Carry it? |
|---|---|---|
| `~/.cache/huggingface` | the HF cache: `01-install.sh` downloads into it, and `serve.sh` reads from it | **yes**, this is the ~50 GB that matters |
| `~/.cache/flashinfer`, `~/.triton` | JIT-compiled kernels and autotune results | optional: they are keyed to the exact FlashInfer / Triton / CUDA versions, so they help only a machine with the same lock file |
| `~/spark/venv` | SGLang and its dependencies | no: rebuild it with `01-install.sh`, which installs the exact versions from `requirements/` |

Survey before packing. `-L` follows the snapshot symlinks, without it you
measure the symlink stubs:

```bash
du -shL ~/.cache/huggingface/hub/models--* 2>/dev/null
```

## Coming from the Docker toolkit

The toolkit's cache is `~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface`.
Merge it into the native cache instead of downloading again:

```bash
rsync -a ~/spark/Qwen3.8-27B-SGLang-DGX-Spark/.cache/huggingface/hub/ \
         ~/.cache/huggingface/hub/
```

`rsync -a` keeps the relative symlinks and skips blobs already present, so a
model in both caches is not copied twice. Delete the toolkit copy once
`serve.sh` has booted from the merged one. Its `.cache/triton` is not worth
moving: it was built by a different Triton than the venv has.

## Pack

```bash
tar -C "$HOME" -cpf /media/root/2TB/gb10-caches.tar .cache/huggingface
# optional, same lock file on both machines only:
# tar -C "$HOME" -rpf /media/root/2TB/gb10-caches.tar .cache/flashinfer .triton
```

- `tar`, `rsync -a`, `cp -a` all preserve the cache's **relative** symlinks
  (`snapshots/<sha>/* -> ../../blobs/<hash>`). Never dereference: `cp -rL`
  doubles the size and breaks the layout.
- The target filesystem must support symlinks: ext4/APFS/btrfs are fine,
  exFAT/FAT are not (`stat -f -c %T /media/root/2TB` to check).

## Restore

```bash
tar -C "$HOME" -xpf /media/root/2TB/gb10-caches.tar
./scripts/01-install.sh      # venv; its downloads are no-ops when the cache is complete
```

None of these checkpoints needs an `hf` token.

## Verify: cheap first, then full

With network, an integrity check that downloads **nothing** when the copy is
complete: `hf` prints the snapshot path and exits. `01-install.sh` does exactly
this for the configured target and draft.

```bash
~/spark/venv/bin/hf download Qwen/Qwen3.8-27B-FP8 \
  --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
```

Offline: `HF_HUB_OFFLINE=1 ./scripts/serve.sh`. The pinned revisions are what
make offline resolution deterministic: `--revision <sha>` resolves straight to
the local snapshot, while an unpinned load consults the repo's default branch.
`serve.sh` pins both target and draft, so both must be in the cache.

While verifying, check provenance too: the revision you *served* is the one
whose snapshot directory you *see*, which is not always the one you think.
See the README trap about the 2026-08-22 default-branch move.

```bash
ls ~/.cache/huggingface/hub/models--*/snapshots/
```

## Revisions used around this repo

| Checkpoint | Revision | Note |
|---|---|---|
| `Qwen/Qwen3.8-27B-FP8` | `017b9c7a…` | `TARGET=fp8`, the default; the FP8 leg in `results/` |
| `RadixArk/Qwen3.8-27B-NVFP4` | `554ebba9…` | `TARGET=nvfp4`; behind every earlier table in `results/` |
| `RadixArk/Qwen3.8-27B-NVFP4` | `319f741c…` | upstream default since 2026-08-22; the 2026-08-27 FP8-comparison leg |
| `orcarouter/Qwen3.8-27B-Uncensored-NVFP4` | `69d21348…` | side measurement |
| `orcarouter/Qwen3.8-27B-Uncensored-FP8` | `0f3cdb83…` | not benchmarked here |
| `z-lab/Qwen3.8-27B-DFlash2` (draft) | `50307d4c…` | pinned by `serve.sh`; `incoai/Qwen3.8-27B-DFlash2` mirrors the same weights |

Carry the snapshot you intend to pin. If the pinned sha has no local snapshot
directory, the offline load will not find it, so download that revision first.
