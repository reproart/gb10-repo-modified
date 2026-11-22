# shellcheck shell=bash disable=SC2034  # read by the scripts here
# Qwen3.8-Flash-Next on stock vLLM 0.34 in Docker, one Spark: the same checkpoint as
# ../vllm-flash-ar (Intel's AutoRound int4 experts + an int8 lm_head + fp8 side layers, the
# fp8 n-gram table on NVMe), on vLLM's release instead of the pinned August dev image. A
# second engine next to the first, not a replacement: the two share the weights and the port,
# one runs at a time. Source it, don't run it; every value is "${VAR:-default}", so the shell
# wins (MTP=3 ./serve.sh). This machine's own values go in config.local.sh (not committed).

V034_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
NATIVE_DIR="$(cd "$V034_ROOT/.." && pwd)"
# shellcheck disable=SC1091
[ -f "$V034_ROOT/config.local.sh" ] && . "$V034_ROOT/config.local.sh"

# --- Weights: ../vllm-flash-ar's (download them there), plus a lean copy of the table.
# vLLM 0.34 reads the n-gram rows from the checkpoint's files (--engram-config disk_offload)
# and loads every tensor of every file it is given, and the table folder holds Qwen's FP8
# shards whole (expert scales and dense weights of other layers next to the rows), so
# ./prepare.sh copies just the rows (../vllm-flash-ar/tools/strip_ple_table.py, ~48 GB once)
# and serve.sh shows vLLM the model and the lean table as one folder (src/make_model_view.py:
# links, a merged index, config.json with the table's dtype; no weights copied).
export MODEL_DIR="${MODEL_DIR:-/models/Qwen3.8-Flash-Next-W4A16-AutoRound-hybrid-MTP_int4RTN}"
export TABLE_SRC_DIR="${TABLE_SRC_DIR:-/models/ple-table-fp8}"
export TABLE_DIR="${TABLE_DIR:-/models/ple-table-fp8-lean}"
export VIEW_DIR="${VIEW_DIR:-$HOME/.cache/gb10/vllm-flash-v034/model-view}"

# --- The image: vllm/vllm-openai:v0.34.0 pinned by digest (image.lock, committed) plus
# build-time patches (./build.sh): the int4+fp8 hybrid dispatch, the int8 lm_head, the private
# MTP draft head, FP8 hyper-connection projections, the refusal-direction ablation.
export BASE_IMAGE="${BASE_IMAGE:-vllm/vllm-openai}"
export BASE_TAG="${BASE_TAG:-v0.34.0}"
BASE_DIGEST="$(cat "$V034_ROOT/image.lock" 2>/dev/null || true)"
export BASE_DIGEST
export IMAGE="${IMAGE:-qwen38-flash-v034}"
export NAME="${NAME:-qwen38-flash-v034}"

# --- The API: the same port and model ids as ../vllm-flash-ar as it runs here, so a client
# does not see which engine answers.
export PORT="${PORT:-8000}"
export SERVED_NAME="${SERVED_NAME:-qwen3.8-flash-next qwen}"
export TOOL_PARSER="${TOOL_PARSER:-qwen3_xml}"
# API key: non-empty -> clients send "Authorization: Bearer <key>". Passed as VLLM_API_KEY,
# so it is not in the command line vLLM logs.
export API_KEY="${API_KEY:-}"
export BIND_ADDR="${BIND_ADDR:-}"

# --- Serving: ../vllm-flash-ar's values, for a like-for-like comparison.
export SEQS="${SEQS:-8}"
export MTP="${MTP:-4}"
export CTX="${CTX:-262144}"
export KV_BYTES="${KV_BYTES:-30g}"
export BATCHED_TOKENS="${BATCHED_TOKENS:-8192}"
export PREFIX_CACHE="${PREFIX_CACHE:-1}"
# The GDN state's dtype (--mamba-ssm-cache-dtype); empty = the checkpoint's float32. Set it
# the way ../vllm-flash-ar runs (bfloat16 there once its A/B says so).
export MAMBA_SSM_DTYPE="${MAMBA_SSM_DTYPE:-}"
# The MTP drafter's own head (../vllm-flash-ar's patch 10; the target verifies every drafted
# token, so outputs never change, only how many drafts are accepted). Without it each draft
# step reads the whole int8 head, 248,320 rows (616 MiB): 68.8 / 43.1 tok/s code / Russian at
# stage 0, against 79.6 / 50.2 on ../vllm-flash-ar with it.
# DRAFT_VOCAB: the ids the drafter scores, every other id -inf; their int8 rows on the same
# Marlin kernel (65,536 rows: 160 MiB per draft step). ru (default) = English, code and
# Russian (../vllm-flash-ar/src/draft_vocab_ru_65536.npy), ru-wide = 81,920 of them, 1 = the
# English/code set, a path = your own ids.npy (../vllm-flash-ar/tools/build_draft_vocab.py),
# 0 = the whole head.
export DRAFT_VOCAB="${DRAFT_VOCAB:-ru}"
# DRAFT_HEAD: int8 = the head's own precision (default); int4 = a private int4 g128 copy of
# the head (the slice with DRAFT_VOCAB, else every row); bf16 = the slice in bf16 (compare).
export DRAFT_HEAD="${DRAFT_HEAD:-int8}"
# HC_FP8 (../vllm-flash-ar's patch 16): the hyper-connection projections (down+inject
# [336 x 10240] and up [10240 x 320], ~98 of each a step, BF16 in the checkpoint) in FP8 with
# a scale per row on a Triton GEMM for decode-sized inputs: half the bytes. On the dev image
# +7% decode, HumanEval unchanged; lossy (answers move a little, deterministically). On 0.34
# the down projection otherwise runs a fused BF16 down+SiLU kernel (CuTe-DSL, up to 48 rows),
# which a converted layer gives up: 1 = both FP8 (default), up = only the up projections (the
# fused down kernel kept), 0 = BF16 as shipped.
export HC_FP8="${HC_FP8:-1}"
# Determinism (the same greedy request gives the same reply, cold or from the prefix cache;
# ../bench/consistency.py section 2):
# DET_MOE_ALIGN: 1 (default) = the MoE token alignment in a fixed order (../vllm-flash-ar's
#   patch 15): vLLM's kernel places an expert's routes in atomicAdd arrival order, which moves
#   a route between Marlin MoE m-blocks and so the order of its sums from run to run. On the
#   dev image that was the whole run-to-run difference, at no measurable cost. 0 = the kernel.
# DET_MOE_ALIGN_MIN_TOKENS: the steps it applies to, from this many tokens up; auto =
#   SEQS x (1 + MTP) + 1: prefill and steps with a prefill chunk, not decode-only steps.
export DET_MOE_ALIGN="${DET_MOE_ALIGN:-1}"
export DET_MOE_ALIGN_MIN_TOKENS="${DET_MOE_ALIGN_MIN_TOKENS:-auto}"
# The sparse attention's block top-k: 0.34's persistent_topk is non-deterministic on GB10
# (vllm#51782): consistency.py forked from the first token with it and gave the same text with
# the exact top-k (2026-11-22).
# QSA_DET_TOPK: 1 (default) = @jschmied's deterministic kernel, built into the image (stock
#   speed); if the image could not build it, the exact top-k runs and the boot log says so.
# QSA_EXACT_TOPK: 1 = an exact torch.topk over the visible columns (wins when set; on the dev
#   image it cost 20-40% of long prefill). Both 0 = the stock, non-deterministic kernel.
export QSA_DET_TOPK="${QSA_DET_TOPK:-1}"
export QSA_EXACT_TOPK="${QSA_EXACT_TOPK:-0}"
# Engine-side metrics (../vllm-flash-ar's patch 13): torch's caching-allocator gauges on their
# own Prometheus port (vllm:torch_cuda_allocated_bytes, ..._reserved_bytes,
# ..._max_reserved_bytes, vllm:torch_idle_released_bytes_total; vLLM's /metrics on PORT cannot
# see the engine process). No auth even with API_KEY. 0 = off.
export METRICS_PORT="${METRICS_PORT:-18400}"
# After this many seconds without a request the engine gives torch's cached GPU blocks back
# (torch.cuda.empty_cache(); CUDA graph pools stay) and runs malloc_trim(0) once: on GB10
# they are host memory, held at the peak of the last large request (stage 1's bench started
# with 2-4 GiB free). 0 = keep them; the next large request allocates again.
export IDLE_RELEASE_SEC="${IDLE_RELEASE_SEC:-60}"
# Refusal-direction ablation at inference (../vllm-flash-ar's patch 17, the same knobs), off by
# default:
#   ABLATE=1 ./serve.sh                    the vectors at ABLATE_VECTORS
#   ABLATE=/path/other.npz ./serve.sh      another file (.npz, .npy or .pt)
# A direction is projected out of everything that writes the model's four residual streams
# (embedding, PLE, each attention / GDN and MLP output, the MTP draft's two inputs): the same as
# orthogonalizing those weights against it, on the fly, no weight changes. The vectors, once:
#   hf download Mambavtt/qwen3.8-flash-next-refusal-ablation-vectors \
#     --revision 7b009deb2d3f21c5bb199366cb39e3ed129b4207 \
#     --local-dir /models/qwen3.8-flash-next-refusal-ablation-vectors
#   ABLATE_SOURCE  mean (default): the normalized mean of the file's per-layer vectors; top2 /
#                  top3: their principal directions (a rank-2 / 3 projection); <n>: layer n's.
#   ABLATE_ALPHA   1 removes the component; above 1 overshoots.
#   ABLATE_LAYERS  all, or e.g. 20-47: whose writers are projected (the embedding with layer 0).
#   ABLATE_MTP     1: the MTP draft projects its inputs too; 0: as shipped (answers never depend
#                  on it, only the draft's acceptance).
# serve.sh prints the boot log's "ABLATION ON ..." line (what is projected). On the dev image:
# a refused request answered, decode 79.3 tok/s (../vllm-flash-ar README patch 17).
export ABLATE="${ABLATE:-0}"
export ABLATE_VECTORS="${ABLATE_VECTORS:-/models/qwen3.8-flash-next-refusal-ablation-vectors/refusal_directions-v4-late.npz}"
export ABLATE_SOURCE="${ABLATE_SOURCE:-mean}"
export ABLATE_ALPHA="${ABLATE_ALPHA:-1}"
export ABLATE_LAYERS="${ABLATE_LAYERS:-all}"
export ABLATE_MTP="${ABLATE_MTP:-1}"
# safetensors: the disk-offloaded table is read through the loader's mmap views, which
# fastsafetensors (../vllm-flash-ar's loader) does not hand over.
export LOAD_FORMAT="${LOAD_FORMAT:-safetensors}"
export RESTART="${RESTART:-no}"
# Anything else: vLLM flags (appended) and NAME=value pairs for the container.
export EXTRA="${EXTRA:-}"
export EXTRA_ENV="${EXTRA_ENV:-}"

# Before a start, with the GPU idle: drop the page cache and compact free memory
# (../scripts/mem-prep.sh). The server waits until the API answers (WAIT_MIN minutes).
export MEM_PREP="${MEM_PREP:-1}"
export WAIT_MIN="${WAIT_MIN:-20}"

log() { echo "[vllm-flash-v034 $(date -u +%H:%M:%SZ)] $*"; }
need_docker() {
  command -v docker >/dev/null || { echo "docker not found (DGX OS ships it)" >&2; exit 1; }
  docker info >/dev/null 2>&1 || {
    echo "docker is not usable by $(id -un): sudo usermod -aG docker $(id -un), then log in again" >&2; exit 1; }
}
base_ref() { echo "$BASE_IMAGE@${BASE_DIGEST:?no image.lock: the base image is pinned by digest}"; }
