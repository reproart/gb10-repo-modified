# shellcheck shell=bash disable=SC2034  # read by scripts/serve-sglang.sh
# Profile: Xiaomi MiMo-V2.6-Flash (309B MoE, 15B active; hybrid sliding-
# window / global attention with sink bias; vision and audio encoders) on
# TWO DGX Sparks, tensor-parallel over the direct ConnectX-7 link, with its
# DFlash draft.
#
# From the SGLang cookbook's MiMo-V2.6 recipe (docs.sglang.io, "MiMo-V2.6";
# 4x H200 / B300, TP=EP=4, FA4, DeepEP, DFlash 8). It has no Spark cell;
# what changes here and why:
#   - two nodes at TP=EP=2: the checkpoint is ~165 GiB (MXFP4 routed experts,
#     block FP8 elsewhere, BF16 routers), more than one Spark's ~121 GiB.
#     Each Spark holds half, ~83 GiB, and every layer exchanges activations
#     over the link (an all-reduce after attention and after the MoE).
#   - MoE experts on Marlin (W4A16: MXFP4 weights dequantized in the kernel),
#     the cookbook's H200 choice. deep_gemm is SM100 only; flashinfer_mxfp4
#     takes the SM12x CUTLASS path (MXFP8 activations x MXFP4 weights) on
#     GB10 in 0.5.21's code, so it is the one alternative worth a run.
#   - attention on Triton: FA4 is SM100, FA3 SM90; Triton's kernels take the
#     sink bias and the sliding window (the cookbook's MI355X cell uses them).
#   - no DeepEP (needs NVSHMEM; with two ranks a plain all-reduce does the
#     same job), no --enable-dp-lm-head (needs DP attention), Spark-sized
#     batch and prefill chunk.
# Nothing here is measured on GB10 yet: run bench/perf.py before trusting it.
#
# SGLang 0.5.21 ships the MiMo-V2 model with the V2.6 pieces the recipe
# needs (checked in the wheel): store_dtype "mxfp4" -> MXFP4 routed experts
# (configs/model_config.py), moe_router_dtype for the BF16 routers,
# Mxfp4MarlinMoEMethod, DFlash layer capture (set_dflash_layers_to_capture),
# and the mimo reasoning / tool-call parsers. 0.5.20 lacks the MXFP4 path.
SGLANG_VERSION="${SGLANG_VERSION:-0.5.21}"
SGLANG_INDEX="${SGLANG_INDEX:-}"

# The MiMo-V2 multimodal processor imports torchcodec at module level, and
# SGLang's aarch64 wheel does not pull it in (x86 gets torchcodec==0.15.0):
# without it the processor is skipped ("Ignore import error when loading
# ...processors.mimo_v2: No module named 'torchcodec'", seen in a 0.5.21
# boot log here) and a MiMoV2ForCausalLM checkpoint stops at "No processor
# registered for architecture". ./serve.sh mimo-v2.6-flash install adds the
# version SGLang pins for x86; torchcodec also needs FFmpeg's libraries
# (sudo apt install ffmpeg).
EXTRA_PACKAGES="${EXTRA_PACKAGES-torchcodec==0.15.0}"

# Weights, downloaded once ON BOTH SPARKS (every rank opens every shard and
# keeps its half; ~165 GiB each, or copy it over the link with rsync):
#   hf download XiaomiMiMo/MiMo-V2.6-Flash --local-dir /models/XiaomiMiMo/MiMo-V2.6-Flash
# The DFlash draft is inside, in dflash/. Which weights:
#   flash   XiaomiMiMo/MiMo-V2.6-Flash, the cookbook's checkpoint
#   rl      XiaomiMiMo/MiMo-V2.6-Flash-RL (same layout and draft directory)
#   <path>  any other checkpoint directory with this layout
WEIGHTS="${WEIGHTS:-flash}"
case "$WEIGHTS" in
  flash) _dir=/models/XiaomiMiMo/MiMo-V2.6-Flash;    _name=mimo-v2.6-flash ;;
  rl)    _dir=/models/XiaomiMiMo/MiMo-V2.6-Flash-RL; _name=mimo-v2.6-flash-rl ;;
  /*)    _dir="$WEIGHTS";                            _name=mimo-v2.6-flash ;;
  *) echo "WEIGHTS must be flash, rl or an absolute path, not '$WEIGHTS'" >&2; exit 2 ;;
esac
MODEL_DIR="${MODEL_DIR:-$_dir}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-$_name}"
unset _dir _name

# The two machines. Run this profile on both, the same way, with the same
# HEAD_ADDR; only NODE_RANK differs:
#   Spark A (serves the API):  NODE_RANK=0 HEAD_ADDR=10.10.10.1 ./serve.sh mimo-v2.6-flash
#   Spark B (worker, no API):  NODE_RANK=1 HEAD_ADDR=10.10.10.1 ./serve.sh mimo-v2.6-flash
# HEAD_ADDR is Spark A's address ON THE DIRECT LINK (not the LAN one): torch
# rendezvous goes to HEAD_ADDR:DIST_PORT, and NCCL / Gloo are pinned to the
# interface that carries it. Either side may start first; each waits for the
# other. The API, as for every profile, is on Spark A at :$PORT.
NODE_RANK="${NODE_RANK:-0}"
HEAD_ADDR="${HEAD_ADDR:-}"
DIST_PORT="${DIST_PORT:-20000}"
NNODES=2
# The link interface for NCCL_SOCKET_IFNAME / GLOO_SOCKET_IFNAME; empty =
# found from HEAD_ADDR (on rank 0 the interface holding it, on rank 1 the
# route to it). NCCL finds the ConnectX-7 RDMA devices itself; set
# NCCL_IB_HCA in the shell to pick them, NCCL_DEBUG=INFO to see the choice
# ("NET/IB" = RoCE, "NET/Socket" = plain TCP, much slower).
LINK_IF="${LINK_IF:-}"

# Parallelism: one GPU per Spark. EP=2 puts half of the experts whole on each
# Spark (the cookbook keeps EP = TP); EP=1 splits every expert's columns
# between them instead.
TP=2
EP="${EP:-2}"

# Kernels (see above).
MOE_RUNNER_BACKEND="${MOE_RUNNER_BACKEND:-marlin}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-triton}"
# Vision / audio encoder attention. Empty = SGLang's choice (fa4, the
# cookbook's, is SM100); sdpa or triton_attn if image input fails.
MM_ATTENTION_BACKEND="${MM_ATTENTION_BACKEND:-}"

# DFlash speculative decoding with the checkpoint's draft (dflash/): 1 = on,
# 0 = off. 8 draft tokens as in the cookbook; on this box the 27B's best
# value came out at 15, so try DRAFT_TOKENS=12 and 16 with bench/perf.py.
# Every verify step crosses the link twice per layer, so a longer block costs
# less here than one more step.
DFLASH="${DFLASH:-1}"
DRAFT_DIR="${DRAFT_DIR:-$MODEL_DIR/dflash}"
DRAFT_TOKENS="${DRAFT_TOKENS:-8}"
[ "$DFLASH" = 1 ] || DRAFT_DIR=""

# Memory, per Spark: ~83 GiB of weights out of ~97 GiB at 0.80 leaves ~14 GiB
# for KV and activations. The SWA layers keep only their window, so most of
# the pool goes to the global layers; 0.03 is the cookbook's Flash ratio of
# SWA-to-full tokens. The boot log's "KV Cache is allocated ... #tokens" is
# the context that fits in one sequence. 0.85 buys ~6 GiB more, at the edge
# of DGX OS earlyoom's margin (see the Qwen profile's note).
MEM_FRACTION="${MEM_FRACTION:-0.80}"
SWA_RATIO="${SWA_RATIO:-0.03}"
# KV cache dtype. Empty = the model's (BF16). fp8_e4m3 halves it; untested
# with the sink-bias Triton kernels here, so off by default.
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-}"
MAX_RUNNING="${MAX_RUNNING:-8}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-8192}"
# The checkpoint allows 1,048,576; what fits is the KV pool above. 262144 by
# default; raise it once the boot log shows the pool for it.
CONTEXT_LENGTH="${CONTEXT_LENGTH:-262144}"

REASONING_PARSER="${REASONING_PARSER:-mimo}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-mimo}"

EXTRA_ARGS="${EXTRA_ARGS:-}"

# The interface that carries HEAD_ADDR on this machine (rank 0) or leads to
# it (rank 1).
mimo_link_if() {
  if [ "$NODE_RANK" = 0 ]; then
    ip -o addr show 2>/dev/null | awk -v a="$HEAD_ADDR" '{ split($4, x, "/"); if (x[1] == a) print $2 }' | head -n 1
  else
    ip -o route get "$HEAD_ADDR" 2>/dev/null | awk '{ for (i = 1; i < NF; i++) if ($i == "dev") print $(i + 1) }'
  fi
}

model_env() {
  if [ -z "$HEAD_ADDR" ]; then
    echo "mimo-v2.6-flash runs on two Sparks: set HEAD_ADDR to Spark A's address on the" \
      "direct link and NODE_RANK (0 on A, 1 on B), e.g." >&2
    echo "  NODE_RANK=0 HEAD_ADDR=10.10.10.1 ./serve.sh mimo-v2.6-flash   (Spark A)" >&2
    echo "  NODE_RANK=1 HEAD_ADDR=10.10.10.1 ./serve.sh mimo-v2.6-flash   (Spark B)" >&2
    exit 1
  fi
  case "$NODE_RANK" in
    0|1) ;;
    *) echo "NODE_RANK must be 0 (Spark A, the API) or 1 (Spark B), not '$NODE_RANK'" >&2; exit 1 ;;
  esac
  if [ -z "$LINK_IF" ]; then
    if ! command -v ip >/dev/null; then
      echo "ip (iproute2) not found: set LINK_IF to the direct link's interface" >&2
      exit 1
    fi
    LINK_IF="$(mimo_link_if || true)"
  fi
  if [ -z "$LINK_IF" ]; then
    if [ "$NODE_RANK" = 0 ]; then
      echo "HEAD_ADDR=$HEAD_ADDR is not an address of this machine (NODE_RANK=0 must be" \
        "Spark A); its addresses:" >&2
      ip -o -4 addr show 2>/dev/null | awk '{ print "  " $2 "  " $4 }' >&2
    else
      echo "no route to HEAD_ADDR=$HEAD_ADDR; check the link (ping $HEAD_ADDR) or set LINK_IF" >&2
    fi
    exit 1
  fi
  if [ "$LINK_IF" = lo ]; then
    echo "HEAD_ADDR=$HEAD_ADDR goes over loopback on NODE_RANK=$NODE_RANK: is this Spark B?" >&2
    exit 1
  fi
  export NCCL_SOCKET_IFNAME="$LINK_IF" GLOO_SOCKET_IFNAME="$LINK_IF"
  # The multimodal architecture needs its processor, which needs torchcodec
  # (and FFmpeg under it): say so here, not after the weights load.
  if grep -q '"MiMoV2ForCausalLM"' "$MODEL_DIR/config.json" 2>/dev/null &&
     ! "$VENV/bin/python" -c 'import torchcodec.decoders' >/dev/null 2>&1; then
    echo "torchcodec does not import in $VENV; the MiMo-V2 processor needs it:" >&2
    echo "  ./serve.sh mimo-v2.6-flash install    (adds $EXTRA_PACKAGES)" >&2
    echo "  sudo apt install ffmpeg               (if it still fails: FFmpeg's libraries)" >&2
    echo "  $VENV/bin/python -c 'import torchcodec.decoders'   shows the error" >&2
    exit 1
  fi
  # Each Spark must hold its half of the weights inside the static fraction,
  # with something left for KV: fail here, not after minutes of loading.
  local ckpt_gib mem_gib budget_gib
  ckpt_gib=$(du -sbL --exclude=dflash "$MODEL_DIR" 2>/dev/null | awk '{ printf "%d", $1 / 1073741824 }')
  mem_gib=$(awk '/^MemTotal:/ { printf "%d", $2 / 1048576 }' /proc/meminfo)
  budget_gib=$(awk -v m="$mem_gib" -v f="$MEM_FRACTION" 'BEGIN { printf "%d", m * f }')
  if [ -n "$ckpt_gib" ] && [ "$ckpt_gib" -gt 0 ] && [ $((ckpt_gib / TP + 4)) -ge "$budget_gib" ]; then
    echo "the checkpoint is $ckpt_gib GiB, ~$((ckpt_gib / TP)) GiB per Spark, and MEM_FRACTION=$MEM_FRACTION" \
      "gives $budget_gib GiB: no room for KV. Raise MEM_FRACTION (0.85) or check MODEL_DIR." >&2
    exit 1
  fi
  MIMO_CKPT_GIB="$ckpt_gib"
}

model_args() {
  args+=(
    --tp "$TP"
    --ep "$EP"
    --nnodes "$NNODES"
    --node-rank "$NODE_RANK"
    --dist-init-addr "$HEAD_ADDR:$DIST_PORT"
    --moe-runner-backend "$MOE_RUNNER_BACKEND"
    --attention-backend "$ATTENTION_BACKEND"
    --page-size 1
    --swa-full-tokens-ratio "$SWA_RATIO"
    --chunked-prefill-size "$CHUNKED_PREFILL"
    --max-running-requests "$MAX_RUNNING"
    --cuda-graph-max-bs-decode "$MAX_RUNNING"
    --cuda-graph-backend-prefill disabled
  )
  [ -n "$MM_ATTENTION_BACKEND" ] && args+=(--mm-attention-backend "$MM_ATTENTION_BACKEND")
  [ -n "$KV_CACHE_DTYPE" ] && args+=(--kv-cache-dtype "$KV_CACHE_DTYPE")
  if [ "$DFLASH" = 1 ]; then
    args+=(
      --speculative-algorithm DFLASH
      --speculative-draft-model-path "$DRAFT_DIR"
      --speculative-num-draft-tokens "$DRAFT_TOKENS"
    )
  fi
  [ -n "$REASONING_PARSER" ] && args+=(--reasoning-parser "$REASONING_PARSER")
  [ -n "$TOOL_CALL_PARSER" ] && args+=(--tool-call-parser "$TOOL_CALL_PARSER")
  return 0
}

model_summary() {
  echo "node $NODE_RANK of $NNODES ($([ "$NODE_RANK" = 0 ] && echo "API here" || echo "worker, API on $HEAD_ADDR")), rendezvous $HEAD_ADDR:$DIST_PORT over $LINK_IF; TP $TP EP $EP; checkpoint ${MIMO_CKPT_GIB:-?} GiB (~$(( ${MIMO_CKPT_GIB:-0} / TP )) per Spark); MoE $MOE_RUNNER_BACKEND, attention $ATTENTION_BACKEND, KV ${KV_CACHE_DTYPE:-bf16}; $([ "$DFLASH" = 1 ] && echo "DFlash, $DRAFT_TOKENS draft tokens" || echo "no draft"); cap $MAX_RUNNING, context $CONTEXT_LENGTH"
}
