# ERNIE 5.5 Flash Preview on vLLM 0.32.0 + Baidu's vLLM-ERNIE plugin, with
# jjang-ai's SM12.1 runtime overlay (ernie-5.5-dgx-spark) and its kernel
# packages, every source pinned. Built by ../build.sh.
ARG BASE=vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34
FROM ${BASE}

ARG VLLM_ERNIE_COMMIT=ba322a5634f91d3e6f96ae711df72ddaf3c38b6e
ARG RUNTIME_COMMIT=5b9f1c2474ac4b26172fa2b01222ce8a90c4f504
ARG SPARKNET_REF=v0.6.0
# The versions the runtime's audit lists (docs/ARCHITECTURE_AND_FLAGS_AUDIT.md).
ARG TILELANG_VERSION=0.1.12
ARG HUMMING_VERSION=0.1.17

RUN apt-get update && apt-get install -y --no-install-recommends git ibverbs-utils rdma-core \
    && rm -rf /var/lib/apt/lists/*

# --no-deps throughout: vllm 0.32.0 and torch stay the base image's.
RUN pip install --no-cache-dir --no-deps \
      "git+https://github.com/baidu/vLLM-ERNIE.git@${VLLM_ERNIE_COMMIT}" \
      "tilelang==${TILELANG_VERSION}" "humming-kernels==${HUMMING_VERSION}" \
      "git+https://github.com/christopherowen/dgx-spark-networking@${SPARKNET_REF}"

# The runtime as a checkout (its launcher and sitecustomize hook read the tree),
# not pip-installed: it is active only when SPARK_ERNIE_RUNTIME=1.
RUN git clone https://github.com/jjang-ai/ernie-5.5-dgx-spark.git /opt/spark-ernie \
    && git -C /opt/spark-ernie checkout --quiet "${RUNTIME_COMMIT}" \
    && rm -rf /opt/spark-ernie/.git /opt/spark-ernie/assets

RUN python3 -c "import vllm, vllm_ernie; print('vllm', vllm.__version__)"
ENTRYPOINT []
