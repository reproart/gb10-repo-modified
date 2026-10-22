# Deploy: from two bare Sparks to a verified server

The steps in order, with what each one must print and the six failures this pair
actually hit. Nine steps, one `sudo` batch, ~45 min of work plus the first start.

This page is the long form of the README's "Setup and run": it exists because
three of the six failures below cost more than ten minutes each and none of them
looks like the thing that is wrong. If you already have a running pair, the short
version is [START-AFTER-REBOOT.md](START-AFTER-REBOOT.md).

Everything runs on the head, in this directory, as the same user that exists on
both nodes. The worker is reached with `ssh $WORKER_SSH`.

Numbers measured on this pair (gx10-230e head, gx10-1cc3 worker) while writing
this page: 2026-10-22.

## 0. Before you start

| | |
|---|---|
| Cable | QSFP between the two ConnectX-7 ports, `ibdev2netdev` shows the ports `(Up)` on both nodes |
| Accounts | the same user on both nodes, in the `docker` group on both, key-based `ssh` head → worker |
| Tools | `rsync`, `git`, `docker`, `earlyoom` (`sudo apt install earlyoom` if missing) |
| Disk | ~400 GB for the weights and ~110 GB per node for `CACHE_DIR` |

`ssh $WORKER_SSH true` must succeed **without a password prompt**: `build.sh`,
`serve.sh`, `stop.sh` and `preflight.sh` call ssh with `BatchMode=yes`, so an
unknown host key fails them instead of asking. If the worker's key is not in
`known_hosts` yet, add it while you still have a network path to the worker
(then it never prompts later):

```bash
ssh-keyscan -T 5 "$WORKER_SSH" >> ~/.ssh/known_hosts
```

## 1. Static IPv4 on the CX7 ports — both nodes

Two ports, two subnets, one address on each port: the recipe expects
`10.100.80.0/24` on partition 1 and `10.100.81.0/24` on partition 2. The
addresses live in the NetworkManager profile of each interface — do not create
new profiles, the existing ones already own the device.

Which profile owns which device (the names are the same on both nodes here, but
check, they are not guaranteed):

```bash
nmcli -t -f NAME,DEVICE con show | grep -E 'enp1s0f1np1|enP2p1s0f1np1'
```

On the **head**:

```bash
sudo nmcli con mod "Wired connection 5" ipv4.method manual ipv4.addresses 10.100.80.1/24
sudo nmcli con mod "Wired connection 2" ipv4.method manual ipv4.addresses 10.100.81.1/24
sudo nmcli con up "Wired connection 5"
sudo nmcli con up "Wired connection 2"
```

On the **worker** (`ssh` over the management network, not over the CX7 address
you are about to bounce: `con up` drops that session):

```bash
ssh andrew@<worker-management-ip>
sudo -i
nmcli con mod "Wired connection 5" ipv4.method manual ipv4.addresses 10.100.80.2/24
nmcli con mod "Wired connection 2" ipv4.method manual ipv4.addresses 10.100.81.2/24
nmcli con up "Wired connection 5"
nmcli con up "Wired connection 2"
```

No `ipv4.gateway`: the default route must stay on the management network, so
CX7 never carries internet traffic. `ipv4.method manual` and `autoconnect=yes`
mean the addresses come back by themselves after a reboot.

Check (on the head):

```bash
ip -br addr | grep -E 'enp1s0f1np1|enP2p1s0f1np1'
ping -c2 -I enp1s0f1np1 10.100.80.2
```

## 2. MTU 9000 on the CX7 ports — both nodes

**This one is not optional and it is not in `NETWORK.md`.** The engine's RDMA
gather hardcodes the path MTU:

```text
/opt/tensorfold/src/tensorfold/cuda/rdma_gather.cu:538   rtr.path_mtu = IBV_MTU_4096;
```

With the interfaces at the default 1500 the QPs connect, the weights load and the
server dies in the first gather of its own warm-up, about 11 minutes in, with:

```text
RuntimeError: rdma gather: RDMA completion: transport retry counter exceeded
```

`ib_write_bw` still reports a healthy 111 Gb/s, because it reads the port's active
MTU instead of assuming one — so the link looks perfect while the engine cannot
pass a single packet. Set the same MTU in the same profiles, then re-activate:

```bash
sudo nmcli con mod "Wired connection 5" ethernet.mtu 9000
sudo nmcli con mod "Wired connection 2" ethernet.mtu 9000
sudo nmcli con up "Wired connection 5"
sudo nmcli con up "Wired connection 2"
```

The same four commands on the worker. Check end to end — a 8972-byte payload only
passes when **both** ends are at 9000:

```bash
ping -M do -s 8972 -c2 10.100.80.2
```

```bash
./preflight.sh 2>&1 | grep -E 'MTU|GID'
```

```text
[ ok ] head: rocep1s0f1 (enp1s0f1np1) MTU 9000
[ ok ] worker: roceP2p1s0f1 (enP2p1s0f1np1) MTU 9000
```

A port under 4096 is now a `[FAIL]` in `preflight.sh` instead of a start that
dies eleven minutes later, so this is worth running before the weights download.

## 3. The RoCE v2 GID index

`docs/NETWORK.md` says index 3, which was true on the measured pair. On a pair
that has just been given IPv4 addresses the table has six entries per device and
the IPv4 RoCE v2 entry is the **last** one:

```bash
show_gids | grep -E 'rocep1s0f1|roceP2p1s0f1'
```

```text
rocep1s0f1    1  0..3  fe80::…                        v1/v2   (link-local)
rocep1s0f1    1  4     0000:…:ffff:0a64:5001  10.100.80.1  v1
rocep1s0f1    1  5     0000:…:ffff:0a64:5001  10.100.80.1  v2   <-- this one
```

Pick the index whose row has **an IPv4 address and `v2`**, and make sure it is the
same index on both nodes and both devices. On this pair it is 5, so
`config.local.sh` has `GID_INDEX=5`. `GID_INDEX=-1` lets NCCL choose instead, at
the cost of determinism.

The table moves after a link flap (cable reseat, one node rebooting): indices are
reassigned, a fixed number then points at different entries on the two sides, and
NCCL hangs at start.

`preflight.sh` refuses a wrong index in five seconds: `FAIL <dev> GID <n> is
link-local (RoCE v2)` means the number points at an `fe80::` entry, and
`FAIL <dev> GID <n> is IB/RoCE v1, not RoCE v2` or `is empty` means the index does
not exist on that device at all. Both are checked on both nodes and both devices.

## 4. `config.local.sh`

```bash
cp config.local.sh.example config.local.sh
```

Set `HEAD_IP`, `WORKER_IP`, `WORKER_SSH`, `RDMA_DEVICES`, `NCCL_SOCKET_IFNAME`
(from `ibdev2netdev` and `ip -br addr`) and `GID_INDEX` from step 3.

`MODEL_DIR` and `ENGRAM_DIR` must be **the same two paths on both nodes**, and
they must be where the weights really are. Worth checking explicitly — this pair
had the Engram folder at `/models/DeepSeek-V4.1-Flash-1017-engram` on the head and
`/models/deepseek-ai/DeepSeek-V4.1-Flash-1017-engram` on the worker, while the
config named the second one on both:

```bash
ls "$MODEL_DIR"/model-*-of-*.safetensors | wc -l        # 43
ls "$ENGRAM_DIR"                                        # 4 files, 189 GiB
ssh "$WORKER_SSH" "ls '$ENGRAM_DIR'"                    # the same 4 files
```

## 5. Weights on both nodes

```bash
./download.sh          # download, engram, verify (sha256, ~385 GiB read), worker
```

Resumable: run it again after any break. `./download.sh verify` alone re-checks
files you copied by hand. A folder copied with `rsync -a` works too.

## 6. The patched chat template

```bash
./patch-template.py --worker
```

It writes the template at a pinned commit plus a one-hunk fix to `MODEL_DIR` on
both nodes. `preflight.sh` compares its sha256 with
`defbdfc2a28ea921d9ad4c6be7244fa82c53f6d552507527a944ad4cbdd24a6d`, so a later
`hf download` that overwrites the file fails the start rather than changing the
prompt encoding behind your back.

## 7. The engine image on both nodes

```bash
./build.sh
```

Builds `tensorfold-dsv41:<TF_COMMIT>` on the head, `docker save | ssh docker load`s
it to the worker (24.5 GB) and refuses if the two image IDs differ. Measured:
3 min 49 s when the base image was already pulled.

## 8. Host tuning — once, sudo on both nodes

```bash
./host-tuning.sh
```

Run it **on the head only**. It checks both nodes first, then applies on the head
and reaches the worker over `ssh -t`, asking for the sudo password twice. Running
it on the worker fails at the first check (`Host key verification failed`),
because it would ssh to `10.100.80.2` — itself — and nothing is wrong.

It creates `tensorfold.slice` (`MemorySwapMax=0`) and sets five sysctls plus
earlyoom. Everything it changes survives a reboot. Without the slice, `preflight`
refuses with `FAIL tensorfold.slice has swap or is missing`. Restart the server
after it: the engine sizes its memory at start.

## 9. Start and prove it

```bash
./serve.sh
```

It runs `preflight.sh` first — every `[ ok ]` line, and `preflight: all checks
passed`, or fix the `[FAIL]` lines. Then rank 0 on the head, rank 1 on the worker,
and it waits for `/v1/models`.

| Start | Measured here |
|---|---|
| first, cold `CACHE_DIR` (writes the rank caches, compiles the CUDA extensions) | ~15 min: 283 s of weights on the head, 484 s on the worker, then the compiles and two warm-ups |
| later, warm cache, first compile of a new extension | 2 min 39 s |
| later, everything warm | 1 min 08 s |

```bash
./verify.sh
```

Five real requests (plain, thinking, strict `json_schema`, `tool_choice`,
`guided_regex`). `5/5 passed` is the only proof that both ranks serve: `/health`
and `/v1/models` answer with rank 0 alone.

Then the three client settings are in [CONNECT.md](CONNECT.md). The API listens on
`0.0.0.0:8888` with **no key** by default: on a shared network set `API_KEY_FILE`
or `HOST=127.0.0.1`.

## The six failures, by signature

| What you see | What it is | Fix |
|---|---|---|
| `rdma gather: … transport retry counter exceeded` ~11 min into the first start, exit 1 | MTU 1500 against the engine's hardcoded 4096 (`preflight` now refuses this in 5 s) | step 2 (MTU 9000 on both nodes) |
| `preflight: FAIL tensorfold.slice has swap or is missing` | `host-tuning.sh` never ran (or ran on the wrong node) | step 8, on the head |
| `preflight: FAIL missing …/model-00047-of-00048.safetensors` | `ENGRAM_DIR` is right on one node only | step 4 |
| `preflight: FAIL … GID 5 is link-local` / `GID 5 is empty` / `is IB/RoCE v1, not RoCE v2` | the index does not point at the IPv4 RoCE v2 entry | step 3 (`show_gids`) |
| `preflight: FAIL the nodes hold different images` | `build.sh` never copied to the worker | step 7 |
| `RuntimeError: NCCL error 4: invalid argument` + one rank `OutOfMemoryError` | secondary: the other rank died of OOM mid-step, NCCL then aborts | see below; restart with `./stop.sh && ./serve.sh` |

## When a request runs a rank out of memory

The engine sized its allocator at start (`rank 0: allocator ceiling 104.5 GiB
(holds 102.8, host available 3.7, floor 2.0)`), so the headroom is whatever the
host had free at that moment: on this pair one start gave 104.3 GiB (head) against
103.0 GiB (worker), the next one 104.5 and 104.6. The smaller rank dies first.
`docs/KNOWN-LIMITS.md` says it plainly: at 1M context memory is close to the limit.
A ~30k-token prompt is comfortable; an agent sending long prompts in several
streams can cross the line by a few hundred megabytes:

```text
rank 1: torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 200.00 MiB.
rank 0: the lane is broken (rank 1 stopped answering) → NCCL error 4 → exit 75
```

Recovery is always the same, and both ranks must come back together:

```bash
./stop.sh && ./serve.sh
```

To buy headroom, in `config.local.sh` (largest effect first), then restart and
re-run `./verify.sh`:

| Setting | Trade |
|---|---|
| `CONTEXT_LENGTH=262144` | frees the KV pool and the per-stream extents; still four times a long agent session |
| `MAX_RUNNING=2` | fewer buffers and graphs per stream |
| `SPILL_GIB=16` | less host memory for kept prompts |
| `EXTRA_ENV="PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"` | less allocator fragmentation |

Stop other GPU programs on both nodes first: every gigabyte they hold comes off
the ceiling.
