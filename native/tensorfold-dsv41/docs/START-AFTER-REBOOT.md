# Start after a reboot

Both nodes are already deployed ([DEPLOY.md](DEPLOY.md)). A reboot keeps
everything except the two rank containers: the CX7 addresses and MTU live in the
NetworkManager profiles, the no-swap slice, the sysctls and earlyoom are files on
disk, and the image, the weights and `CACHE_DIR` stay where they are. So the whole
routine is three commands on the head.

## 1. Check the link (10 s)

```bash
ip -br addr | grep -E 'enp1s0f1np1|enP2p1s0f1np1'
ping -c2 -I enp1s0f1np1 10.100.80.2
```

Expect `10.100.80.1/24` and `10.100.81.1/24` and a reply from the worker. If the
addresses are missing, the profile did not autoconnect: `sudo nmcli con up "Wired
connection 5"; sudo nmcli con up "Wired connection 2"`.

## 2. Start (1-3 min warm, ~15 min if `CACHE_DIR` was wiped)

```bash
cd ~/gb10-repo-modified/native/tensorfold-dsv41
./serve.sh
```

It runs `preflight.sh` on both nodes first, then rank 0 on the head and rank 1 on
the worker. Wait for:

```text
serving DeepSeek-V4.1-Flash-1017 at http://127.0.0.1:8888/v1; ./verify.sh checks both ranks with real replies
```

Measured on this pair with a warm cache: 1 min 08 s (and 2 min 39 s the first time
the engine had to compile a new CUDA extension).

## 3. Prove it (1 min)

```bash
./verify.sh
```

`5/5 passed`. `/health` and `/v1/models` alone prove nothing: rank 0 answers those
without rank 1.

## 4. Connect

| Setting | Value |
|---|---|
| Base URL | `http://<head>:8888/v1` |
| Model | `DeepSeek-V4.1-Flash-1017` |
| API key | any text, if `API_KEY_FILE` is not set |

Thinking, tools and structured output: [CONNECT.md](CONNECT.md).

## If it does not come up

| Symptom | Do |
|---|---|
| `preflight` `[FAIL]` lines | the line says what to fix; all six appear in [DEPLOY.md](DEPLOY.md#the-six-failures-by-signature) |
| both ranks exit with code 75, or `NCCL error 4` after an `OutOfMemoryError` | `./stop.sh && ./serve.sh` — the lane never recovers inside a dead process |
| NCCL hangs at start, no error, right after a cable event or a reboot | the GID index moved: `show_gids`, then `GID_INDEX=-1` (or the new v2 IPv4 index) in `config.local.sh`, then `./stop.sh && ./serve.sh` |
| `ssh: connect to 10.100.80.2: No route to host` | step 1: the CX7 address or the link is missing; `ibdev2netdev` shows the port `(Up)` or not |
| `docker: permission denied` | the user is not in the `docker` group on that node |

Both ranks must always be restarted together, rank 0 first — `./stop.sh &&
./serve.sh` does that. `RESTART_POLICY=on-failure` lets docker restart them
separately but cannot guarantee that order; [OPERATIONS.md](OPERATIONS.md) covers
that and the systemd unit that can.

Do not start a second model server on the same pair (port 8888, ~100 GB a node),
and do not `docker exec` into a rank while it serves.
