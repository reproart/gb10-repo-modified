# Notes for AI agents

You can set up and run this recipe for a user. Follow these steps in this order. Run every command on the head node, in this folder.

## Setup

1. Check that the two nodes are cabled and reach each other (NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) guide; [docs/NETWORK.md](docs/NETWORK.md)). `ibdev2netdev` must show the CX7 ports as Up on both nodes.
2. `cp config.local.sh.example config.local.sh`, then fill in the values from `ibdev2netdev`, `ip -br addr` and `show_gids`. Ask the user for paths you cannot find.
3. `./download.sh` (~400 GB; resumable: run it again after a break). If a download needs a Hugging Face login, ask the user to run `hf auth login`. Never type a token yourself.
4. `./patch-template.py --worker`
5. `./build.sh`
6. `./host-tuning.sh` (once; it needs sudo on both nodes, so the user types the password). It turns swap off for the engine, sets the kernel memory values and makes earlyoom stop the engine (not the node) when memory runs out; preflight fails without it. Restart the server after it, because the engine sizes its memory at start.
7. `./serve.sh` (it runs `./preflight.sh` first; fix each `[FAIL]` line it prints). The first start takes ~15 min.
8. `./verify.sh` must print PASS for all five checks. Only a real reply proves that both nodes work; `/health` alone does not.

Then give the user the three client settings from [docs/CONNECT.md](docs/CONNECT.md).

## Rules

- **Never `docker exec` into `tfv41-r0` or `tfv41-r1` while they serve.** The exec process counts against the rank's memory.
- **Never stop the display manager** (`systemctl stop gdm`) on a Spark. On the tested pair it powered the node off.
- **Run one model server at a time.** Each one needs port 8888 and ~100 GB of memory on each node.
- **The API listens on the network** (`HOST=0.0.0.0`, as the project's other servers), with no key. On a shared network suggest `API_KEY_FILE` or `HOST=127.0.0.1` ([docs/CONNECT.md](docs/CONNECT.md)).
- **If the server stops with exit code 75**, the lane broke on a device fault. Run `./stop.sh && ./serve.sh`.
- **If NCCL hangs at start after a cable or reboot event**, the RoCE GID index moved. Set `GID_INDEX=-1` in `config.local.sh` and restart ([docs/NETWORK.md](docs/NETWORK.md)).
- **The CX7 ports need MTU 9000.** The engine's RDMA gather sets `IBV_MTU_4096`, and at the default 1500 the first start dies ~11 min in with `transport retry counter exceeded` while `ib_write_bw` still looks healthy. `preflight.sh` checks the MTU and the GID (it must be the IPv4 RoCE v2 entry) on both nodes ([docs/DEPLOY.md](docs/DEPLOY.md) steps 2-3).

## Where to look

| Question | File |
|---|---|
| A fresh pair, from two bare Sparks | [docs/DEPLOY.md](docs/DEPLOY.md) |
| After a reboot | [docs/START-AFTER-REBOOT.md](docs/START-AFTER-REBOOT.md) |
| A start fails | `logs/` (saved by `./stop.sh`), `docker logs tfv41-r0`, [docs/OPERATIONS.md](docs/OPERATIONS.md) |
| How to call the API | [docs/CONNECT.md](docs/CONNECT.md) |
| Run the tests | [tests/README.md](tests/README.md) |
| What is not supported | [docs/KNOWN-LIMITS.md](docs/KNOWN-LIMITS.md) |
| Exact versions and checksums | [runs/2026-10-17/PROVENANCE.md](runs/2026-10-17/PROVENANCE.md) |
