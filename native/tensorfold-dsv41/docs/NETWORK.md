# Network: TP=2 over the ConnectX-7 link

The two Sparks run one model with tensor parallelism (TP=2). Every layer exchanges activations between the GPUs, so the link between the nodes decides the speed. This recipe uses the QSFP cable between the two ConnectX-7 (CX7) ports, with RoCE v2 (RDMA over Ethernet).

## What you need

- **Cable.** One QSFP cable between the Sparks (the setup in NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) playbook; do it first). The recipe was measured with both CX7 ports active.
- **Addresses.** Static IPv4 addresses on the CX7 netdev, one subnet, for example `10.100.80.1/24` (head) and `10.100.80.2/24` (worker). Put them in `config.local.sh` as `HEAD_IP` and `WORKER_IP`.
- **ssh.** Key-based ssh from the head to the worker (`WORKER_SSH`). `./build.sh`, `serve.sh` and `stop.sh` use it.
- **RDMA devices.** The RDMA device names and the netdev name for `config.local.sh`:

```bash
ibdev2netdev
```

```bash
ip -br addr
```

  Example from the measured pair: `rocep1s0f1 port 1 ==> enp1s0f1np1 (Up)` and `roceP2p1s0f1 port 1 ==> enP2p1s0f1np1 (Up)`. Then:
  - `RDMA_DEVICES` and `NCCL_IB_HCA` = both RoCE devices;
  - `NCCL_SOCKET_IFNAME` = the netdev that holds `HEAD_IP`.

## The GID index

RoCE v2 over IPv4 uses one entry of the port's GID table. On the measured pair it was index 3 (`GID_INDEX=3`).

**A pair that has just been given its IPv4 addresses is not that pair.** The table then has six entries per device and index 3 is the *link-local* RoCE v2 entry. Take the entry whose row has an IPv4 address and `v2` — on this pair that is index **5**:

```text
rocep1s0f1 1 0..3 fe80::…                      v1/v2  (link-local)
rocep1s0f1 1 4    0000:…:ffff:0a64:5001 10.100.80.1 v1
rocep1s0f1 1 5    0000:…:ffff:0a64:5001 10.100.80.1 v2   <-- this one
```

**MTU.** The engine's RDMA gather sets `rtr.path_mtu = IBV_MTU_4096` in `src/tensorfold/cuda/rdma_gather.cu`, so both CX7 interfaces must be at MTU 9000 (the port's default 1500 makes every gather fail with `transport retry counter exceeded`, ~11 min into the first start, while `ib_write_bw` still reports 111 Gb/s). Check end to end with `ping -M do -s 8972 10.100.80.2`. Both settings are in [DEPLOY.md](DEPLOY.md) steps 1-3.

`preflight.sh` now checks both on each node and both devices: `OK <dev> (<netdev>) MTU 9000` and `OK <dev> GID <n> 0000:…:ffff:… (RoCE v2)`. A wrong or link-local GID (`FAIL … is link-local (RoCE v2)`) and a port under 4096 (`FAIL … MTU 1500`) stop the start in five seconds instead of eleven minutes.

**Trap:** after a link flap (a cable reseat, or one node rebooting), the GID tables on the two nodes can shift, and a fixed index then points at different entries on each side. NCCL init then hangs or fails. If that happens:

1. Look at the table on both nodes:

```bash
show_gids
```

2. Set `GID_INDEX` to the RoCE v2 IPv4 entry on both nodes, or set `GID_INDEX=-1` to let NCCL choose.
3. Restart:

```bash
./stop.sh && ./serve.sh
```

## Check the link before you start

```bash
ping -c 3 -I enp1s0f1np1 10.100.80.2
```

```bash
ib_write_bw -d rocep1s0f1 -x 3 10.100.80.1
```

Run `ib_write_bw -d rocep1s0f1 -x 3` (no address) on the head first, then the command above on the worker. A healthy link shows tens of GB/s.

## Ports used

| Port | Node | What |
|---|---|---|
| 29551/tcp | head | TensorFold rendezvous store (rank 1 connects to it) |
| ephemeral tcp | head | TensorFold step link and watchdog heartbeats (rank 0 to rank 1) |
| `PORT` (8888) | head | OpenAI-compatible API (rank 0 only) |
