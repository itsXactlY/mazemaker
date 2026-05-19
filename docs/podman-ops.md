# Podman Ops — Start, Stop, Check, Update

The TL;DR card for running the Mazemaker pod. Everything is rootless
Podman + Quadlet under `systemd --user`. No sudo needed for any of this.

> **Where things live.**
> Unit files: `~/.config/containers/systemd/mazemaker*.{container,pod}`
> Image storage: `~/.local/share/containers/storage/`
> Logs: `journalctl --user -u <service>`

---

## TL;DR

```bash
# Status
systemctl --user list-units 'mazemaker-*.service'
podman ps --format "{{.Names}}\t{{.Status}}"

# Start everything
systemctl --user start mazemaker-pod.service

# Stop everything
systemctl --user stop mazemaker-pod.service

# Restart one container
systemctl --user restart mazemaker-wonderland.service

# Update (pull new images + reload Quadlet + restart)
bash install.sh --refresh

# Tail logs (one container)
journalctl --user -u mazemaker-wonderland.service -f
```

That's 90 % of what you'll ever need.

---

## What's running

The pod is one Quadlet-defined `.pod` plus seven `.container` units:

| Unit                                    | Image                              | What it does                                  |
|-----------------------------------------|------------------------------------|-----------------------------------------------|
| `mazemaker-pod.service`                 | (pod, no image)                    | Parent pod owning the shared network/IPC      |
| `mazemaker-pgvector.service`            | `pgvector/pgvector:pg16-trixie`    | PG primary store for memories + edges         |
| `mazemaker-license-client.service`      | `mazemaker-v2-license-client`      | Ed25519 JWT verify + quota counter            |
| `mazemaker-embedding-worker.service`    | `mazemaker-v2-embedding-worker:gpu`| BGE-M3 1024-d via FastEmbed/ONNX or CUDA       |
| `mazemaker-mcp.service`                 | `mazemaker-v2-mcp:gpu`             | MCP server + neural-memory engine             |
| `mazemaker-dream-worker.service`        | `mazemaker-v2-mcp:gpu`             | Standalone NREM / REM / Insight / AFE / DAE loop |
| `mazemaker-wonderland.service`          | `mazemaker-v2-wonderland`          | AES-256-GCM zero-knowledge MCP proxy on `127.0.0.1:8765` |
| `mazemaker-hermes-bridge.service`       | (Python, no container)             | HTTP sidecar on `127.0.0.1:8769` for the Architect cockpit |
| `mazemaker-mcp-socket-bridge.service`   | (Python, no container)             | UNIX socket → HTTP `/mcp` adapter             |

`mazemaker-mcp` and `mazemaker-dream-worker` share the same image but run
different commands. The dream worker is the standalone daemon variant —
see [`dream-engine.md`](dream-engine.md#standalone-daemon--dream_workerpy)
for why it runs as its own process.

---

## Status

### Quick one-liner

```bash
systemctl --user list-units 'mazemaker-*.service' --no-pager
```

Sample healthy output:

```
mazemaker-dream-worker.service        loaded active running
mazemaker-embedding-worker.service    loaded active running
mazemaker-hermes-bridge.service       loaded active running
mazemaker-license-client.service      loaded active running
mazemaker-mcp.service                 loaded active running
mazemaker-mcp-socket-bridge.service   loaded active running
mazemaker-pgvector.service            loaded active running
mazemaker-pod.service                 loaded active running
mazemaker-wonderland.service          loaded active running
```

Any unit showing `failed` instead of `active running` — look at its log:

```bash
systemctl --user status mazemaker-<unit>.service --no-pager
journalctl --user -u mazemaker-<unit>.service --since "10 min ago"
```

### Container view (podman)

```bash
podman ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
```

### Health probe

```bash
# Wonderland health
curl -s http://127.0.0.1:8765/healthz | jq

# pgvector
podman exec systemd-mazemaker-pgvector pg_isready -U mazemaker

# Memory count
curl -s -X POST http://127.0.0.1:8765/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name":"mazemaker_stats","arguments":{}}' | jq .result.memories
```

### Forward-check (per the operator rule)

Right after an init or restart, **verify within 3 min** that the GPU
recall engine armed. If the log doesn't show `GPU recall ARMED`,
the engine is on CPU.

```bash
journalctl --user -u mazemaker-mcp.service --since "3 min ago" | grep -E "GPU recall ARMED|CPU/numpy"
```

See [`production-lessons.md#verify-forward-not-after`](production-lessons.md#verify-forward-not-after)
for the full operator rule.

---

## Start

Starting the pod cascades to every member container in the right order
(license-client → pgvector → embedding-worker → mcp → wonderland →
dream-worker).

```bash
systemctl --user start mazemaker-pod.service
```

Wait ~30 s for everything to settle, then health-check:

```bash
curl -fsS http://127.0.0.1:8765/healthz && echo " ✓"
```

If the host just rebooted, also linger the user (so units stay up after
logout):

```bash
loginctl enable-linger "$USER"
```

---

## Stop

```bash
# Stop everything
systemctl --user stop mazemaker-pod.service

# Stop just one container
systemctl --user stop mazemaker-wonderland.service
```

**Volumes are preserved.** `pg_dump`, embeddings cache, license JWT,
GPU cache — all live on disk and survive any stop/start cycle.

To stop AND remove the units but keep data:

```bash
bash install.sh --uninstall
```

To hard-wipe everything (including memory.db, embeddings, license):

```bash
bash install.sh --uninstall
rm -rf ~/.local/share/mazemaker ~/.mazemaker
podman volume rm mazemaker-pgdata mazemaker-models 2>/dev/null
```

---

## Restart one container

```bash
systemctl --user restart mazemaker-wonderland.service
systemctl --user restart mazemaker-mcp.service
systemctl --user restart mazemaker-dream-worker.service
```

After restarting `wonderland`, hold on the health probe before issuing
calls — the wonderland → mcp → engine chain takes ~3 s to come back up.

```bash
until curl -fsS http://127.0.0.1:8765/healthz >/dev/null 2>&1; do sleep 1; done && echo "ready"
```

---

## Update

### The supported path

```bash
bash install.sh --refresh
```

This:

1. Pulls fresh container images for every Quadlet that has an
   updatable image.
2. Re-renders the Quadlet units (idempotent — no-op if unchanged).
3. `systemctl --user daemon-reload`.
4. Restarts every changed unit in dependency order.
5. Re-probes `/healthz` to confirm green.

Volumes are untouched. Data, license, embeddings, GPU cache — all
preserved across updates.

### Manual image pull (debugging)

If you suspect a registry issue and want to pull one image manually:

```bash
podman pull localhost/mazemaker-v2-mcp:gpu          # for local-built images
podman pull docker.io/pgvector/pgvector:pg16-trixie # for upstream images
systemctl --user restart mazemaker-mcp.service
```

### After a host kernel / CUDA driver update

```bash
# Re-generate the NVIDIA CDI spec (root, one-time per driver update)
sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml

# Restart the GPU-using units
systemctl --user restart mazemaker-embedding-worker.service mazemaker-mcp.service mazemaker-dream-worker.service

# Verify GPU is detected inside the container
podman exec systemd-mazemaker-mcp nvidia-smi | head -5
```

If `nvidia-smi` works on the host but fails inside the container, the
host `libnvidia-container` stack needs updating. Engine falls back to
CPU FastEmbed gracefully — you'll see `CPU/numpy` in the logs instead
of `GPU recall ARMED`.

---

## Logs

```bash
# Live tail one service
journalctl --user -u mazemaker-wonderland.service -f

# Last 100 lines, no follow
journalctl --user -u mazemaker-mcp.service -n 100 --no-pager

# All mazemaker services since a time
journalctl --user --since "10 min ago" \
  -u mazemaker-pod.service \
  -u mazemaker-mcp.service \
  -u mazemaker-wonderland.service \
  -u mazemaker-dream-worker.service \
  --no-pager

# Container-level (alternate path)
podman logs --tail 100 systemd-mazemaker-mcp
podman logs -f systemd-mazemaker-wonderland
```

---

## Quadlet introspection

When you suspect a unit file got hand-edited or `--refresh` lost a change:

```bash
ls -la ~/.config/containers/systemd/mazemaker*
cat   ~/.config/containers/systemd/mazemaker-mcp.container
```

After hand-editing a `.container` file, always:

```bash
systemctl --user daemon-reload
systemctl --user restart mazemaker-<name>.service
```

Quadlet generates the actual `.service` unit on `daemon-reload` from
the `.container`. Skipping daemon-reload means systemd is still running
the old generated service definition.

---

## Common failure modes

### "Address already in use" on a socket

Usually the wonderland sidecar didn't shut down cleanly:

```bash
ss -ltnp | grep -E "8765|8769"
# Find the PID, then:
kill <pid>
systemctl --user start mazemaker-wonderland.service
```

### `pg_isready` returns "no response"

```bash
journalctl --user -u mazemaker-pgvector.service --since "5 min ago"
```

Most often a uid/gid mismatch on the data volume from a prior install:

```bash
# Non-destructive fix:
podman unshare chown -R 999:999 \
  ~/.local/share/containers/storage/volumes/mazemaker-pgdata/
systemctl --user restart mazemaker-pgvector.service
```

### Dream cycle sleeping at 0% CPU / 0% GPU

A long-running PG query is hung. See
[`dream-engine.md`](dream-engine.md#observability) for the
`pg_stat_activity` query that surfaces it, and the
[`production-lessons.md#patched-bug-index`](production-lessons.md#patched-bug-index)
table for the `prune_orphans` NOT-EXISTS rewrite (10 min hang →
milliseconds).

### License JWT missing / expired

```bash
ls -la ~/.local/share/mazemaker/license.jwt
journalctl --user -u mazemaker-license-client.service --since "10 min ago"
```

Refresh via wonderland:

```bash
curl -s -X POST http://127.0.0.1:8765/license/refresh-now
```

---

## Reboot behavior

With `loginctl enable-linger "$USER"` set, the pod auto-restarts on
host reboot. Otherwise it stays down until you log in again.

If you used the ephemeral `MM_DREAM_DISABLED=1` drop-in (see
[`dream-engine.md`](dream-engine.md#standalone-daemon--dream_workerpy)),
it clears on reboot and the in-pod dream engine auto-re-enables.

---

## Going deeper

- **Install one-liner walkthrough** — [the onboarding page on the marketing site](https://mazemaker.online/onboarding/)
- **What each container actually does** — [`architecture.md`](architecture.md)
- **Dream engine standalone daemon** — [`dream-engine.md`](dream-engine.md#standalone-daemon--dream_workerpy)
- **Operator rules (verify forward, etc.)** — [`production-lessons.md`](production-lessons.md)
