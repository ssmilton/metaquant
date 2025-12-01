# Runtimes, platforms, and nodes

This document describes how MetaQuant executes models across hosts and containers, including Windows container support.

## Manifest schema extensions
- `runtime`: `host` or `docker`.
- `platform`: `linux/amd64`, `windows/amd64`, or `any` (host-only, OS-agnostic models).
- Docker-only fields: `docker_image`, `docker_entrypoint`.
- Host-only fields: `entrypoint`, optional `dependencies` (e.g., `python`, `requirements_file`, language/tool versions).

Examples:

**Linux docker model**
```json
{
  "model_id": "sector_momentum_v1",
  "version": "1.0.0",
  "description": "Sector rotation momentum strategy.",
  "author": "alice",
  "runtime": "docker",
  "platform": "linux/amd64",
  "docker_image": "registry.example.com/quant/sector_momentum:v1.0.0",
  "docker_entrypoint": "python /app/run_model.py",
  "input_types": ["prices", "market_features"],
  "output_type": "signals",
  "tags": ["equity", "sector", "daily"]
}
```

**Windows docker model**
```json
{
  "model_id": "legacy_windows_iv_model",
  "version": "0.9.0",
  "description": "Windows-only IV model using .NET / COM libraries.",
  "author": "bob",
  "runtime": "docker",
  "platform": "windows/amd64",
  "docker_image": "registry.example.com/quant/legacy_iv:0.9.0",
  "docker_entrypoint": "C:\\app\\run_model.exe",
  "input_types": ["prices", "options"],
  "output_type": "signals",
  "tags": ["options", "windows-only"]
}
```

**Host Python model**
```json
{
  "model_id": "simple_equity_momentum",
  "version": "0.1.0",
  "description": "Simple host-based Python momentum model.",
  "author": "charlie",
  "runtime": "host",
  "platform": "any",
  "entrypoint": "python run_model.py",
  "dependencies": {
    "python": ">=3.11",
    "requirements_file": "requirements.txt"
  },
  "input_types": ["prices"],
  "output_type": "signals",
  "tags": ["equity", "daily"]
}
```

Validation rules:
- Docker runtime requires `docker_image` and `docker_entrypoint` and cannot use `platform: any`.
- Host runtime requires `entrypoint`; dependencies are optional.
- Platform must be one of `linux/amd64`, `windows/amd64`, or `any`.

## Execution nodes and scheduling
Nodes declare where models can run. `config/base.yaml` ships with a default Linux node; add more for Windows or specialized hardware.

Example:
```yaml
nodes:
  - name: "local-linux-dev"
    platform: "linux/amd64"
    runtimes: ["host", "docker"]
    tags: ["dev", "default"]
  - name: "win-quant-01"
    platform: "windows/amd64"
    runtimes: ["docker"]
    tags: ["windows", "options"]
```

Scheduling rules:
- Node platform must match the model platform (or the model declares `any`).
- Node runtimes must include the model runtime.
- The scheduler prefers nodes whose tags intersect requested `--node-tags` and falls back to the first compatible `default` node.
- Passing `--node <name>` on the CLI enforces an explicit node and will raise if the model is incompatible.

## Runner behavior
- **HostModelRunner**
  - Creates a per-model virtual environment under `env_root` (default `.envs/<model>-<version>`).
  - Rewrites `python ...` entrypoints to use the venv interpreter and installs `requirements_file` when specified.
- **DockerModelRunner**
  - Uses `docker run --rm --platform <platform> -i <image> <docker_entrypoint>`.
  - Honors config defaults for timeout, CPU, memory, and bind mounts; supports both Linux and Windows containers (given an appropriate host/daemon).

## Local workflow tips
- Prefer `runtime: docker` with `platform: linux/amd64` for reproducibility.
- Use `runtime: host` for quick Python iterations; keep dependencies minimal and OS-agnostic when possible (`platform: any`).
- When testing docker models locally:
  - `docker build -t mymodel:dev .`
  - `echo '{"mode":"backtest",...}' | docker run --rm --platform linux/amd64 -i mymodel:dev python /app/run_model.py`
- For Windows-only dependencies, publish Windows container images and configure a Windows node in `config/base.yaml` (or a separate config file) with `runtimes: ["docker"]`.
