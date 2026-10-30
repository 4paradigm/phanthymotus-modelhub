# phanthymotus-modelhub

On-robot model serving for [PhanthyMotus](https://github.com/4paradigm/phanthymotus):
large models (LLM first, then VLM and others) running **on the robot's own
accelerator**, exposed over an OpenAI-compatible HTTP API.

> **Status: empty, design stage.** Nothing here runs yet. This README records what
> the project is for and where its boundaries are, so the first code lands in the
> right place.

## Where it sits

PhanthyMotus has four on-robot components. This is a fifth, alongside them:

| Component | Repo | Does |
|---|---|---|
| Drivers | `phanthymotus-driver` | MCP servers for physical devices |
| Perception | `phanthymotus/perception` | ASR, TTS, vision — streaming, in-process models |
| Agent Core | `phanthymotus/agent-core` | LLM agent loop, canvas, dashboard — **CPU only** |
| ActuCore | `phanthymotus/actucore` | Execution layer, including local VLA |
| **ModelHub** | **this repo** | **Request/response serving of large models on the robot** |

Agent Core already talks to any LLM through a list of `{url, key, model}` entries
and an OpenAI client. A ModelHub endpoint is just one more entry:

```
{"url": "http://127.0.0.1:<port>/v1", "key": "local", "model": "<model>"}
```

So Agent Core stays a small CPU application, and local and cloud models sit behind
the same interface — it does not know which one it is talking to.

## Why a separate component

- **Different runtime shape.** Perception loads models into its own process and
  works on streaming ROS topics. An LLM server is a standalone process with its own
  scheduler (continuous batching, KV cache).
- **Different image and release cadence.** Inference engines bring their own large
  CUDA stack. Folding one into the perception image couples two release cycles
  and invites the shared-runtime conflicts perception has already hit.
- **Not the execution layer.** ActuCore drives the robot and is gated as such;
  a language model server is not an actuator.

## Boundary with `phanthymotus-cloud`

Split by **where the model runs and who owns the resource contention**, not by
"is it inference":

- **ModelHub (here):** models on the robot. They compete with perception for the
  same unified memory and GPU, so the memory budget is a robot-side decision.
- **phanthymotus-cloud:** models in a machine room. Throughput, cost, autoscaling,
  API keys and billing live there.

## What this repo owns — and does not

| Owns | Does not own |
|---|---|
| Packaging an upstream engine (vLLM / TensorRT-LLM / llama.cpp) for each target | Writing an inference engine |
| Model manifests, weight download and checksum | Billing, API key issuance |
| Memory budget alongside perception | The agent loop |
| Warmup and health gating (ready only after the model is loaded) | Robot actuation |
| OpenAI-compatible endpoint; optional status/switch-model hook for the dashboard | |

Name things after the role, not after one model class or one framework — the same
service will host VLMs and embeddings too.

## First target: Jetson Thor T4000

| | |
|---|---|
| JetPack | 7.2.1 (L4T R39.2.1, Ubuntu 24.04) |
| CUDA / TensorRT | 13.2 / 10.16 |
| Memory | 59 GB unified (CPU + GPU + perception share it) |

Engines must be Thor (CUDA 13) builds; Orin JetPack 5/6 images do not apply.

## Open questions

1. **Engine.** Which of vLLM, TensorRT-LLM, llama.cpp runs reliably on Thor with
   reliable tool calling — to be measured on the device.
2. **Memory budget.** On Jetson, an engine's "GPU memory fraction" is a fraction of
   the *whole* unified memory; the default will starve perception. The budget must
   be explicit.
3. **Local-first routing in Agent Core.** Agent Core currently sends each request to
   **all** live endpoints at once and takes the first answer. With a local and a
   cloud entry configured, every turn hits both. "Local first, cloud as fallback"
   needs a change there.

## License

Apache-2.0, see [LICENSE](LICENSE).
