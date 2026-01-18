# Burn Backends Mental Model (Backend vs Device vs Tensor)

## Goal

Build an intuition for what changes (and what stays the same) when you “switch backends” in Burn.

## Mental Model (Practical)

In Burn, a `Backend` (`B`) is a compile-time choice that determines:

- the _tensor storage representation_ and kernels used to implement ops
- the _device type_ (`B::Device`) you must construct/pass in
- which accelerators are available (CPU-only, GPU via wgpu, etc.)

Think of it like: **`Tensor<B, D>` is the same API shape, but `B` decides where/how it runs.**

### Device is part of the backend

This is the big “click”:

- `NdArray` → device is “CPU only” (simple default device)
- `Wgpu` → device is `WgpuDevice` and can be CPU or a GPU adapter

So “backend swapping” often implies “device construction swapping”.

### Transfers are explicit and expensive

Burn makes device transfers explicit via `to_device(...)`. Treat it as **allocate + copy**:

- OK for demos/debugging
- avoid doing it inside tight training loops

## Backend Wrappers

- `Autodiff<B>`: adds gradient tracking/backprop on top of a base backend. Forward outputs should match the inner backend (`B`) for the same ops, but you pay extra memory/compute and you can now call `backward()` / inspect grads.
- `Fusion<B>`: changes how ops are executed (kernel fusion) for speed. The intent is “same math”, but you may see tiny floating-point differences (different kernel implementations / operation ordering), and determinism can vary by device/backend.

1. **Forward equality (tolerance)**: run a tiny forward graph on two backends/wrappers and check outputs are “close enough” (not necessarily bit-identical).
2. **Performance**: run the same small workload a few times and compare rough timings (even `std::time::Instant` is enough at first).
3. **Determinism**: seed where possible; rerun and see if the same backend gives stable outputs run-to-run.

### Update (Burn 0.19.1 source/README)

- `Autodiff<B>`: a backend decorator that swaps float tensor primitives to `AutodiffTensor` and records a graph for backprop. Forward math should match the inner backend, but it adds memory/compute to track grads. Only Autodiff backends expose `backward()` (compile-time safety), and Burn autodiff is first-order reverse mode.
- `Fusion<B>`: a backend decorator that queues ops and fuses them on supported backends. Execution is lazy and drained on sync/host reads; the intent is identical math, but expect tiny floating-point differences from fused kernels and changed op ordering.
- Composition: the common training stack is `Autodiff<Fusion<B>>` if the base backend supports fusion. Fusion alone has `ad_enabled = false`, so gradients are off unless you wrap it. On first-party accelerated backends (wgpu/cuda), fusion is enabled by default via their default features, so you often don't wrap it manually.
- Sanity check in code: `03_backends_mental_model` now prints a tiny forward/timing comparison for `Fusion` (Wgpu default) vs `Autodiff<Fusion<...>>`, plus a single grad sample to prove tracking.

### WGPU vs CUDA (Quick Contrast)

- WGPU: cross-platform via wgpu (Vulkan/Metal/DX12/WebGPU), supports many vendors, and has a CPU device option for portability.
- CUDA: NVIDIA-only GPU backend (`CudaDevice`), typically fastest on NVIDIA hardware but requires CUDA drivers/toolkit and is not available on macOS.
- Both support fusion; WGPU and CUDA default features enable fusion in Burn unless you turn it off.

## Code

- `crates/burn_lab/src/bin/03_backends_mental_model.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 03_backends_mental_model`

## References

- Burn Book: <https://burn.dev/book/>
- docs.rs burn: <https://docs.rs/burn>
