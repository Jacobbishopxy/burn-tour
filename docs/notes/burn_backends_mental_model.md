# Burn Backends Mental Model (Backend vs Device vs Tensor)

## Goal

Build an intuition for what changes (and what stays the same) when you “switch backends” in Burn.

## Mental Model (Practical)

In Burn, a `Backend` (`B`) is a compile-time choice that determines:

- the *tensor storage representation* and kernels used to implement ops
- the *device type* (`B::Device`) you must construct/pass in
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

## Tomorrow: Backend Wrappers (Open Questions)

I was confused about “backend wrappers” like `Autodiff<B>` and `Fusion<B>`: are they just “plumbing”, or do they change the meaning of what the model computes?

Here’s the working mental model to revisit:

- `Autodiff<B>`: adds gradient tracking/backprop on top of a base backend. Forward outputs should match the inner backend (`B`) for the same ops, but you pay extra memory/compute and you can now call `backward()` / inspect grads.
- `Fusion<B>`: changes how ops are executed (kernel fusion) for speed. The intent is “same math”, but you may see tiny floating-point differences (different kernel implementations / operation ordering), and determinism can vary by device/backend.

What to measure early (cheap sanity checks):

1) **Forward equality (tolerance)**: run a tiny forward graph on two backends/wrappers and check outputs are “close enough” (not necessarily bit-identical).
2) **Performance**: run the same small workload a few times and compare rough timings (even `std::time::Instant` is enough at first).
3) **Determinism**: seed where possible; rerun and see if the same backend gives stable outputs run-to-run.

## Code

- `crates/burn_lab/src/bin/03_backends_mental_model.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 03_backends_mental_model`

## References

- Burn Book: <https://burn.dev/book/>
- docs.rs burn: <https://docs.rs/burn>
