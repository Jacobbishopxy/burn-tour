# Burn Tensor Basics (Tensors + Shapes + Devices)

## Goal

Build comfort with Burn's `Tensor<B, D>` API:

- creating tensors
- reshaping/broadcasting
- basic ops (matmul, reductions)
- moving tensors across `Device`s (CPU/GPU)

## Key Concepts

- `Tensor<B, D>` is generic over a `Backend` (`B`) and a dimension (`D`).
- Shapes are part of the type (`D`), but actual sizes are runtime values.
- `Device` selection determines where computation happens.
- “Moving” a tensor typically means allocating/copying it onto another device.

## Common Backends (What `B` Can Be)

- `NdArray<...>`: CPU backend; simplest to start with and highly portable.
- `Wgpu<...>`: GPU backend via wgpu/WebGPU; cross-platform GPU compute (device often selected with `WgpuDevice`).
- `Candle<F = f32, I = i64>`: backend powered by `candle_core`; runs on `CandleDevice::{Cpu, Cuda(..), Metal(..)}` depending on platform/build.
- `LibTorch<...>` (`tch`): backend that uses LibTorch (PyTorch C++); good interoperability with the PyTorch ecosystem.
- `CudaJit` / `HipJit`: JIT backends targeting CUDA / ROCm (when supported).

Backend “decorators” wrap another backend:

- `Autodiff<B>`: adds gradient tracking/training capability on top of a base backend `B`.
- `Fusion<B>`: fuses sequences of tensor ops for performance on supported backends.

Other backends you may encounter:

- `Router`: routes ops (e.g., across devices/backends) behind one `Backend`.
- `RemoteBackend`: executes tensor ops via a remote server process.

## Pitfalls (Finance + ML)

- Shape mismatches show up at runtime; keep prints/asserts while learning.
- Autodiff vs inference: use an autodiff backend only when you need gradients.
- CPU/GPU parity: expect small numeric differences; prefer invariants/tests with tolerances.

## References

- Burn Book: <https://burn.dev/book/>
- docs.rs burn: <https://docs.rs/burn>

## Code

- `crates/burn_lab/src/bin/01_tensor_basics.rs`
- `crates/burn_lab/src/bin/02_device_movement.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 01_tensor_basics`
- `cargo run -p burn_lab --features burn --bin 02_device_movement`
