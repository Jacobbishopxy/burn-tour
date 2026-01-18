//! A practical “backend mental model” for Burn.
//!
//! Why this example exists:
//! - A Burn `Backend` is a *compile-time choice* that determines tensor storage + kernels + device type.
//! - The `Device` type is part of the backend; that’s why “switching backends” often changes device setup.
//! - Keeping this straight makes it easier to reason about portability (CPU vs GPU) and performance.

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This binary requires Burn. Run with:");
    eprintln!("  cargo run -p burn_lab --features burn --bin 03_backends_mental_model");
}

#[cfg(feature = "burn")]
fn main() {
    use burn::backend::wgpu::WgpuDevice;
    use burn::backend::{Autodiff, NdArray, Wgpu};
    use burn::tensor::Tensor;
    use burn::tensor::backend::{AutodiffBackend, Backend};
    use std::time::Instant;

    fn demo_backend<B>(label: &str, device: &B::Device)
    where
        B: Backend,
        B::Device: core::fmt::Debug,
    {
        println!("\n== {label} ==");
        println!("B = {}", core::any::type_name::<B>());
        println!("B::Device = {}", core::any::type_name::<B::Device>());

        let x = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device);
        println!("x.device() = {:?}", x.device());
        println!("x.dims() = {:?}", x.dims());
        println!("x = {x}");

        let y = (x.clone().matmul(x.clone().transpose()) + 1.0).mean();
        println!("y.dims() = {:?}", y.dims());
        println!("y = {y}");
    }

    fn timed_forward<B>(device: &B::Device, iters: usize) -> (f32, std::time::Duration)
    where
        B: Backend,
    {
        let start = Instant::now();
        let mut out = 0.0;

        for _ in 0..iters {
            let x = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device);
            let y = (x.clone().matmul(x.transpose()) + 1.0).mean();
            out = y.to_data().iter::<f32>().next().expect("scalar output");
        }

        B::sync(device);

        (out, start.elapsed())
    }

    fn timed_forward_autodiff<B>(device: &B::Device, iters: usize) -> (f32, std::time::Duration)
    where
        B: AutodiffBackend,
    {
        let start = Instant::now();
        let mut out = 0.0;

        for _ in 0..iters {
            let x = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device).require_grad();
            let y = (x.clone().matmul(x.transpose()) + 1.0).mean();
            out = y.to_data().iter::<f32>().next().expect("scalar output");
        }

        B::sync(device);

        (out, start.elapsed())
    }

    fn grad_sample<B>(device: &B::Device) -> f32
    where
        B: AutodiffBackend,
    {
        let x = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], device).require_grad();
        let y = (x.clone().matmul(x.clone().transpose()) + 1.0).mean();
        let grads = y.backward();
        let grad = x.grad(&grads).expect("grad should exist");

        grad.to_data().iter::<f32>().next().expect("grad sample")
    }

    // Backend choice #1: pure CPU with minimal setup.
    type CpuBackend = NdArray;
    let cpu = Default::default();
    demo_backend::<CpuBackend>("NdArray (CPU)", &cpu);

    // Backend choice #2: WGPU. Even when you *plan* to run on GPU, it helps to start on CPU:
    // - keeps demos runnable on machines without a discrete GPU
    // - makes transfers explicit so you notice when you’re paying for copies
    type GpuBackend = Wgpu;
    let wgpu_cpu = WgpuDevice::Cpu;
    demo_backend::<GpuBackend>("Wgpu (CPU device)", &wgpu_cpu);

    // Moving to another device is an allocation + copy.
    let wgpu_gpu = WgpuDevice::DefaultDevice;
    let x = Tensor::<GpuBackend, 2>::from_data([[10.0, 20.0], [30.0, 40.0]], &wgpu_cpu);
    let x = x.to_device(&wgpu_gpu);
    println!("\n== Wgpu transfer demo ==");
    println!("Moved tensor to: {wgpu_gpu:?}");
    println!("x.device() = {:?}", x.device());
    println!("x = {x}");

    println!("\n== Wrapper sanity check (Autodiff vs Fusion) ==");
    let iters = 100;
    let (fusion_out, fusion_time) = timed_forward::<GpuBackend>(&wgpu_cpu, iters);
    let (autodiff_out, autodiff_time) =
        timed_forward_autodiff::<Autodiff<GpuBackend>>(&wgpu_cpu, iters);
    let grad_sample = grad_sample::<Autodiff<GpuBackend>>(&wgpu_cpu);
    println!("iters = {iters}");
    println!("fusion output = {fusion_out}");
    println!("autodiff output = {autodiff_out}");
    println!("abs diff = {}", (fusion_out - autodiff_out).abs());
    println!("fusion time = {fusion_time:?}");
    println!("autodiff time = {autodiff_time:?}");
    println!("autodiff grad sample = {grad_sample}");
    println!("note: timings include host reads + sync, so treat as ballpark.");
}
