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
    use burn::backend::NdArray;
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::Tensor;
    use burn::tensor::backend::Backend;

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

        let y = (x.clone().matmul(x.transpose()) + 1.0).mean();
        println!("y.dims() = {:?}", y.dims());
        println!("y = {y}");
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
}

