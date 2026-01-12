//! Device placement and transfers with the WGPU backend.
//!
//! Why this example exists:
//! - Make “where does this tensor live?” explicit (host vs accelerator).
//! - Practice `to_device` so you recognize when you’re paying a copy/transfer.

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This binary requires Burn. Run with:");
    eprintln!("  cargo run -p burn_lab --features burn --bin 02_device_movement");
}

#[cfg(feature = "burn")]
fn main() {
    use burn::backend::Wgpu;
    use burn::backend::wgpu::WgpuDevice;
    use burn::tensor::Tensor;

    // WGPU backend can target multiple graphics APIs; `BestAvailable` picks a reasonable default.
    type B = Wgpu;

    // Start from CPU so this example still runs on machines without a discrete GPU.
    let from = WgpuDevice::Cpu;
    // `DefaultDevice` respects wgpu’s “high power” preference and can be overridden via env vars.
    let to = WgpuDevice::DefaultDevice;

    // Create on `from` first; it keeps the “data origin” obvious when you debug transfers.
    let x = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &from);
    println!("x.device() = {:?}", x.device());
    println!("x = {x}");

    // `to_device` allocates + copies; avoid doing this inside tight training loops.
    let y = x.to_device(&to);
    println!("\nMoved to: {to:?}");
    println!("y.device() = {:?}", y.device());
    println!("y = {y}");

    // Burn tensor ops often take `self` by value, so reusing a tensor may require `clone()`.
    // Here we compute `y @ y.T` and then move the result back to CPU for inspection.
    let y_t = y.clone().transpose();
    let z = (y.clone().matmul(y_t) + 1.0).to_device(&from);
    println!("\nComputed on '{to:?}' then moved back to CPU:");
    println!("z.device() = {:?}", z.device());
    println!("z = {z}");
}
