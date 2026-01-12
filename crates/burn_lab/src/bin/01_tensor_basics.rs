//! Minimal tensor operations on a CPU backend.
//!
//! Why this example exists:
//! - Start with a backend that’s easy to run anywhere (no GPU setup required).
//! - Keep the focus on shapes, broadcasting rules, and basic ops.

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This binary requires Burn. Run with:");
    eprintln!("  cargo run -p burn_lab --features burn --bin 01_tensor_basics");
}

#[cfg(feature = "burn")]
fn main() {
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    // Why `NdArray`: simplest “get something working” backend before adding GPU/autodiff.
    type B = NdArray;

    // This backend’s device is just CPU; making it explicit keeps the API consistent across backends.
    let device = Default::default();

    // A small 2D tensor you can reason about when debugging shapes.
    let a = Tensor::<B, 2>::from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], &device);
    println!("a.dims() = {:?}", a.dims());
    println!("a = {a}");

    // Row shape [1, 3] broadcasts over the first dimension (like adding a per-feature bias).
    let row = Tensor::<B, 2>::from_data([[10.0, 20.0, 30.0]], &device);
    let b = a.clone() + row;
    println!("\nBroadcast add (row):");
    println!("b = {b}");

    // Column shape [2, 1] broadcasts over the second dimension (common gotcha when dims don’t align).
    let col = Tensor::<B, 2>::from_data([[100.0], [200.0]], &device);
    let c = a.clone() + col;
    println!("\nBroadcast add (col):");
    println!("c = {c}");

    // Reshape is a frequent source of bugs; print `dims()` early while learning.
    let r = a.clone().reshape([3, 2]);
    println!("\nReshape a -> [3, 2]:");
    println!("r.dims() = {:?}", r.dims());
    println!("r = {r}");

    // `r @ r.T` produces a square Gram-like matrix; useful for sanity-checking matmul shapes.
    let m = r.clone().matmul(r.transpose());
    println!("\nMatmul r @ r.T:");
    println!("m.dims() = {:?}", m.dims());
    println!("m = {m}");

    // Many reductions return a 1D tensor; treat it as a scalar container when printing/logging.
    let mean = m.mean();
    println!("\nMean(m) over all elements:");
    println!("mean.dims() = {:?}", mean.dims());
    println!("mean = {mean}");
}
