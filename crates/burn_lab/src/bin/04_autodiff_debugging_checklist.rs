//! Autodiff debugging checklist: “why are my grads missing / zero?”
//!
//! Why this example exists:
//! - Burn autodiff is explicit: you need an `Autodiff<...>` backend *and* leaf tensors marked with `require_grad()`.
//! - It’s easy to accidentally break the graph with `detach()` (or by switching to the inner backend).
//! - This file prints the most common “sanity checks” you want during early experiments.

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This binary requires Burn. Run with:");
    eprintln!("  cargo run -p burn_lab --features burn --bin 04_autodiff_debugging_checklist");
}

#[cfg(feature = "burn")]
fn main() {
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::tensor::Tensor;

    type B = Autodiff<NdArray>;
    let device = Default::default();

    let x = Tensor::<B, 2>::from_data([[1.0, 2.0], [3.0, 4.0]], &device);
    let target = Tensor::<B, 2>::from_data([[1.0], [1.0]], &device);

    println!("B = {}", core::any::type_name::<B>());

    // Case 1: leaf parameter is NOT tracked => `grad(...)` is `None`.
    let w = Tensor::<B, 2>::from_data([[0.1], [0.2]], &device);
    let loss = mse_loss(x.clone().matmul(w.clone()), target.clone());
    let grads = loss.backward();
    print_case("no require_grad()", &w, &grads);

    // Case 2: mark leaf tensor tracked => grads show up.
    let w = w.require_grad();
    let loss = mse_loss(x.clone().matmul(w.clone()), target.clone());
    let grads = loss.backward();
    print_case("require_grad()", &w, &grads);

    // Case 3: detach in the forward pass => graph is cut => grads disappear.
    let pred = x.matmul(w.clone()).detach();
    let loss = mse_loss(pred, target);
    let grads = loss.backward();
    print_case("detach() breaks the graph", &w, &grads);
}

#[cfg(feature = "burn")]
use burn::tensor::Tensor;

#[cfg(feature = "burn")]
fn mse_loss<B>(pred: Tensor<B, 2>, target: Tensor<B, 2>) -> Tensor<B, 1>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    (pred - target).powi_scalar(2).mean()
}

#[cfg(feature = "burn")]
fn print_case<const D: usize, B>(
    label: &str,
    tracked: &Tensor<B, D>,
    grads: &B::Gradients,
) where
    B: burn::tensor::backend::AutodiffBackend,
{
    println!("\n== {label} ==");
    println!("tracked.is_require_grad() = {}", tracked.is_require_grad());

    match tracked.grad(grads) {
        Some(grad) => {
            println!("grad (inner backend) = {grad}");
        }
        None => {
            println!("grad = None");
        }
    }
}
