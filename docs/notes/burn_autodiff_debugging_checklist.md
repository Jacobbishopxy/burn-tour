# Burn Autodiff Debugging Checklist

## Goal

Have a quick, repeatable checklist for diagnosing “my gradients are missing/zero”.

## Checklist (Practical)

When grads are `None` (or unexpectedly zero), check:

1. **You’re using an autodiff backend**
   - Use `type B = Autodiff<...>` (not just `NdArray`, `Wgpu`, etc.)

2. **Leaf tensors you care about are tracked**
   - Call `require_grad()` on parameters/inputs you want gradients for

3. **You didn’t cut the graph**
   - `detach()` intentionally breaks gradient flow
   - Converting to the inner backend (`tensor.inner()`) also removes autodiff info

4. **You call `backward()` from the right place**
   - Backprop from a scalar loss (or a reduced tensor) so the gradient seed is well-defined

5. **You inspect grads the Burn way**
   - `let grads = loss.backward();`
   - `tensor.grad(&grads)` for non-destructive inspection, or `tensor.grad_remove(&mut grads)` for one-time use

## Graph Sketches

These tiny graphs show where gradient flow exists (or is cut) in each case:

```text
Case 1: no require_grad()
x ------------------> matmul ----> pred ----> mse ----> loss
w (leaf, no grad) --^

Case 2: require_grad()
x ------------------> matmul ----> pred ----> mse ----> loss
w (leaf, tracked) --^

Case 3: detach() breaks the graph
x ------------------> matmul ----> pred -- detach --> pred_detached --> mse --> loss
w (leaf, tracked) --^               (graph cut)
```

## Code

- `crates/burn_lab/src/bin/04_autodiff_debugging_checklist.rs`

## How to Run

- `cargo run -p burn_lab --features burn --bin 04_autodiff_debugging_checklist`
