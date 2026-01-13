//! Training with a custom loop (no `Learner`).
//!
//! Why this example exists:
//! - Understand what `Learner` is doing for you (forward → loss → backward → optimizer step).
//! - Learn where you might hook in custom logic (logging, gradient clipping, custom schedules, etc.).
//! - Make “model moves each step” explicit (Burn optimizers return an updated module).

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This binary requires Burn. Run with:");
    eprintln!("  cargo run -p burn_lab --features burn --bin 08_training_custom_loop");
}

#[cfg(feature = "burn")]
use burn::{data::dataset::Dataset, module::AutodiffModule, tensor::Tensor};

#[cfg(feature = "burn")]
fn main() {
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::data::dataloader::DataLoaderBuilder;
    use burn::optim::AdamConfig;
    use burn::optim::GradientsParams;
    use burn::optim::Optimizer;

    type B = Autodiff<NdArray>;

    let device = Default::default();
    let dataset_train = LinDataset::synthetic(512, 123);
    let dataset_valid = LinDataset::synthetic(128, 456);

    let dataloader_train = DataLoaderBuilder::new(LinBatcher::<B>::new())
        .batch_size(32)
        .shuffle(42)
        .num_workers(0)
        .build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(LinBatcher::<NdArray>::new())
        .batch_size(64)
        .shuffle(43)
        .num_workers(0)
        .build(dataset_valid);

    let mut model = LinReg::<B>::new(&device);
    let mut optim = AdamConfig::new().init();
    let lr = 1e-2;

    for epoch in 0..5 {
        // Training: forward → loss → backward → optimizer step.
        let mut last_loss = None;

        for batch in dataloader_train.iter() {
            let preds = model.forward(batch.x);
            let loss_per_item = (preds - batch.y)
                .powi_scalar(2)
                .mean_dim(1)
                .squeeze_dim::<1>(1);
            let loss = loss_per_item.mean();

            // `backward()` returns a graph-owned gradients object.
            let grads = loss.backward();
            // Map grads to the module parameters expected by optimizers.
            let grads = GradientsParams::from_grads(grads, &model);

            // Burn optimizers return an updated model (no in-place mutation).
            model = optim.step(lr, model, grads);

            last_loss = Some(loss);
        }

        // Validation: use the inner module (no autodiff).
        let model_valid = model.valid();
        let mut valid_losses = Vec::new();

        for batch in dataloader_valid.iter() {
            let preds = model_valid.forward(batch.x);
            let loss = (preds - batch.y)
                .powi_scalar(2)
                .mean_dim(1)
                .squeeze_dim::<1>(1)
                .mean();

            valid_losses.push(loss.into_data().iter::<f64>().next().expect("scalar loss"));
        }

        let train_loss = last_loss
            .map(|t| t.into_data().iter::<f64>().next().unwrap())
            .unwrap_or(f64::NAN);
        let valid_loss = valid_losses.iter().sum::<f64>() / valid_losses.len().max(1) as f64;

        println!("epoch {epoch}: train_loss={train_loss:.4} valid_loss={valid_loss:.4}");
    }
}

#[cfg(feature = "burn")]
#[derive(Clone, Debug)]
struct LinSample {
    x: [f32; 2],
    y: [f32; 1],
}

#[cfg(feature = "burn")]
struct LinDataset {
    samples: Vec<LinSample>,
}

#[cfg(feature = "burn")]
impl LinDataset {
    fn synthetic(n: usize, seed: u64) -> Self {
        let mut samples = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f32;
            let x1 = (t * 0.017 + seed as f32 * 0.001).sin();
            let x2 = (t * 0.031 + seed as f32 * 0.002).cos();
            let noise = (hash01(i as u64, seed) - 0.5) * 0.1;
            let y = 2.0 * x1 - 3.0 * x2 + 0.5 + noise;
            samples.push(LinSample {
                x: [x1, x2],
                y: [y],
            });
        }

        Self { samples }
    }
}

#[cfg(feature = "burn")]
fn hash01(i: u64, seed: u64) -> f32 {
    let mut x = i ^ (seed.wrapping_mul(0x9E3779B97F4A7C15));
    x = x.wrapping_mul(6364136223846793005).wrapping_add(1);
    let top = (x >> 40) as u32;
    (top as f32) / (u32::MAX as f32)
}

#[cfg(feature = "burn")]
impl Dataset<LinSample> for LinDataset {
    fn get(&self, index: usize) -> Option<LinSample> {
        self.samples.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.samples.len()
    }
}

#[cfg(feature = "burn")]
#[derive(Clone)]
struct LinBatcher<B: burn::tensor::backend::Backend> {
    _b: core::marker::PhantomData<B>,
}

#[cfg(feature = "burn")]
impl<B: burn::tensor::backend::Backend> LinBatcher<B> {
    fn new() -> Self {
        Self {
            _b: core::marker::PhantomData,
        }
    }
}

#[cfg(feature = "burn")]
impl<B> burn::data::dataloader::batcher::Batcher<B, LinSample, LinBatch<B>> for LinBatcher<B>
where
    B: burn::tensor::backend::Backend,
{
    fn batch(&self, items: Vec<LinSample>, device: &B::Device) -> LinBatch<B> {
        use burn::tensor::TensorData;

        let batch_size = items.len();
        let mut x_vals = Vec::with_capacity(items.len() * 2);
        let mut y_vals = Vec::with_capacity(items.len());

        for item in items {
            x_vals.extend_from_slice(&item.x);
            y_vals.extend_from_slice(&item.y);
        }

        let x = Tensor::<B, 2>::from_data(TensorData::new(x_vals, [batch_size, 2]), device);
        let y = Tensor::<B, 2>::from_data(TensorData::new(y_vals, [batch_size, 1]), device);

        LinBatch { x, y }
    }
}

#[cfg(feature = "burn")]
#[derive(Clone, Debug)]
struct LinBatch<B: burn::tensor::backend::Backend> {
    x: Tensor<B, 2>,
    y: Tensor<B, 2>,
}

#[cfg(feature = "burn")]
#[derive(burn::module::Module, Debug)]
struct LinReg<B: burn::tensor::backend::Backend> {
    linear: burn::nn::Linear<B>,
}

#[cfg(feature = "burn")]
impl<B: burn::tensor::backend::Backend> LinReg<B> {
    fn new(device: &B::Device) -> Self {
        use burn::nn::LinearConfig;
        let linear = LinearConfig::new(2, 1).init(device);
        Self { linear }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        self.linear.forward(x)
    }
}
