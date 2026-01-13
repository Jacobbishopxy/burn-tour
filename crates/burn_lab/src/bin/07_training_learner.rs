//! Training with the high-level `Learner`.
//!
//! Why this example exists:
//! - `LearnerBuilder` wires together: model + optimizer + LR schedule + metrics + checkpoints.
//! - You implement `TrainStep` / `ValidStep` once, then reuse the training infra.
//! - This is a good default until you need custom training loops.

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This binary requires Burn. Run with:");
    eprintln!("  cargo run -p burn_lab --features burn --bin 07_training_learner");
}

#[cfg(feature = "burn")]
use burn::{data::dataset::Dataset, tensor::Tensor};

#[cfg(feature = "burn")]
fn main() {
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::data::dataloader::DataLoaderBuilder;
    use burn::optim::AdamConfig;
    use burn::tensor::Tensor;
    use burn::train::metric::LossMetric;
    use burn::train::LearnerBuilder;

    type TrainB = Autodiff<NdArray>;
    type ValidB = NdArray;

    let device_train = Default::default();
    let device_valid = Default::default();

    let dataset_train = LinDataset::synthetic(512, 123);
    let dataset_valid = LinDataset::synthetic(128, 456);

    let batcher_train = LinBatcher::<TrainB>::new();
    let batcher_valid = LinBatcher::<ValidB>::new();

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(32)
        .shuffle(42)
        .num_workers(0)
        .build(dataset_train);
    let dataloader_valid = DataLoaderBuilder::new(batcher_valid)
        .batch_size(64)
        .shuffle(43)
        .num_workers(0)
        .build(dataset_valid);

    let model = LinReg::<TrainB>::new(&device_train);
    let optim = AdamConfig::new().init();

    // `Learner` writes metrics/checkpoints under this directory.
    let dir = std::path::PathBuf::from("target/burn_lab/07_training_learner");

    let learner = LearnerBuilder::new(dir)
        .num_epochs(5)
        .metric_train_numeric(LossMetric::<ValidB>::new())
        .metric_valid_numeric(LossMetric::<ValidB>::new())
        .summary()
        .build(model, optim, 1e-2);

    let trained = learner.fit(dataloader_train, dataloader_valid);

    // `trained.model` is on the *valid backend* (no autodiff) and ready for inference.
    let x = Tensor::<ValidB, 2>::from_data([[0.5, -1.0]], &device_valid);
    let y = trained.model.forward(x).to_data();
    println!("example prediction y = {y:?}");
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
        // Ground-truth: y = 2*x1 - 3*x2 + 0.5 + noise
        let mut samples = Vec::with_capacity(n);

        for i in 0..n {
            let t = i as f32;
            // Deterministic “random-ish” inputs without an extra `rand` dependency.
            let x1 = (t * 0.017 + seed as f32 * 0.001).sin();
            let x2 = (t * 0.031 + seed as f32 * 0.002).cos();
            let noise = (hash01(i as u64, seed) - 0.5) * 0.1;
            let y = 2.0 * x1 - 3.0 * x2 + 0.5 + noise;
            samples.push(LinSample { x: [x1, x2], y: [y] });
        }

        Self { samples }
    }
}

#[cfg(feature = "burn")]
fn hash01(i: u64, seed: u64) -> f32 {
    // Simple LCG-ish hash for stable noise in [0, 1).
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

#[cfg(feature = "burn")]
impl<B> burn::train::TrainStep<LinBatch<B>, burn::train::RegressionOutput<B>> for LinReg<B>
where
    B: burn::tensor::backend::AutodiffBackend,
{
    fn step(&self, item: LinBatch<B>) -> burn::train::TrainOutput<burn::train::RegressionOutput<B>> {
        let preds = self.forward(item.x);
        let loss_per_item = (preds.clone() - item.y.clone())
            .powi_scalar(2)
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        let loss = loss_per_item.clone().mean();
        let grads = loss.backward();

        let out = burn::train::RegressionOutput::new(loss_per_item, preds, item.y);
        burn::train::TrainOutput::new(self, grads, out)
    }
}

#[cfg(feature = "burn")]
impl<B> burn::train::ValidStep<LinBatch<B>, burn::train::RegressionOutput<B>> for LinReg<B>
where
    B: burn::tensor::backend::Backend,
{
    fn step(&self, item: LinBatch<B>) -> burn::train::RegressionOutput<B> {
        let preds = self.forward(item.x);
        let loss_per_item = (preds.clone() - item.y.clone())
            .powi_scalar(2)
            .mean_dim(1)
            .squeeze_dim::<1>(1);
        burn::train::RegressionOutput::new(loss_per_item, preds, item.y)
    }
}
