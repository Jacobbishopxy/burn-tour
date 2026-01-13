//! Configs + reproducible hyperparams.
//!
//! Why this example exists:
//! - Burn’s `#[derive(Config)]` gives you a serializable hyperparam struct (JSON IO included).
//! - Backends can be seeded (`Backend::seed`) so randomized initialization becomes reproducible.
//! - This is the minimum setup to make “run to run” experiments debuggable.

#[cfg(not(feature = "burn"))]
fn main() {
    eprintln!("This binary requires Burn. Run with:");
    eprintln!("  cargo run -p burn_lab --features burn --bin 06_configs_repro_hparams");
}

#[cfg(feature = "burn")]
fn main() {
    use burn::backend::Autodiff;
    use burn::backend::NdArray;
    use burn::config::Config;
    use burn::module::Module;
    use burn::tensor::Tensor;
    use burn::tensor::backend::Backend;

    type B = Autodiff<NdArray>;
    let device = Default::default();

    let cfg = TinyNetConfig {
        seed: 42,
        d_in: 2,
        d_hidden: 4,
        d_out: 1,
    };

    // Persist hyperparams (so reruns can be compared / reproduced).
    let cfg_path = std::env::temp_dir().join("burn_lab").join("06_tiny_net_config.json");
    cfg.save(&cfg_path).expect("config save failed");
    let cfg_loaded = TinyNetConfig::load(&cfg_path).expect("config load failed");
    println!("cfg_path = {cfg_path:?}");
    println!("cfg_loaded = {cfg_loaded:?}");

    // Repro: if you reseed before initialization, your initial weights should match exactly.
    let x = Tensor::<B, 2>::from_data([[1.0, -1.0]], &device);

    <B as Backend>::seed(&device, cfg_loaded.seed);
    let model_1 = TinyNet::<B>::from_config(&cfg_loaded, &device);
    let y_1 = model_1.forward(x.clone()).to_data();

    <B as Backend>::seed(&device, cfg_loaded.seed);
    let model_2 = TinyNet::<B>::from_config(&cfg_loaded, &device);
    let y_2 = model_2.forward(x.clone()).to_data();

    println!("y_1 = {y_1:?}");
    println!("y_2 = {y_2:?}");
    assert_eq!(y_1, y_2, "same seed + same config should reproduce");

    // Changing the seed should (usually) change initialization.
    let cfg_other = TinyNetConfig {
        seed: 43,
        ..cfg_loaded
    };
    <B as Backend>::seed(&device, cfg_other.seed);
    let model_3 = TinyNet::<B>::from_config(&cfg_other, &device);
    let y_3 = model_3.forward(x).to_data();

    println!("y_3 (different seed) = {y_3:?}");
    assert_ne!(y_1, y_3, "different seed should change outputs");

    // Small extra sanity check: configs are not “models”; they just define how you build them.
    println!("model_1.num_params() = {}", model_1.num_params());
}

#[cfg(feature = "burn")]
#[derive(burn::config::Config, Debug)]
struct TinyNetConfig {
    seed: u64,
    d_in: usize,
    d_hidden: usize,
    d_out: usize,
}

#[cfg(feature = "burn")]
#[derive(burn::module::Module, Debug)]
struct TinyNet<B: burn::tensor::backend::Backend> {
    l1: burn::nn::Linear<B>,
    l2: burn::nn::Linear<B>,
}

#[cfg(feature = "burn")]
impl<B: burn::tensor::backend::Backend> TinyNet<B> {
    fn from_config(cfg: &TinyNetConfig, device: &B::Device) -> Self {
        use burn::nn::LinearConfig;

        // Default Linear initializer is random (KaimingUniform), which is why seeding matters.
        let l1 = LinearConfig::new(cfg.d_in, cfg.d_hidden).init(device);
        let l2 = LinearConfig::new(cfg.d_hidden, cfg.d_out).init(device);
        Self { l1, l2 }
    }

    fn forward(&self, x: burn::tensor::Tensor<B, 2>) -> burn::tensor::Tensor<B, 2> {
        use burn::tensor::activation::relu;
        let h = relu(self.l1.forward(x));
        self.l2.forward(h)
    }
}
