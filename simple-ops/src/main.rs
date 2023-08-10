use candle_core::{Tensor, Device};

fn main() -> Result<(), candle_core::Error> {
    let a = Tensor::randn(0.0f32, 1.0, (1028, 1028), &Device::Cpu)?;
    let b = Tensor::randn(0.0f32, 1.0, (1028, 1028), &Device::Cpu)?;

    let c = a.matmul(&b)?;
    println!("{}", c);

    Ok(())
}
