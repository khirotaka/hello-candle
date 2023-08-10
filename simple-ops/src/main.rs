use candle_core::{Tensor, Device};

fn main() -> Result<(), candle_core::Error> {
    println!("# 単純なスカラー値同士の足し算");
    let x = Tensor::new(1.0f32, &Device::Cpu)?;
    let y = Tensor::new(2.0f32, &Device::Cpu)?;

    let z = x.add(&y)?;

    println!(
        "{} + {} = {}",
        x.to_scalar::<f32>()?,
        y.to_scalar::<f32>()?,
        z.to_scalar::<f32>()?
    );

    println!("# randn で正規分布から 1028×1028 の2次元配列を生成し、matmul する");
    let a = Tensor::randn(0.0f32, 1.0, (1028, 1028), &Device::Cpu)?;
    let b = Tensor::randn(0.0f32, 1.0, (1028, 1028), &Device::Cpu)?;

    let c = a.matmul(&b)?;
    println!("{}", c);

    Ok(())
}
