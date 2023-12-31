use candle_core::{Device, Tensor, Var};

fn f(x: &Tensor) -> candle_core::Result<Tensor> {
    x.mul(&x)
}

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

    println!("# 自動微分を試してみる");
    let x = Var::new(3.0f32, &Device::Cpu)?;
    // Tensorは値が普遍なものを対象とするのに対し、Varは値が変化する物に使う。
    // PyTorchでいう torch.tensor(..., requires_grad =True) ってやるのと同じものと理解
    let x = x.as_tensor();
    let y = f(x)?;
    println!("x^2 = {}", y.to_scalar::<f32>()?);
    let grads = y.backward()?;
    if let Some(grad_foo) = grads.get(&x) {
        println!("grad for x: {}", grad_foo.to_scalar::<f32>()?);
    } else {
        println!("no grad");
    }

    Ok(())
}
