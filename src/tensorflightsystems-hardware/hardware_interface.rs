use tensorflow::Tensor;

pub trait HardwareInterface {
    fn interface(&self, input: Tensor<f32>) -> Tensor<f32>;
}

pub struct DummyInterface;

impl HardwareInterface for DummyInterface {
    fn interface(&self, input: Tensor<f32>) -> Tensor<f32> {
        input
    }
}
