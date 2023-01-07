use tensorflow::Tensor;

pub trait SensorProcessing {
    fn process(&self, input: Tensor<f32>) -> Tensor<f32>;
}

pub struct DummyProcessing;

impl SensorProcessing for DummyProcessing {
    fn process(&self, input: Tensor<f32>) -> Tensor<f32> {
        input
    }
}
