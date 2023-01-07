use tensorflow::Tensor;

pub trait FlightControl {
    fn control(&self, input: Tensor<f32>) -> Tensor<f32>;
}

pub struct DummyControl;

impl FlightControl for DummyControl {
    fn control(&self, input: Tensor<f32>) -> Tensor<f32> {
        input
    }
}
