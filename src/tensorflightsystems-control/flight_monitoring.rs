use tensorflow::Tensor;

pub trait FlightMonitoring {
    fn monitor(&self, input: Tensor<f32>) -> Tensor<f32>;
}

pub struct DummyMonitoring;

impl FlightMonitoring for DummyMonitoring {
    fn monitor(&self, input: Tensor<f32>) -> Tensor<f32> {
        input
    }
}
