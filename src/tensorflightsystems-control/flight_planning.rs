use tensorflow::Tensor;

pub trait FlightPlanning {
    fn plan(&self, input: Tensor<f32>) -> Tensor<f32>;
}

pub struct DummyPlanning;

impl FlightPlanning for DummyPlanning {
    fn plan(&self, input: Tensor<f32>) -> Tensor<f32> {
        input
    }
}
