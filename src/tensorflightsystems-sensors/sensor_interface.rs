use tensorflow::Tensor;

pub trait SensorInterface {
    fn read(&self) -> Tensor<f32>;
    fn write(&mut self, data: Tensor<f32>);
}

pub struct DummySensor {
    pub data: Tensor<f32>,
}

impl SensorInterface for DummySensor {
    fn read(&self) -> Tensor<f32> {
        self.data.clone()
    }

    fn write(&mut self, data: Tensor<f32>) {
        self.data = data;
    }
}
