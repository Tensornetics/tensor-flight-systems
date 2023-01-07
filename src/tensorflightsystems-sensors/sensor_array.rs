use ndarray::{Array, Array2};
use tensorflow::Tensor;

pub struct SensorArray {
    pub data: Array2<f32>,
}

impl SensorArray {
    pub fn new(data: Array2<f32>) -> SensorArray {
        SensorArray { data }
    }

    pub fn to_tensor(&self) -> Tensor<f32> {
        let data = self.data.as_slice().unwrap();
        Tensor::new(&self.data.dim()).with_values(data).unwrap()
    }
}
