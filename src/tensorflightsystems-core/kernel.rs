use ndarray::Array;
use tensorflow::{Graph, Session, Tensor};

pub struct TensorFlightSystem {
    graph: Graph,
    session: Session,
}

impl TensorFlightSystem {
    pub fn new() -> TensorFlightSystem {
        let mut graph = Graph::new();
        let mut session = Session::new(&SessionOptions::new(), &graph).unwrap();
        TensorFlightSystem { graph, session }
    }

    pub fn run_inference(&mut self, input: &Array<f32>) -> Array<f32> {
        let input_tensor = Tensor::new(&[input.len() as u64, 1]).with_values(input).unwrap();
        let mut output_tensor = Tensor::new(&[1]);
        self.session.run(&[("input", &input_tensor), ("output", &mut output_tensor)], &[]).unwrap();
        output_tensor.to_array::<f32>().unwrap()
    }
}
