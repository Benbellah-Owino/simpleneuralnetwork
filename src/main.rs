use ndarray::iter::Windows;
use ndarray::{Array, Array1, Array2, ArrayBase, Axis, Dim, ViewRepr, s};
use ndarray_rand::RandomExt;
use ndarray_rand::rand_distr::StandardNormal;
use rand::{Rng, RngExt};
use rand::{seq::SliceRandom, rng}
use rand::rngs::ThreadRng;


/// This is an array of the number of neurons in each layer in order eg [2, 4, 5]
type Sizes = Vec<usize>;
type Biases = Vec<Array1<f64>>;
type Weights = Vec<Array2<f64>>;

// trait Input : Iterator{}
// impl<T:Iterator> Input for T{}
type Input = Array1<f64>;

struct Network {
    pub num_layers: usize,
    pub sizes: Sizes,
    pub biases: Biases,
    pub weights: Weights,
}

impl Network {
    fn new(sizes: Sizes) -> Network {
    
        let biases: Biases = sizes[1..]
            .iter()
            .map(|&y| Array::random(y, StandardNormal))
            .collect();

        let weights: Weights = sizes[..sizes.len() - 1]
            .iter()
            .zip(sizes[1..].iter())
            .map(|(&x, &y)| Array::random((y,x), StandardNormal))
            .collect();
        ;
        Network {
            num_layers: sizes.len(),
            sizes: sizes,
            biases,
            weights
        }
    }

    // pub fn sigmoid(&self, input: Input) -> Input{
    //     let mut output = Vec::new();
    //     for val in input{
    //         let a = 1.0/(1.0 + -val.exp());
    //         output.push(a);
    //     }

    //     return output;
    // }


    pub fn feed_forward(&self , a: Input) -> Input{
        let mut output : Input = a.clone();
        for (b, w) in self.biases.iter().zip(self.weights.iter()){
            output = (w.dot(&a) + b).mapv(sigmoid);
        }

        output
    }


    ///Train the neural network using mini-batch stochastic
    ///gradient descent.  The "training_data" is a list of tuples
    ///"(x, y)" representing the training inputs and the desired
    ///outputs.  The other non-optional parameters are
    ///self-explanatory.  If "test_data" is provided then the
    ///network will be evaluated against the test data after each
    ///epoch, and partial progress printed out.  This is useful for
    ///tracking progress, but slows things down substantially."""
    #[allow(non_snake_case)]
    pub fn SDG(&self,mut training_data: Input,epochs: u16, mini_batch_size: u16, eta: f64, test_data: Option<Input>){
        let n = training_data.len();

        for j in 0..epochs{
            let mut rng = rng();
            training_data.as_slice_mut().unwrap().shuffle(&mut rng);
            let mut mini_batches = Vec::new();
            for k in (0..n).step_by(mini_batch_size as usize){
                let stop = k + mini_batch_size as usize;
                let mb = training_data.slice(s![k..stop]);
                mini_batches.push(mb);
            }
        let test_data = test_data.clone();
        match test_data{
            Some(ref _data) => {
                eprintln!("Epoch {j}")
            },
            None => {
                eprintln!("Epoch {j}")
            },
        }
        } 
    }

    
    pub fn update_mini_batch(&mut self, mini_batch: ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>, f64>, eta: f64){
        // Update the network's weights and biases by applying
        // gradient descent using backpropagation to a single mini batch.
        // The ``mini_batch`` is a list of tuples ``(x, y)``, and ``eta``
        // is the learning rate.
        let mut nabla_b : Biases = self.biases.iter()
            .map(|b| Array1::zeros(b.len()))
            .collect();

        let mut nabla_w: Weights = self.weights.iter()
            .map(|w| Array2::zeros(w.dim()))
            .collect();

        for (x, y) in mini_batch.iter(){
            let (delta_nabla_b, delta_nabla_w) = self.backprop(x, y);
            nabla_b = nabla_b.iter().zip(delta_nabla_b.iter())
                .map(|(nb, dnb)| nb + dnb )
                .collect();

            nabla_w = nabla_w.iter().zip(delta_nabla_w.iter())
                .map(|(nw, dnw)| nw + dnw )
                .collect()
        }

        let m = mini_batch.len() as f64;
        self.weights = self.weights.iter().zip(nabla_w.iter())
            .map(|(w, nw)| w - (eta / m) * nw)
            .collect();

        self.biases = self.biases.iter().zip(nabla_b.iter())
            .map(|(b, nb)| b - (eta / m) * nb)
            .collect();

    }

    fn backprop(&self, x: &Array1<f64>, y: &Array1<f64>) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
    let mut nabla_b: Vec<Array1<f64>> = self.biases.iter()
        .map(|b| Array1::zeros(b.len()))
        .collect();
    let mut nabla_w: Vec<Array2<f64>> = self.weights.iter()
        .map(|w| Array2::zeros(w.dim()))
        .collect();

    // Feedforward
    let mut activation = x.clone();
    let mut activations: Vec<Array1<f64>> = vec![x.clone()];
    let mut zs: Vec<Array1<f64>> = vec![];

    for (b, w) in self.biases.iter().zip(self.weights.iter()) {
        let z = w.dot(&activation) + b;
        zs.push(z.clone());
        activation = sigmoid(&z);
        activations.push(activation.clone());
    }

    // Backward pass
    let last = activations.len() - 1;
    let mut delta = self.cost_derivative(&activations[last], y) * sigmoid_prime(&zs[last]);
    *nabla_b.last_mut().unwrap() = delta.clone();
    *nabla_w.last_mut().unwrap() = outer(&delta, &activations[last - 1]);

    for l in 2..self.num_layers {
        let z = &zs[zs.len() - l];
        let sp = sigmoid_prime(z);
        let w = &self.weights[self.weights.len() - l + 1];
        delta = w.t().dot(&delta) * sp;
        let nl = nabla_b.len();
        nabla_b[nl - l] = delta.clone();
        let na = activations.len();
        nabla_w[nabla_w.len() - l] = outer(&delta, &activations[na - l - 1]);
    }

    (nabla_b, nabla_w)
}


    pub fn evaluate(self, test_data: &[(Array1<f64>, usize)]) -> usize{
        // Return the number of test inputs for which the neural
        // network outputs the correct result. Note that the neural
        // network's output is assumed to be the index of whichever
        // neuron in the final layer has the highest activation.
        test_data.iter()
            .filter(|(x, y)|{
                let output = self.feed_forward(x.to_owned());
                let predicted = output.iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .map(|(i, _)| i)
                .unwrap();

            predicted == *y
            }).count()
    }

    fn cost_derivative(&self, output_activations: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        output_activations - y
    }
}

fn sigmoid(z: f64) -> f64{
    1.0 / ( 1.0 + (-z).exp())
}

fn sigmoid_prime(z: f64) -> f64{
    sigmoid(z) * (1.0 - sigmoid(z))
}

fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let a = a.view().insert_axis(Axis(1)); // column vector
    let b = b.view().insert_axis(Axis(0)); // row vector
    a.dot(&b)
}

fn main() {
    println!("Hello, world!");
}


