use ndarray::iter::Windows;
use ndarray::{Array, Array1, Array2, ArrayBase, Dim, ViewRepr, s};
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
            for k in (0..n).step_by(n){
                let stop = k + mini_batch_size as usize;
                let mb = training_data.slice(s![k..stop]);
                mini_batches.push(mb);
            }
        } 
        match test_data{
            Some(data) => {},
            None => {},
        }
    }

    
    pub fn update_mini_batch(&self, mini_batch: ArrayBase<ViewRepr<&f64>, Dim<[usize; 1]>, f64>, eta: f64){

    }
}

fn sigmoid(z: f64) -> f64{
    1.0 / ( 1.0 + (-z).exp())
}


fn main() {
    println!("Hello, world!");
}


