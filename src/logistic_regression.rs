use serde::{Deserialize, Serialize};
use std::fs;

// Define model parameters
#[derive(Serialize, Deserialize)]
pub struct LogisticRegressionModel {
    weight: Vec<f64>,
    bias: Vec<f64>,
}

// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub fn load_logistic_model() -> LogisticRegressionModel {
    // Load model parameters from file
    let data = fs::read_to_string("model_weights/logistic_regression_architecture.json")
        .expect("Failed to read JSON file");
    return serde_json::from_str(&data).expect("Failed to parse JSON");
}

pub fn logistic_prediction(features: &Vec<f64>, model: &LogisticRegressionModel) -> f64 {
    // predicts the probability of surival
    // if greater than 0.5 assume survival and vice versa death :(
    let z: f64 = features
        .iter()
        .zip(model.weight.iter())
        .map(|(x, w)| x * w)
        .sum::<f64>()
        + model.bias[0];
    if 0.5 < sigmoid(z) {
        return 1.0;
    } else {
        return 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        // approximate testing whether the sigmoid function is working as expected
        // rounding output to 2dp
        let actual = sigmoid(0.92);
        let rounded_actual = (actual * 100.0).round() / 100.0;
        assert_eq!(rounded_actual, 0.72)
    }

    #[test]
    fn test_logistic_prediction() {
        let features = vec![25.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let model = LogisticRegressionModel {
            weight: vec![
                -0.03914839850205495,
                1.149281815000846,
                -1.1499821391157823,
                1.1996213235549225,
                0.007154342389428081,
                -1.207475990059137,
            ],
            bias: vec![1.2122577308090265],
        };
        let expected = logistic_prediction(&features, &model);
        assert_eq!(expected, 1.0)
    }
}
