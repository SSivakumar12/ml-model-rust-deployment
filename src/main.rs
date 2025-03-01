use actix_web::{post, web, App, HttpServer, Responder};
use serde::{Deserialize, Serialize};
use std::fs;

// Define request structure
#[derive(Deserialize)]
struct InputData {
    features: Vec<f64>,
}

// Define model parameters
#[derive(Serialize, Deserialize)]
struct Model {
    weight: Vec<f64>,
    bias: Vec<f64>,
}

// Sigmoid function
fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// Load model parameters from file
fn load_model() -> Model {
    let data = fs::read_to_string("logistic_regression_architecture.json").expect("Failed to read model file");
    serde_json::from_str(&data).expect("Failed to parse JSON")
}

fn predict(features: &Vec<f64>, model: &Model) -> f64 {
    // predicts the probability of surival
    // if greater than 0.5 assume survival and vice versa death :(
    let z: f64 = features.iter().zip(model.weight.iter())
                         .map(|(x, w)| x * w)
                         .sum::<f64>() + model.bias[0];
    if 0.5 < sigmoid(z) {
        return 1.0
    } else {
        return 0.0
    }
    
}

// API endpoint
#[post("/predict")]
async fn predict_handler(input: web::Json<InputData>) -> impl Responder {
    let model = load_model();
    let probability = predict(&input.features, &model);
    format!("Probability: {:.4}", probability)
}

// Main function
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("running service now");
    HttpServer::new(|| App::new().service(predict_handler))
        .bind("127.0.0.1:8080")?
        .run()
        .await
}



