use actix_web::{post, web, App, HttpServer, Responder};
use serde::Deserialize;
mod decision_tree_classifier;
mod logistic_regression;
mod random_forest_classifier;

// Define request structure
#[derive(Deserialize)]
struct InputData {
    features: Vec<f64>,
    model_architecture: String,
}

// API endpoint
#[post("/predict")]
async fn predict_handler(input: web::Json<InputData>) -> impl Responder {
    if input.model_architecture == "logistic" {
        let model = logistic_regression::load_logistic_model();
        let prediction = logistic_regression::logistic_prediction(&input.features, &model);
        format!("{}", prediction)
    } else if input.model_architecture == "decisiontree" {
        let model = decision_tree_classifier::load_decision_tree_model().unwrap();
        let prediction =
            decision_tree_classifier::decision_tree_prediction(&input.features, &model);
        format!("{}", prediction)
    } else if input.model_architecture == "randomforest" {
        let model = random_forest_classifier::load_random_forest_model().unwrap();
        let prediction =
            random_forest_classifier::random_forest_prediction(&input.features, &model);
        format!("{}", prediction)
    } else {
        format!(
            "model architecture specified {input_data} not supported",
            input_data = input.model_architecture
        )
    }
}

// Main function
#[actix_web::main]
async fn main() -> std::io::Result<()> {
    println!("running service now");
    HttpServer::new(|| App::new().service(predict_handler))
        // bind to port 0.0.0.0:8080 for docker
        // bind to 127.0.0.1:8080 for local development
        .bind("0.0.0.0:8080")?
        .run()
        .await
}
