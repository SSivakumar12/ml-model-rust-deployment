use serde_json::Value;
use std::error::Error;
use std::fs;

pub fn load_decision_tree_model() -> Result<Value, Box<dyn Error>> {
    // Read the JSON file as a string but doesn't enforce schema checks unlike logistic regression
    // this is primarily due to the complexity of decisiontreeclassifier with the depth being particularly complex
    let data = fs::read_to_string("model_weights/tree_classifier_architecture.json")?;
    let json_value: Value = serde_json::from_str(&data)?;
    Ok(json_value)
}
