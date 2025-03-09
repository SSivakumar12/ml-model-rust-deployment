use serde_json::Value;
use std::cmp::Ordering::Equal;
use std::collections::HashMap;
use std::error::Error;
use std::fs;

pub fn load_decision_tree_model() -> Result<Value, Box<dyn Error>> {
    // Read the JSON file as a string but doesn't enforce schema checks unlike logistic regression
    // this is primarily due to the complexity of decisiontreeclassifier with the depth being particularly complex
    let data = fs::read_to_string("model_weights/tree_classifier_architecture.json")?;
    let json_value = serde_json::from_str(&data)?;
    return Ok(json_value);
}

fn move_down_tree(features: &HashMap<&str, &f64>, hierarchy: &Value) -> usize {
    if hierarchy.get("left").is_none() || hierarchy.get("right").is_none() {
        if let Some(value) = hierarchy.get("value") {
            if let Some(probs) = value[0].as_array() {
                if let Some((idx, _)) = probs
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.as_f64().partial_cmp(&b.1.as_f64()).unwrap_or(Equal))
                {
                    return idx;
                }
            }
        }
        panic!("Empty list no probabilities");
    } else {
        let threshold = hierarchy["threshold"].as_f64().unwrap();
        let feature_name = hierarchy["feature"].as_str().unwrap();
        if features.get(feature_name).copied().unwrap_or(&0.0) <= &threshold {
            return move_down_tree(features, &hierarchy["left"]);
        } else {
            return move_down_tree(features, &hierarchy["right"]);
        }
    }
}

pub fn decision_tree_prediction(features: &Vec<f64>, model: &Value) -> usize {
    let fields = vec![
        "Age",
        "Sex_female",
        "Sex_male",
        "Pclass_1",
        "Pclass_2",
        "Pclass_3",
    ];
    let features_map: HashMap<_, _> = fields.into_iter().zip(features.into_iter()).collect();
    return move_down_tree(&features_map, model);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decision_tree_prediction() {
        let features = vec![25.0, 1.0, 0.0, 0.0, 0.0, 0.0];
        let model = load_decision_tree_model().unwrap();
        let expected = decision_tree_prediction(&features, &model);
        assert_eq!(expected, 1)
    }
}
